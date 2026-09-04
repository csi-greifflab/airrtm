import numpy as np
import torch

from airrtm.types import AIRRTM_ModelOutput, AIRRTM_ModelTarget


eps = 1e-12


class CompositeLoss(torch.nn.Module):
    """A composite loss for an AIRRTM model. Consists of four terms:

      - TM likelihood: the likelihood of observing a given sequence in a given repertoire.
      - Label likelihood: the likelihood of the predicted repertoire label (repeated for each sequence).
      - Reconstruction loss: the VAE reconstruction loss.
      - KL divergence: the VAE KullbackLeibler divergence.
    The final loss is calculated as:
      vae_coef * (reconstruction_loss_coef * reconstruction_loss + (1 - reconstruction_loss_coef) * kl_devergence)
      (1 - vae_coef) * (tm_likelihood_coef * tm_likelihood + (1 - tm_likelihood_coef) * label_likelihood) +

    Suffix notation
    ---------------
    S: sequence in the dataset
    P: position in the sequence
    A: index of the amino acid
    L: latent space dimension
    T: topic
    R: repertoire
    """

    def __init__(
        self,
        vae_coef: float,
        tm_likelihood_coef: float,
        positive_class_weight: float,
        reconstruction_loss_coef: float,
        topic_l1_coef: float,
        n_sequences_per_repertoire: int,
        return_individual_compotents: bool = True,
    ):
        super().__init__()
        self.vae_coef = vae_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.kld_coef = 1 - self.reconstruction_loss_coef

        self.tm_likelihood_coef = tm_likelihood_coef
        self.label_likelihood_coef = 1 - self.tm_likelihood_coef

        self.positive_class_weight = positive_class_weight

        self.topic_l1_coef = topic_l1_coef

        self.n_sequences_per_repertoire = n_sequences_per_repertoire

        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.return_individual_compotents = return_individual_compotents

    def forward(
        self,
        inputs: AIRRTM_ModelOutput,
        targets: AIRRTM_ModelTarget,
        tau: float,
    ):
        # We average the loss by sequences within one repertoire but sum over repertoires

        tm_loss = -(
            targets["sequence_repertoire_indicators"]
            * torch.log(inputs["tm_likelihoods"])
            + (1 - targets["sequence_repertoire_indicators"])
            * torch.log(1 - inputs["tm_likelihoods"])
        ).mean()

        label_loss_positive = 0.0
        label_loss_negative = 0.0
        is_positive = targets["sequence_repertoire_labels"] == 1
        label_predictions_pos_RS = inputs["label_likelihoods"][is_positive].view(
            -1, self.n_sequences_per_repertoire
        )
        n_pos_repertoires = is_positive.sum() / self.n_sequences_per_repertoire
        label_predictions_neg_RS = inputs["label_likelihoods"][~is_positive].view(
            -1, self.n_sequences_per_repertoire
        )
        n_neg_repertoires = (~is_positive).sum() / self.n_sequences_per_repertoire
        n_positives = is_positive.sum()
        label_accuracy = 0
        if n_positives > 0:
            label_loss_positive += (
                -torch.log(
                    torch.nn.functional.sigmoid(
                        (
                            torch.logsumexp(tau * label_predictions_pos_RS, dim=1)
                            - np.log(self.n_sequences_per_repertoire)
                        )
                        / tau
                    )
                ).sum()
                # * self.positive_class_weight
                # / n_pos_repertoires
            )
            # label_loss_positive += (
            #     # -torch.log(label_predictions_pos_RS.mean(dim=1)).sum()
            #     -torch.log(label_predictions_pos_RS.max(dim=1).values).sum()
            #     * self.positive_class_weight
            #     / n_pos_repertoires
            # )
            label_accuracy += (
                (label_predictions_pos_RS.mean(dim=1) >= 0.5).to(torch.float64).mean()
            )
        if n_positives < targets["sequence_repertoire_labels"].shape[0]:
            label_loss_negative += (
                -torch.log(
                    1
                    - torch.nn.functional.sigmoid(
                        (
                            torch.logsumexp(tau * label_predictions_neg_RS, dim=1)
                            - np.log(self.n_sequences_per_repertoire)
                        )
                        / tau
                    )
                ).sum()
                # * (1 - self.positive_class_weight)
                # / n_neg_repertoires
            )
            # label_loss_negative += (
            #     # -torch.log((1 - label_predictions_neg_RS.mean(dim=1))).sum()
            #     -torch.log((1 - label_predictions_neg_RS.max(dim=1).values)).sum()
            #     * (1 - self.positive_class_weight)
            #     / (n_neg_repertoires)
            # )
            label_accuracy += (
                (label_predictions_neg_RS.mean(dim=1) < 0.5).to(torch.float64).mean()
            )
        label_accuracy /= 2
        label_loss = label_loss_positive + label_loss_negative

        topic_l1_reg = (
            torch.abs(torch.nn.functional.sigmoid(inputs["label_likelihoods"])).sum()
            / self.n_sequences_per_repertoire
        )

        # label_loss = -(
        #     targets["sequence_repertoire_labels"]
        #     * torch.log(inputs["label_likelihoods"])
        #     + (1 - targets["sequence_repertoire_labels"])
        #     * torch.log(1 - inputs["label_likelihoods"])
        # ).mean()  # TODO: check dimensions for the multi-label case (should be sum by dim=1 then mean by dim=0)

        symbol_predictions_SPA = inputs["decoded_sequences"]
        symbol_predictions_SAP = symbol_predictions_SPA.permute(0, 2, 1)
        true_symbol_indices_SP = targets["sequences"]
        reconstruction_loss = self.ce_loss(
            symbol_predictions_SAP, true_symbol_indices_SP
        )
        if self.return_individual_compotents:
            reconstruction_accuracy = (
                (torch.argmax(symbol_predictions_SPA, dim=2) == true_symbol_indices_SP)
                .to(torch.float32)
                .mean()
            )

        kl_divergence = inputs["kl_divergences"].sum() / self.n_sequences_per_repertoire

        vae_loss = (
            self.reconstruction_loss_coef * reconstruction_loss
            + self.kld_coef * kl_divergence
        )
        non_vae_loss = (
            self.tm_likelihood_coef * tm_loss
            + self.label_likelihood_coef * label_loss
            + self.topic_l1_coef * topic_l1_reg
        )
        total_loss = self.vae_coef * vae_loss + (1 - self.vae_coef) * non_vae_loss
        if self.return_individual_compotents:
            return {
                "total_loss": total_loss,
                "reconstruction_accuracy": reconstruction_accuracy,
                "kl_divergence": kl_divergence,
                # "tm_loss": tm_loss,
                "tm_loss": (tm_loss + torch.log(torch.tensor(0.5))) * 1e3,
                "label_accuracy": label_loss,
                # "label_accuracy": (
                #     label_loss_positive / self.positive_class_weight
                #     + label_loss_negative / (1 - self.positive_class_weight)
                # ),
                "topic_l1_reg": topic_l1_reg,
            }
            # return {
            #     "total_loss": total_loss,
            #     "reconstruction_accuracy": reconstruction_loss,
            #     "kl_divergence": kl_divergence,
            #     "tm_loss": tm_loss,
            #     "label_accuracy": label_loss,
            #     "topic_l1_reg": topic_l1_reg,
            # }
        else:
            return {"total_loss": total_loss}
