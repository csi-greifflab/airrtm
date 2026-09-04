import torch

from airrtm.types import AIRRTM_ModelOutput


class AIRRTM_Model(torch.nn.Module):
    """
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
        n_repertoires: int,
        latent_dim: int,
        n_topics_signal: int,
        n_topics_nonsignal: int,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super().__init__()

        self.n_repertoires = n_repertoires
        self.latent_dim = latent_dim
        self.n_topics_signal = n_topics_signal
        self.n_topics_nonsignal = n_topics_nonsignal
        self.n_topics = self.n_topics_signal + self.n_topics_nonsignal

        self.repertoire_topic_proportions = torch.nn.Embedding(
            num_embeddings=self.n_repertoires, embedding_dim=self.n_topics
        )
        self.encoder = encoder
        self.decoder = decoder

        self.z_mean_layer = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=self.latent_dim,
            bias=True,
        )
        self.z_log_sigma_param = torch.nn.Parameter(
            data=torch.Tensor([1.0]), requires_grad=True
        )

        self.z_batch_norm = torch.nn.BatchNorm1d(num_features=self.latent_dim)

        self.latent_space_to_topic_proportions_layer = torch.nn.Linear(
            in_features=self.latent_dim,
            out_features=self.n_topics,
            bias=True,
        )
        self.repertoire_label_prediction_layer = torch.nn.Linear(
            # in_features=self.latent_dim,
            in_features=self.n_topics_signal,
            out_features=1,
            bias=True,
        )

    def forward(
        self,
        repertoire_ids: torch.Tensor,
        x_sequence_SP: torch.Tensor,
    ) -> AIRRTM_ModelOutput:
        topic_proportions_ST = self.get_topic_proportion_matrix()[repertoire_ids]

        z_mean_SL = self.sequence_to_latent(x_sequence_SP)

        # print(z_mean_SL.shape)
        # raise Exception
        # z_mean_SL = torch.nn.functional.relu(z_mean_SL)
        z_SL, z_log_sigma_SL = self._sample_latent(z_mean_SL, self.z_log_sigma_param)
        kl_divergence = (
            -0.5
            * (1 + z_log_sigma_SL - z_mean_SL**2 - torch.exp(z_log_sigma_SL)).sum(
                axis=1
            )
            / self.latent_dim
        )

        seq_topic_logits_ST = self.latent_space_to_topic_proportions_layer(z_SL)
        # l1_topic_layer_reg = torch.abs(seq_topic_logits_ST).sum(dim=1)
        seq_topic_probabilities_ST = torch.nn.functional.sigmoid(seq_topic_logits_ST)
        seq_total_probability_S = (
            topic_proportions_ST * seq_topic_probabilities_ST
        ).sum(dim=1)

        # signal_topic_proportions_ST = topic_proportions_ST[:, : self.n_topics_signal]
        # repertoire_label_prediction_S = self.repertoire_label_prediction_layer(
        #     signal_topic_proportions_ST
        # ).flatten()
        seq_signal_topic_probabilities_ST = seq_topic_probabilities_ST[
            :, : self.n_topics_signal
        ]
        seq_label_prediction_S = self.repertoire_label_prediction_layer(
            seq_signal_topic_probabilities_ST
        ).flatten()
        # seq_label_prediction_S = self.repertoire_label_prediction_layer(z_SL).flatten()

        # l1_label = torch.abs(seq_label_prediction_S)
        # seq_label_prediction_S = torch.nn.functional.sigmoid(seq_label_prediction_S)

        decoded_x_SPA = self.latent_to_sequence(z_SL)
        # decoded_x_SPA = self.latent_to_sequence(z_mean_SL)

        return AIRRTM_ModelOutput(
            # l1_topic_layer_reg=l1_label,
            kl_divergences=kl_divergence,
            tm_likelihoods=seq_total_probability_S,
            label_likelihoods=seq_label_prediction_S,
            decoded_sequences=decoded_x_SPA,
        )

    def _sample_latent(
        self, z_mean: torch.Tensor, z_log_sigma_param: torch.Tensor
    ) -> torch.Tensor:
        device = z_mean.device
        z_log_sigma = torch.ones_like(z_mean).to(device) * z_log_sigma_param
        eps = torch.randn((z_mean.shape[0], z_mean.shape[1])).to(device)
        return z_mean + eps * torch.exp(z_log_sigma * 0.5), z_log_sigma

    def get_topic_proportion_matrix(self) -> torch.Tensor:
        topic_logits = self.repertoire_topic_proportions(
            torch.arange(self.n_repertoires).to(self._get_device())
        )
        topic_proportions = torch.nn.functional.softmax(topic_logits, dim=1)
        return topic_proportions

    def sequence_to_latent(self, x_sequence_SP: torch.Tensor) -> torch.Tensor:
        batch_size = x_sequence_SP.shape[0]
        x_SPE = self.encoder(x_sequence_SP)
        # return x_SPE
        # hidden_SH = (
        #     hidden_SPH[0].view(batch_size, self.latent_dim).to(self._get_device())
        # )

        z_mean_SL = self.z_mean_layer(x_SPE.reshape(batch_size, -1))
        return z_mean_SL

    def latent_to_sequence(
        self,
        z_SL: torch.Tensor,
    ) -> torch.Tensor:
        x_SPA = self.decoder(z_SL)
        # x_SPA = torch.nn.functional.softmax(x_SPA, dim=2)
        return x_SPA

    def predict_topic_probabilities(self, x_sequence_SP: torch.Tensor) -> torch.Tensor:
        z_mean_SL = self.sequence_to_latent(x_sequence_SP)
        return self.latent_space_to_topic_proportions_layer(z_mean_SL)

    def predict_signal_intensity(
        self,
        x_sequence_SP: torch.Tensor,
        repertoire_labels_R: torch.Tensor,
        label_of_interest: int = 1,
    ) -> torch.Tensor:
        topic_signed_weights_T = self._compute_signal_topic_weights(
            repertoire_labels_R, label_of_interest
        )
        z_SL = self.sequence_to_latent(x_sequence_SP)
        seq_topic_probabilities_ST = self.latent_space_to_topic_proportions_layer(z_SL)
        seq_signal_topic_probabilities_ST = seq_topic_probabilities_ST[
            :, : self.n_topics_signal
        ]
        signal_intensity = (
            seq_signal_topic_probabilities_ST * topic_signed_weights_T
        ).sum(axis=1)
        return signal_intensity

    def _get_device(self) -> torch.device:
        return self.z_log_sigma_param.device

    def _compute_signal_topic_weights(
        self,
        repertoire_labels_R: torch.Tensor,
        label_of_interest: int,
    ) -> torch.Tensor:
        repertoire_topic_proportions_RT = self.get_topic_proportion_matrix()
        repertoire_signal_topic_proportions_RT = repertoire_topic_proportions_RT[
            :, : self.n_topics_signal
        ]
        avg_proportion_by_topic_pos_repertoires_T = (
            repertoire_signal_topic_proportions_RT[
                repertoire_labels_R == label_of_interest
            ].mean(axis=0)
        )
        avg_proportion_by_topic_neg_repertoires_T = (
            repertoire_signal_topic_proportions_RT[
                repertoire_labels_R != label_of_interest
            ].mean(axis=0)
        )
        topic_differences_T = (
            avg_proportion_by_topic_pos_repertoires_T
            - avg_proportion_by_topic_neg_repertoires_T
        )
        topic_weights_T = topic_differences_T.abs() / topic_differences_T.abs().sum()
        topic_signs_T = topic_differences_T / topic_differences_T.abs()
        topic_signed_weights_T = topic_weights_T * topic_signs_T
        return topic_signed_weights_T
