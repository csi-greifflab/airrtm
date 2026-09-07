import torch

from airrtm.types import AIRRTM_ModelOutput, AIRRTM_ModelTarget


EPS = 1e-12
NEG_INF = float("-inf")


class CompositeLoss(torch.nn.Module):
    """A composite loss for an AIRRTM model.

    Terms
    -----
    TM likelihood
        The likelihood of observing a sequence in its own repertoire,
        ``p(s | R) = sum_t theta_Rt * phi_ts``, **normalised over the sequences in
        the batch**::

            log p(s | R) = log sum_t theta_Rt phi_ts
                           - log sum_{s' in batch} sum_t theta_Rt phi_ts'

        The normaliser is what makes this a topic model rather than a bag of
        independent per-topic scores. Without it the term is monotone increasing in
        every ``phi``, so its optimum is to saturate all of them (which is exactly
        what happened in earlier runs: ``tm_likelihoods`` sat at 0.999 for both the
        sequence's own repertoire and for unrelated ones, and the coefficient had to
        be set to zero). Normalising makes topics compete for sequences, so ``Phi``
        has to specialise.

        Since the AIR "vocabulary" cannot be enumerated, the batch acts as the
        negative pool -- the standard sampled-softmax / noise-contrastive treatment
        for embedded topic models.
    Label likelihood
        Multiple-instance learning: per-sequence label logits are pooled within each
        repertoire by a temperature-controlled smooth maximum
        ``(logsumexp(tau * l) - log n) / tau`` (``tau -> 0`` gives the mean,
        ``tau -> inf`` the max), then scored against the repertoire label with a
        class-weighted binary cross-entropy.
    Reconstruction and KL
        The usual VAE pair. Padding positions are excluded from the reconstruction
        loss, so the reported accuracy is not inflated by the (in Emerson, ~54%)
        share of pad tokens.
    Regularisers
        Optional L1 sparsity on sequence-topic probabilities, per-repertoire topic
        entropy, corpus-level topic usage (load balancing), and topic decorrelation.
        The two entropy terms pull in opposite directions on purpose:
        ``theta_entropy_coef`` acts on the *mean of the entropies* and should be kept
        small or zero, while ``topic_usage_coef`` acts on the *entropy of the mean*
        and is what prevents topic collapse.

    The terms combine as::

        total = vae_coef * (reconstruction_loss_coef * reconstruction
                            + (1 - reconstruction_loss_coef) * kl)
              + (1 - vae_coef) * (tm_likelihood_coef * tm
                                  + (1 - tm_likelihood_coef) * label)
              + topic_l1_coef * topic_l1
              + theta_entropy_coef * theta_entropy
              + topic_usage_coef * (-topic_usage_entropy)
              + topic_decorrelation_coef * topic_decorrelation

    Note that ``reconstruction_loss_coef = 0`` makes the VAE branch pure KL, and
    ``tm_likelihood_coef = 0`` makes the other branch pure label loss -- which is
    what reduced earlier runs to plain noisy-label learning.

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
        reconstruction_loss_coef: float,
        topic_l1_coef: float = 0.0,
        positive_class_weight: float = 0.5,
        n_sequences_per_repertoire: int | None = None,
        theta_entropy_coef: float = 0.0,
        topic_usage_coef: float = 0.0,
        topic_usage_momentum: float = 0.9,
        topic_decorrelation_coef: float = 0.0,
        pad_value: int | None = None,
        tm_max_negatives: int | None = None,
        default_tau: float = 1.0,
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
        self.theta_entropy_coef = theta_entropy_coef
        self.topic_usage_coef = topic_usage_coef
        self.topic_usage_momentum = topic_usage_momentum
        # Running estimate of corpus-level topic usage; created on first use, when
        # the topic count and device are known.
        self._theta_bar: torch.Tensor | None = None
        self.topic_decorrelation_coef = topic_decorrelation_coef

        # Retained so old configs keep working; grouping is now derived from the
        # repertoire ids in the target rather than from the batch layout.
        self.n_sequences_per_repertoire = n_sequences_per_repertoire

        self.pad_value = pad_value
        self.tm_max_negatives = tm_max_negatives
        self.default_tau = default_tau

        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=-100 if pad_value is None else pad_value
        )

        self.return_individual_compotents = return_individual_compotents

    def forward(
        self,
        inputs: AIRRTM_ModelOutput,
        targets: AIRRTM_ModelTarget,
        tau: float | None = None,
    ) -> dict[str, torch.Tensor]:
        tau = self.default_tau if tau is None else tau
        repertoire_ids_S = targets["repertoire_ids"]
        grouping = _group_by_repertoire(repertoire_ids_S)

        tm_loss = self._tm_loss(inputs, grouping)
        label_loss, label_accuracy = self._label_loss(inputs, targets, grouping, tau)
        reconstruction_loss, reconstruction_accuracy = self._reconstruction(
            inputs, targets
        )
        kl_divergence = inputs["kl_divergences"].mean()

        topic_l1 = inputs["seq_topic_probabilities"].sum(dim=1).mean()
        theta_entropy = _mean_entropy(inputs["log_topic_proportions"], grouping)
        topic_usage_entropy, usage_loss = self._topic_usage(
            inputs["log_topic_proportions"], grouping
        )
        topic_decorrelation = _topic_decorrelation(inputs["seq_topic_probabilities"])
        predicted_witness_rate = torch.sigmoid(inputs["label_likelihoods"]).mean()

        vae_loss = (
            self.reconstruction_loss_coef * reconstruction_loss
            + self.kld_coef * kl_divergence
        )
        non_vae_loss = (
            self.tm_likelihood_coef * tm_loss
            + self.label_likelihood_coef * label_loss
        )
        total_loss = (
            self.vae_coef * vae_loss
            + (1 - self.vae_coef) * non_vae_loss
            + self.topic_l1_coef * topic_l1
            + self.theta_entropy_coef * theta_entropy
            + self.topic_usage_coef * usage_loss
            + self.topic_decorrelation_coef * topic_decorrelation
        )

        if not self.return_individual_compotents:
            return {"total_loss": total_loss}
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_accuracy": reconstruction_accuracy,
            "kl_divergence": kl_divergence,
            "tm_loss": tm_loss,
            "label_loss": label_loss,
            "label_accuracy": label_accuracy,
            "topic_l1": topic_l1,
            "theta_entropy": theta_entropy,
            "topic_usage_entropy": topic_usage_entropy,
            "topic_decorrelation": topic_decorrelation,
            "predicted_witness_rate": predicted_witness_rate,
        }

    def _topic_usage(
        self, log_topic_proportions_ST: torch.Tensor, grouping: "_Grouping"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Corpus-level topic usage, and the loss that keeps it spread out.

        Note carefully which average sits where. ``theta_entropy`` is the *mean of
        the entropies*, ``(1/R) sum_r H(theta_r)``: per-repertoire spread, which we
        want low, because a repertoire using few topics is what makes topics
        interpretable. This is the *entropy of the mean*,
        ``H((1/R) sum_r theta_r)``: how many topics the corpus uses at all, which we
        want high. Collapse is the state where both are low -- every repertoire sharp
        *and* all of them sharp on the same two topics.

        Penalising ``-H(theta_bar)`` leaves each repertoire free to be one-hot; it
        only requires that different repertoires disagree about which topic. Same
        device as load balancing in mixture-of-experts, where a gate collapsing onto
        a couple of experts is the identical failure.

        A batch holds only a handful of repertoires, and R one-hot vectors cannot
        exceed ``log R`` of entropy -- far below ``log n_topics``. Asking a single
        batch to spread over every topic is unattainable by construction, so the
        estimate is smoothed across steps: the running mean supplies corpus context
        while the current batch carries the gradient.
        """
        theta_RT = torch.exp(log_topic_proportions_ST[grouping.first_index_R])
        batch_mean_T = theta_RT.mean(dim=0)

        if self._theta_bar is None or self._theta_bar.shape != batch_mean_T.shape:
            self._theta_bar = batch_mean_T.detach().clone()
        self._theta_bar = self._theta_bar.to(batch_mean_T.device)

        momentum = self.topic_usage_momentum
        smoothed_T = momentum * self._theta_bar + (1 - momentum) * batch_mean_T
        self._theta_bar = smoothed_T.detach()

        normalised_T = smoothed_T / smoothed_T.sum().clamp(min=EPS)
        usage_loss = (normalised_T * torch.log(normalised_T.clamp(min=EPS))).sum()
        return -usage_loss.detach(), usage_loss

    # ---------------------------------------------------------------- TM branch

    def _tm_loss(
        self, inputs: AIRRTM_ModelOutput, grouping: "_Grouping"
    ) -> torch.Tensor:
        """Negative mean log-likelihood, normalised over the batch's sequences."""
        seq_topic_logits_ST = inputs["seq_topic_logits"]
        # theta is constant within a repertoire, so one row per group suffices.
        log_theta_RT = inputs["log_topic_proportions"][grouping.first_index_R]

        negatives_ST = seq_topic_logits_ST
        if (
            self.tm_max_negatives is not None
            and seq_topic_logits_ST.shape[0] > self.tm_max_negatives
        ):
            sample = torch.randperm(
                seq_topic_logits_ST.shape[0], device=seq_topic_logits_ST.device
            )[: self.tm_max_negatives]
            negatives_ST = seq_topic_logits_ST[sample]

        # (R, S'): log sum_t theta_rt * phi_ts' for every candidate sequence s'.
        cross_RS = torch.logsumexp(
            log_theta_RT.unsqueeze(1) + negatives_ST.unsqueeze(0), dim=2
        )
        log_partition_R = torch.logsumexp(cross_RS, dim=1)
        log_likelihood_S = (
            inputs["tm_log_scores"] - log_partition_R[grouping.inverse_S]
        )
        return -log_likelihood_S.mean()

    # ------------------------------------------------------------- label branch

    def _label_loss(
        self,
        inputs: AIRRTM_ModelOutput,
        targets: AIRRTM_ModelTarget,
        grouping: "_Grouping",
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_RN, mask_RN = grouping.scatter(inputs["label_likelihoods"])
        counts_R = grouping.counts_R.to(logits_RN.dtype)

        # Smooth maximum over the instances of each bag.
        pooled_logits_R = (
            torch.logsumexp(tau * logits_RN.masked_fill(~mask_RN, NEG_INF), dim=1)
            - torch.log(counts_R)
        ) / tau

        labels_R = targets["sequence_repertoire_labels"][grouping.first_index_R].to(
            pooled_logits_R.dtype
        )
        weights_R = torch.where(
            labels_R > 0.5,
            torch.full_like(labels_R, self.positive_class_weight),
            torch.full_like(labels_R, 1 - self.positive_class_weight),
        )
        label_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pooled_logits_R, labels_R, weight=weights_R, reduction="mean"
        )
        label_accuracy = ((pooled_logits_R >= 0).to(labels_R.dtype) == labels_R).to(
            torch.float32
        ).mean()
        return label_loss, label_accuracy

    # ------------------------------------------------------------- VAE branch

    def _reconstruction(
        self, inputs: AIRRTM_ModelOutput, targets: AIRRTM_ModelTarget
    ) -> tuple[torch.Tensor, torch.Tensor]:
        symbol_logits_SPA = inputs["decoded_sequences"]
        true_symbols_SP = targets["sequences"]
        reconstruction_loss = self.ce_loss(
            symbol_logits_SPA.permute(0, 2, 1), true_symbols_SP
        )
        if not self.return_individual_compotents:
            return reconstruction_loss, reconstruction_loss.detach()

        with torch.no_grad():
            correct_SP = torch.argmax(symbol_logits_SPA, dim=2) == true_symbols_SP
            if self.pad_value is None:
                reconstruction_accuracy = correct_SP.to(torch.float32).mean()
            else:
                real_SP = true_symbols_SP != self.pad_value
                reconstruction_accuracy = (correct_SP & real_SP).sum() / real_SP.sum().clamp(
                    min=1
                )
        return reconstruction_loss, reconstruction_accuracy


class _Grouping:
    """Maps a flat batch of sequences onto the repertoires they came from.

    Built from repertoire ids rather than batch layout, so nothing breaks when the
    caller shuffles the batch, and groups may have different sizes.
    """

    def __init__(self, repertoire_ids_S: torch.Tensor):
        device = repertoire_ids_S.device
        self.unique_ids_R, self.inverse_S = torch.unique(
            repertoire_ids_S, return_inverse=True
        )
        self.n_groups = int(self.unique_ids_R.shape[0])
        self.counts_R = torch.bincount(self.inverse_S, minlength=self.n_groups)

        self.order_S = torch.argsort(self.inverse_S, stable=True)
        sorted_inverse_S = self.inverse_S[self.order_S]
        offsets_R = torch.cumsum(self.counts_R, dim=0) - self.counts_R
        n_S = torch.arange(sorted_inverse_S.shape[0], device=device)
        self.position_S = n_S - offsets_R[sorted_inverse_S]
        self.sorted_inverse_S = sorted_inverse_S
        self.n_max = int(self.counts_R.max())
        # Index of one representative sequence per group, for values that are
        # constant within a repertoire (theta, the repertoire label).
        self.first_index_R = self.order_S[offsets_R]

    def scatter(self, values_S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape ``(S,)`` into a padded ``(R, n_max)`` matrix plus a validity mask."""
        matrix_RN = torch.zeros(
            self.n_groups, self.n_max, dtype=values_S.dtype, device=values_S.device
        )
        mask_RN = torch.zeros(
            self.n_groups, self.n_max, dtype=torch.bool, device=values_S.device
        )
        matrix_RN[self.sorted_inverse_S, self.position_S] = values_S[self.order_S]
        mask_RN[self.sorted_inverse_S, self.position_S] = True
        return matrix_RN, mask_RN


def _group_by_repertoire(repertoire_ids_S: torch.Tensor) -> _Grouping:
    return _Grouping(repertoire_ids_S)


def _mean_entropy(
    log_topic_proportions_ST: torch.Tensor, grouping: _Grouping
) -> torch.Tensor:
    """Mean of the per-repertoire topic entropies, ``(1/R) sum_r H(theta_r)``.

    Minimising it pushes each repertoire towards using few topics, the additive
    sparsity regularisation that ARTM applies to Theta. Do not confuse it with
    :meth:`CompositeLoss._topic_usage`, the entropy of the *mean* -- that one should
    be maximised, and minimising this one alone drives topic collapse.
    """
    log_theta_RT = log_topic_proportions_ST[grouping.first_index_R]
    return -(torch.exp(log_theta_RT) * log_theta_RT).sum(dim=1).mean()


def _topic_decorrelation(seq_topic_probabilities_ST: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal correlation between topics across the batch.

    Penalising it stops several topics from collapsing onto the same signal, which
    the paper reports as a failure mode on the multi-signal dataset.
    """
    centred_ST = seq_topic_probabilities_ST - seq_topic_probabilities_ST.mean(
        dim=0, keepdim=True
    )
    std_T = centred_ST.pow(2).mean(dim=0).sqrt().clamp(min=EPS)
    correlation_TT = (centred_ST.T @ centred_ST) / (
        centred_ST.shape[0] * std_T.unsqueeze(0) * std_T.unsqueeze(1)
    )
    n_topics = correlation_TT.shape[0]
    off_diagonal = ~torch.eye(
        n_topics, dtype=torch.bool, device=correlation_TT.device
    )
    return correlation_TT[off_diagonal].pow(2).mean()
