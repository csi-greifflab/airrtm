import math

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
        for embedded topic models. ``tm_negative_mode="batch"`` (default) normalises
        each repertoire against every sequence in the batch, regardless of which
        repertoire or class it came from. ``tm_negative_mode="opposite_class_pair"``
        normalises each repertoire only against its own sequences plus those of one
        paired repertoire of the *opposite* label -- a sharper, explicitly
        class-contrastive signal, at the cost of requiring at least one repertoire
        of each class in every batch (true today via the label-stratified grouping).

        ``tm_likelihood_family="raw_bce"`` replaces the whole scheme above with v1's
        original formulation (``archive/model.py:252``, ``archive/losses.py:19-26``):
        a plain, clipped binary cross-entropy on an *unnormalised* ``theta . phi`` dot
        product, target 1 against the sequence's own repertoire and target 0 against
        one paired repertoire of the opposite class. It requires
        ``AIRRTM_Model(theta_normalization="none")`` -- Theta must be a raw,
        unconstrained score, not a softmax-normalised probability -- and
        ``theta_is_normalized=False`` here, which also forces
        ``theta_entropy_coef``/``topic_usage_coef`` to zero (they assume a
        probability simplex that does not exist in this mode).
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
                                  + (1 - tm_likelihood_coef) * label
                                  + topic_l1_coef * topic_l1
                                  + theta_entropy_coef * theta_entropy
                                  + topic_usage_coef * (-topic_usage_entropy)
                                  + topic_decorrelation_coef * topic_decorrelation
                                  + phi_l2_coef * phi_l2)

    All the topic-model regularisers (everything reading phi/theta) live under the
    ``(1 - vae_coef)`` branch alongside tm/label, not as flat top-level terms -- so a
    ``vae_coef`` warm start toward ``1.0`` (a "pure VAE" phase, see ``train_model``'s
    ``vae_coef_start``) genuinely stops all topic-model gradient, regularisers
    included, rather than leaving them pushing on Theta unopposed.

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
        tm_negative_mode: str = "batch",
        tm_likelihood_family: str = "batch_softmax",
        phi_l2_coef: float = 0.0,
        theta_is_normalized: bool = True,
        abundance_weighted_losses: bool = False,
        normalize_loss_scales: bool = False,
        loss_scale_mode: str = "chance",
        default_tau: float = 1.0,
        return_individual_compotents: bool = True,
    ):
        super().__init__()
        if tm_negative_mode not in ("batch", "opposite_class_pair"):
            raise ValueError(
                f"tm_negative_mode must be 'batch' or 'opposite_class_pair', got "
                f"{tm_negative_mode!r}"
            )
        self.tm_negative_mode = tm_negative_mode
        if tm_likelihood_family not in ("batch_softmax", "raw_bce"):
            raise ValueError(
                f"tm_likelihood_family must be 'batch_softmax' or 'raw_bce', got "
                f"{tm_likelihood_family!r}"
            )
        self.tm_likelihood_family = tm_likelihood_family
        self.phi_l2_coef = phi_l2_coef
        # Where clonal abundance belongs. The sampler can apply it (draw indices
        # proportional to duplicate_count) or the likelihood terms can (draw
        # uniformly over distinct clonotypes, then weight each sequence's TM and
        # reconstruction term by its abundance) -- but not both, and never in
        # Theta's pooling, which should mirror the Fisher burden score's
        # "fraction of DISTINCT clonotypes" statistic. Sampler-side draws with
        # replacement, so it covers fewer distinct clonotypes and cannot be
        # combined with a disjoint Theta sample; loss-side keeps the same target
        # distribution while every distinct clonotype in the draw is seen exactly
        # once. Pair abundance_weighted_losses=True with
        # abundance_weighted_sampling=False and
        # AIRRTM_Model(theta_pooling_weights=False).
        self.abundance_weighted_losses = abundance_weighted_losses
        # The four likelihood terms live on wildly different scales, so a
        # coefficient does not mean what it looks like. Chance values, in nats:
        # reconstruction ln(21) = 3.04, tm ln(4*8192) = 10.40 (floor ln(8192), so a
        # range of only 1.39 and realistically ~0.05), label 0.5*ln2 = 0.35. With
        # the flagship coefficients, tm carries the largest weight (0.665) and the
        # smallest usable range, while reconstruction's range is 9x the label's.
        # Setting this True divides each term by its own chance value before the
        # weighted sum, so every term is ~1 at chance and 0 at perfect and the
        # coefficients become comparable. The *reported* per-term metrics stay
        # unnormalised, so `tm`, `rec` and `label` in the logs remain directly
        # comparable to every run recorded in FINDINGS.md -- only `total_loss`,
        # and therefore the gradient balance and early stopping, change.
        self.normalize_loss_scales = normalize_loss_scales
        if loss_scale_mode not in ("chance", "range"):
            raise ValueError(
                f"loss_scale_mode must be 'chance' or 'range', got {loss_scale_mode!r}"
            )
        # "chance" divides by each term's value at chance; "range" divides by how far
        # the term can actually FALL, which is the more defensible choice and differs
        # only for tm. Reconstruction and the label can reach 0, so their range equals
        # their chance value. The batch-softmax tm cannot: even a perfect model only
        # concentrates p(s|r) onto that repertoire's own S/R sequences in the pool, so
        # it bottoms out at log(S/R) and its reachable range is log(R) = log(
        # n_repertoires_in_batch) -- 1.386 at R=4, against a chance value of 10.397.
        # "chance" therefore under-weights the tm gradient by 7.5x at the default batch
        # composition. Note the *empirically* reachable range at full scale is smaller
        # still (~0.03 nats, best ever 0.0324 unopposed); normalising by that would
        # amplify a term that is mostly noise, which is why the structural range is
        # used rather than a measured one.
        self.loss_scale_mode = loss_scale_mode
        self.theta_is_normalized = theta_is_normalized
        if not theta_is_normalized and (theta_entropy_coef != 0 or topic_usage_coef != 0):
            raise ValueError(
                "theta_is_normalized=False means Theta is a raw, unconstrained score "
                "(v1-style; pair with AIRRTM_Model(theta_normalization='none')), not "
                "a probability distribution -- theta_entropy_coef/topic_usage_coef "
                "both assume a normalised Theta and must be zero in this mode."
            )
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

        has_topic_model = inputs["seq_topic_logits"] is not None
        if not has_topic_model and (
            self.tm_likelihood_coef != 0
            or self.topic_l1_coef != 0
            or self.topic_decorrelation_coef != 0
        ):
            raise ValueError(
                "The model was built with use_topic_model=False (phi dropped), but "
                "tm_likelihood_coef/topic_l1_coef/topic_decorrelation_coef -- all of "
                "which read phi -- are not all zero. Set them to 0.0, or build the "
                "model with use_topic_model=True."
            )

        loss_weights_S = None
        if self.abundance_weighted_losses:
            loss_weights_S = _normalised_abundance(targets.get("weights"), grouping)

        if has_topic_model:
            partner_index_R = None
            if self.tm_negative_mode == "opposite_class_pair" or self.tm_likelihood_family == "raw_bce":
                labels_R = targets["sequence_repertoire_labels"][grouping.first_index_R]
                partner_index_R = _opposite_class_partners(labels_R)
            if self.tm_likelihood_family == "raw_bce":
                tm_loss = self._tm_loss_raw_bce(
                    inputs, grouping, partner_index_R, loss_weights_S
                )
            else:
                tm_loss = self._tm_loss(
                    inputs, grouping, partner_index_R, loss_weights_S
                )
            topic_l1 = inputs["seq_topic_probabilities"].sum(dim=1).mean()
            topic_decorrelation = _topic_decorrelation(inputs["seq_topic_probabilities"])
            phi_l2 = (inputs["seq_topic_logits"] ** 2).mean()
        else:
            zero = inputs["kl_divergences"].sum() * 0.0
            tm_loss, topic_l1, topic_decorrelation, phi_l2 = zero, zero, zero, zero
        label_loss, label_accuracy = self._label_loss(inputs, targets, grouping, tau)
        reconstruction_loss, reconstruction_accuracy = self._reconstruction(
            inputs, targets, loss_weights_S
        )
        kl_divergence = inputs["kl_divergences"].mean()

        theta_entropy = _mean_entropy(inputs["log_topic_proportions"], grouping)
        topic_usage_entropy, usage_loss = self._topic_usage(
            inputs["log_topic_proportions"], grouping
        )
        predicted_witness_rate = torch.sigmoid(inputs["label_likelihoods"]).mean()

        scales = self._loss_scales(inputs, targets, grouping)

        vae_loss = (
            self.reconstruction_loss_coef * reconstruction_loss / scales["reconstruction"]
            + self.kld_coef * kl_divergence
        )
        # Every regulariser here reads phi/theta -- topic-model quantities -- so they
        # live under the non-VAE branch's (1 - vae_coef) weight, same as tm/label.
        # Otherwise a vae_coef warm-start (vae_coef_start near 1.0) would still push
        # gradient onto Theta through these terms even while tm/label are fully
        # suppressed, undercutting a "pure VAE, nothing else touches the encoder" phase.
        non_vae_loss = (
            self.tm_likelihood_coef * tm_loss / scales["tm"]
            + self.label_likelihood_coef * label_loss / scales["label"]
            + self.topic_l1_coef * topic_l1
            + self.theta_entropy_coef * theta_entropy / scales["entropy"]
            + self.topic_usage_coef * usage_loss / scales["entropy"]
            + self.topic_decorrelation_coef * topic_decorrelation
            + self.phi_l2_coef * phi_l2
        )
        total_loss = self.vae_coef * vae_loss + (1 - self.vae_coef) * non_vae_loss

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
            "phi_l2": phi_l2,
            "predicted_witness_rate": predicted_witness_rate,
        }

    def _loss_scales(
        self,
        inputs: AIRRTM_ModelOutput,
        targets: AIRRTM_ModelTarget,
        grouping: "_Grouping",
    ) -> dict[str, float]:
        """Chance value of each term, or 1.0 when normalisation is off.

        Every scale is derived from the batch actually in hand rather than
        hard-coded, so it stays correct when the alphabet, the negative pool, the
        class weights or the topic count change.
        """
        if not self.normalize_loss_scales:
            return {"reconstruction": 1.0, "tm": 1.0, "label": 1.0, "entropy": 1.0}

        # Reconstruction: uniform over the alphabet. pad_value is one past the last
        # real symbol and pad positions are excluded from the loss, so the number of
        # classes the term can actually be wrong about is pad_value itself.
        reconstruction = math.log(self.pad_value) if self.pad_value else 1.0

        # TM: a softmax NLL over its candidate pool sits at log(pool) with no signal;
        # the raw-BCE family is a binary cross-entropy, so its chance value is log 2.
        if self.tm_likelihood_family == "raw_bce":
            # A binary cross-entropy: chance log 2, floor 0, so both modes agree.
            tm = math.log(2.0)
        else:
            pool = int(inputs["seq_topic_logits"].shape[0])
            n_pools = 1  # how many repertoires' worth of sequences the pool holds
            if self.tm_negative_mode == "opposite_class_pair":
                # own repertoire + one opposite-class partner.
                pool = int(2 * grouping.counts_R.to(torch.float32).mean().item())
                n_pools = 2
            else:
                if self.tm_max_negatives is not None:
                    pool = min(pool, self.tm_max_negatives)
                n_pools = max(grouping.n_groups, 1)
            if self.loss_scale_mode == "range":
                # chance log(pool) minus the floor log(pool / n_pools).
                tm = math.log(max(n_pools, 2))
            else:
                tm = math.log(max(pool, 2))

        # Label: a constant 0.5 prediction costs mean(class weight) * log 2. Reading
        # the weights off the batch keeps this right for any positive_class_weight.
        labels_R = targets["sequence_repertoire_labels"][grouping.first_index_R]
        positive_share = float((labels_R > 0.5).to(torch.float32).mean().item())
        mean_weight = (
            positive_share * self.positive_class_weight
            + (1 - positive_share) * (1 - self.positive_class_weight)
        )
        label = max(mean_weight, EPS) * math.log(2.0)

        # Both entropy terms live on [0, log n_topics].
        entropy = math.log(max(int(inputs["log_topic_proportions"].shape[1]), 2))

        return {
            "reconstruction": max(reconstruction, EPS),
            "tm": max(tm, EPS),
            "label": max(label, EPS),
            "entropy": max(entropy, EPS),
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
        self,
        inputs: AIRRTM_ModelOutput,
        grouping: "_Grouping",
        partner_index_R: torch.Tensor | None = None,
        weights_S: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Negative mean log-likelihood, normalised over a per-repertoire negative pool.

        ``weights_S`` (from ``abundance_weighted_losses``) reweights the outer
        expectation only: the partition function still runs over the drawn
        candidate pool, which defines the support, while the weights restore the
        abundance-proportional target measure that uniform sampling removed. They
        are normalised to mean exactly 1, so the weighted mean is a drop-in for
        the plain one and reduces to it when every weight is equal.
        """
        log_likelihood_S = self._tm_log_likelihood_per_sequence(
            inputs, grouping, partner_index_R
        )
        if weights_S is None:
            return -log_likelihood_S.mean()
        return -(weights_S * log_likelihood_S).mean()

    def _tm_log_likelihood_per_sequence(
        self,
        inputs: AIRRTM_ModelOutput,
        grouping: "_Grouping",
        partner_index_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``log p(s | r_own)``, normalised over a per-repertoire negative pool.

        ``tm_negative_mode="batch"``: the pool is every sequence in the batch
        (optionally capped to ``tm_max_negatives``), regardless of repertoire or
        class. ``tm_negative_mode="opposite_class_pair"``: the pool for repertoire
        ``r`` is restricted to ``r``'s own sequences plus ``partner_index_R[r]``'s --
        one repertoire of the opposite class, so the TM gradient is a direct
        "look like my own repertoire, not like the opposite-class one" signal
        instead of a generic partition function. ``tm_max_negatives`` is not
        applied in this mode (the per-repertoire pool is already small).
        """
        seq_topic_logits_ST = inputs["seq_topic_logits"]
        # theta is constant within a repertoire, so one row per group suffices.
        log_theta_RT = inputs["log_topic_proportions"][grouping.first_index_R]

        negatives_ST = seq_topic_logits_ST
        if self.tm_negative_mode == "batch" and (
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
        if self.tm_negative_mode == "opposite_class_pair":
            if partner_index_R is None:
                raise ValueError(
                    "tm_negative_mode='opposite_class_pair' requires partner_index_R"
                )
            own_index_R = torch.arange(grouping.n_groups, device=cross_RS.device)
            own_mask_RS = grouping.inverse_S.unsqueeze(0) == own_index_R.unsqueeze(1)
            partner_mask_RS = grouping.inverse_S.unsqueeze(0) == partner_index_R.unsqueeze(1)
            cross_RS = cross_RS.masked_fill(~(own_mask_RS | partner_mask_RS), NEG_INF)
        log_partition_R = torch.logsumexp(cross_RS, dim=1)
        return inputs["tm_log_scores"] - log_partition_R[grouping.inverse_S]

    def _tm_loss_raw_bce(
        self,
        inputs: AIRRTM_ModelOutput,
        grouping: "_Grouping",
        partner_index_R: torch.Tensor,
        weights_S: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """v1's original TM likelihood (``archive/model.py:252``,
        ``archive/losses.py:19-26``): a plain, clipped binary cross-entropy on an
        unnormalised ``theta . phi`` dot product -- target 1 for a sequence scored
        against its own repertoire's (raw) Theta, target 0 against one paired
        repertoire of the opposite class. Requires
        ``AIRRTM_Model(theta_normalization="none")``: ``inputs["tm_log_scores"]`` is
        then the raw positive-pair dot product (computed in the model, where Theta
        is already broadcast per-sequence), and ``inputs["log_topic_proportions"]``
        is the raw (unnormalised) Theta needed here to score the negative pair.
        """
        theta_ST = inputs["log_topic_proportions"]
        phi_ST = inputs["seq_topic_logits"]

        positive_score_S = inputs["tm_log_scores"]
        partner_of_S = partner_index_R[grouping.inverse_S]
        partner_repr_index_S = grouping.first_index_R[partner_of_S]
        theta_partner_ST = theta_ST[partner_repr_index_S]
        negative_score_S = (theta_partner_ST * phi_ST).sum(dim=1)

        eps = 1e-7
        positive_prob_S = positive_score_S.clamp(min=eps, max=1 - eps)
        negative_prob_S = negative_score_S.clamp(min=eps, max=1 - eps)
        # weights_S is normalised to mean 1, so passing it as `weight` with the
        # default 'mean' reduction is exactly the weighted mean.
        positive_loss = torch.nn.functional.binary_cross_entropy(
            positive_prob_S, torch.ones_like(positive_prob_S), weight=weights_S
        )
        negative_loss = torch.nn.functional.binary_cross_entropy(
            negative_prob_S, torch.zeros_like(negative_prob_S), weight=weights_S
        )
        return (positive_loss + negative_loss) / 2

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
        self,
        inputs: AIRRTM_ModelOutput,
        targets: AIRRTM_ModelTarget,
        weights_S: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        symbol_logits_SPA = inputs["decoded_sequences"]
        true_symbols_SP = targets["sequences"]
        if weights_S is None:
            reconstruction_loss = self.ce_loss(
                symbol_logits_SPA.permute(0, 2, 1), true_symbols_SP
            )
        else:
            # Same per-token mean as self.ce_loss, but with each sequence's tokens
            # weighted by its clonal abundance -- the loss-side half of moving
            # abundance out of the sampler (abundance_weighted_losses).
            per_token_SP = torch.nn.functional.cross_entropy(
                symbol_logits_SPA.permute(0, 2, 1),
                true_symbols_SP,
                ignore_index=self.ce_loss.ignore_index,
                reduction="none",
            )
            if self.pad_value is None:
                real_SP = torch.ones_like(per_token_SP)
            else:
                real_SP = (true_symbols_SP != self.pad_value).to(per_token_SP.dtype)
            weights_S1 = weights_S.reshape(-1, 1)
            reconstruction_loss = (weights_S1 * per_token_SP).sum() / (
                weights_S1 * real_SP
            ).sum().clamp(min=EPS)
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


def _normalised_abundance(
    weights_S: torch.Tensor | None, grouping: "_Grouping"
) -> torch.Tensor | None:
    """Clonal abundances rescaled to mean 1 *within each repertoire*.

    Scaling per repertoire rather than globally keeps every repertoire's total
    contribution to the loss equal to its sequence count, exactly as the plain
    mean does -- so a repertoire holding one enormous clone does not drown the
    other three in the batch. The global mean of the result is exactly 1 for any
    group sizes, because each repertoire's weights sum to its own count, which is
    what lets callers use it as a drop-in `weight=` with a 'mean' reduction.

    Returns None when no abundances are available, so the caller falls back to
    the unweighted path.
    """
    if weights_S is None:
        return None
    weights_S = weights_S.to(torch.float32).clamp(min=0.0)
    sums_R = torch.zeros(
        grouping.n_groups, dtype=weights_S.dtype, device=weights_S.device
    ).index_add_(0, grouping.inverse_S, weights_S)
    means_R = sums_R / grouping.counts_R.to(weights_S.dtype).clamp(min=1)
    return weights_S / means_R[grouping.inverse_S].clamp(min=EPS)


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


def _opposite_class_partners(labels_R: torch.Tensor) -> torch.Tensor:
    """For each repertoire-group index, the index of one repertoire of the other class.

    Cycles through the opposite-class pool when the two classes aren't evenly split
    within the batch. Raises if a batch holds only one class -- the label-stratified
    grouping (``train.py::_stratified_groups``) is expected to prevent that, so this
    makes the assumption explicit rather than silently pairing same-class
    repertoires, which would make ``tm_negative_mode="opposite_class_pair"`` a no-op.
    """
    n = labels_R.shape[0]
    device = labels_R.device
    index_R = torch.arange(n, device=device)
    partner_R = torch.empty(n, dtype=torch.long, device=device)
    for cls in (0, 1):
        this_index = index_R[labels_R == cls]
        other_index = index_R[labels_R != cls]
        if this_index.numel() == 0:
            continue
        if other_index.numel() == 0:
            raise ValueError(
                "tm_negative_mode='opposite_class_pair' requires at least one "
                "repertoire of each class per batch; got a single-class batch."
            )
        reps = -(-this_index.numel() // other_index.numel())  # ceil division
        partner_R[this_index] = other_index.repeat(reps)[: this_index.numel()]
    return partner_R


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
