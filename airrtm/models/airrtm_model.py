import torch

from airrtm.types import AIRRTM_ModelOutput


EPS = 1e-12

#: How repertoire topic proportions (Theta) are obtained.
#:
#: - ``"free"``: one free parameter row per training repertoire (the original v1/v2
#:   behaviour). Cannot score a repertoire the model was not trained on.
#: - ``"amortized"``: Theta is inferred from a sample of the repertoire's own
#:   sequences by a permutation-invariant pooling head, so unseen repertoires get
#:   topic proportions for free.
#: - ``"both"``: the free row acts as a per-repertoire residual on top of the
#:   amortized logits; unseen repertoires fall back to the amortized part alone.
THETA_MODES = ("free", "amortized", "both")

#: How a repertoire's sequences are collapsed into its topic proportions.
#:
#: - ``"mean"``: average the per-sequence topic logits. Represents *typical*
#:   composition, and cannot represent the presence of a rare subset.
#: - ``"attention"``: gated attention with one map per topic, so a topic can detect
#:   a small set of sequences rather than shift an average. See
#:   :class:`TopicAttentionPooling`.
THETA_POOLINGS = ("mean", "attention")

#: What the repertoire-label head reads.
#:
#: - ``"sequence"``: a per-sequence logit from that sequence's own signal topics,
#:   pooled into a bag prediction by the MIL term in the loss.
#: - ``"repertoire"``: the repertoire's topic proportions Theta directly. Combined
#:   with attention pooling this lets the label be driven by *detected* sequences
#:   rather than by an average; with ``theta_mode="free"`` it would instead be the v1
#:   shortcut, a lookup table that memorises training labels and cannot score an
#:   unseen repertoire, so that combination is rejected.
LABEL_INPUTS = ("sequence", "repertoire")


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
        theta_mode: str = "amortized",
        theta_pooling: str = "mean",
        label_input: str = "sequence",
        attention_hidden_dim: int = 64,
        n_v_genes: int = 0,
        n_j_genes: int = 0,
        vj_embedding_dim: int = 8,
    ):
        super().__init__()

        self.n_repertoires = n_repertoires
        self.latent_dim = latent_dim
        self.n_topics_signal = n_topics_signal
        self.n_topics_nonsignal = n_topics_nonsignal
        self.n_topics = self.n_topics_signal + self.n_topics_nonsignal

        if theta_mode not in THETA_MODES:
            raise ValueError(
                f"Unsupported theta_mode {theta_mode}, expected one of {THETA_MODES}"
            )
        self.theta_mode = theta_mode

        self.encoder = encoder
        self.decoder = decoder

        self.use_vj = n_v_genes > 0 and n_j_genes > 0
        self.n_v_genes = n_v_genes
        self.n_j_genes = n_j_genes
        self.vj_embedding_dim = vj_embedding_dim if self.use_vj else 0
        if self.use_vj:
            self.v_gene_embedding = torch.nn.Embedding(n_v_genes, vj_embedding_dim)
            self.j_gene_embedding = torch.nn.Embedding(n_j_genes, vj_embedding_dim)

        # Repertoire topic proportions.
        if self.theta_mode in ("free", "both"):
            self.repertoire_topic_proportions = torch.nn.Embedding(
                num_embeddings=self.n_repertoires, embedding_dim=self.n_topics
            )
        if theta_pooling not in THETA_POOLINGS:
            raise ValueError(
                f"Unsupported theta_pooling {theta_pooling}, expected one of "
                f"{THETA_POOLINGS}"
            )
        self.theta_pooling = theta_pooling
        if label_input not in LABEL_INPUTS:
            raise ValueError(
                f"Unsupported label_input {label_input}, expected one of {LABEL_INPUTS}"
            )
        if label_input == "repertoire" and theta_mode == "free":
            raise ValueError(
                "label_input='repertoire' with theta_mode='free' is the v1 shortcut: "
                "the label head reads a free per-repertoire embedding, which memorises "
                "training labels and cannot score an unseen repertoire"
            )
        self.label_input = label_input
        if self.theta_mode in ("amortized", "both"):
            if theta_pooling == "attention":
                self.theta_attention = TopicAttentionPooling(
                    input_dim=self._topic_input_dim(),
                    n_topics=self.n_topics,
                    hidden_dim=attention_hidden_dim,
                )
            else:
                self.repertoire_topic_head = torch.nn.Linear(
                    in_features=self._topic_input_dim(),
                    out_features=self.n_topics,
                    bias=True,
                )

        self.z_mean_layer = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=self.latent_dim,
            bias=True,
        )
        # Per-sequence posterior width. The previous implementation shared a single
        # scalar across every sequence and every latent dimension, which reduced the
        # KL term to a norm penalty on the mean and left generation isotropic.
        self.z_log_sigma_layer = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=self.latent_dim,
            bias=True,
        )

        self.latent_space_to_topic_proportions_layer = torch.nn.Linear(
            in_features=self._topic_input_dim(),
            out_features=self.n_topics,
            bias=True,
        )
        self.repertoire_label_prediction_layer = torch.nn.Linear(
            in_features=self.n_topics_signal,
            out_features=1,
            bias=True,
        )

    def _topic_input_dim(self) -> int:
        return self.latent_dim + 2 * self.vj_embedding_dim

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        repertoire_ids: torch.Tensor,
        x_sequence_SP: torch.Tensor,
        v_ids: torch.Tensor | None = None,
        j_ids: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        theta_sequences: torch.Tensor | None = None,
        theta_repertoire_ids: torch.Tensor | None = None,
        theta_v_ids: torch.Tensor | None = None,
        theta_j_ids: torch.Tensor | None = None,
        theta_weights: torch.Tensor | None = None,
    ) -> AIRRTM_ModelOutput:
        """
        ``theta_sequences`` and friends supply a *separate* sample of each repertoire
        from which to infer its topic proportions. Without them, Theta is pooled from
        the very sequences whose likelihood is then scored against it, so raising
        ``p(s | r)`` for one sequence moves ``theta_r`` and therefore lowers it for
        every other sequence in the bag -- the TM term ends up fighting itself, with a
        large gradient and no progress. Inferring Theta from a disjoint sample breaks
        that loop, and matches how amortized topic models are normally fit.
        """
        z_mean_SL, z_log_sigma_SL = self.sequence_to_latent_distribution(x_sequence_SP)
        z_SL = self._sample_latent(z_mean_SL, z_log_sigma_SL)
        kl_divergence_S = (
            -0.5
            * (1 + z_log_sigma_SL - z_mean_SL**2 - torch.exp(z_log_sigma_SL)).sum(dim=1)
            / self.latent_dim
        )

        topic_input_ST = self._with_vj(z_SL, v_ids, j_ids)
        seq_topic_logits_ST = self.latent_space_to_topic_proportions_layer(topic_input_ST)
        seq_topic_probabilities_ST = torch.sigmoid(seq_topic_logits_ST)

        theta_context = None
        if theta_sequences is not None:
            # The posterior mean, not a sample: Theta should not carry the VAE's
            # sampling noise into every sequence's likelihood.
            context_z_SL = self.sequence_to_latent(theta_sequences)
            theta_context = (
                self._with_vj(context_z_SL, theta_v_ids, theta_j_ids),
                theta_repertoire_ids,
                theta_weights,
            )
        log_topic_proportions_ST = self.log_topic_proportions(
            repertoire_ids=repertoire_ids,
            topic_input_ST=topic_input_ST,
            weights=weights,
            context=theta_context,
        )
        # Unnormalised log score of observing this sequence in its own repertoire:
        #   log sum_t theta_rt * phi_ts   with   phi_ts = exp(seq_topic_logits_ts).
        # The loss turns this into a normalised likelihood by dividing through a
        # partition function estimated over the sequences in the batch.
        tm_log_scores_S = torch.logsumexp(
            log_topic_proportions_ST + seq_topic_logits_ST, dim=1
        )
        # Kept for backwards compatibility and for reporting: the original bounded
        # "likelihood" sum_t theta_rt * sigmoid(phi_ts), which lies in (0, 1).
        tm_likelihoods_S = (
            torch.exp(log_topic_proportions_ST) * seq_topic_probabilities_ST
        ).sum(dim=1)

        if self.label_input == "repertoire":
            # Theta is constant within a repertoire, so this broadcasts one logit per
            # repertoire across its sequences; the MIL pooling in the loss then reduces
            # to the identity and the bag prediction is exactly this value.
            label_features_ST = torch.exp(log_topic_proportions_ST)
        else:
            label_features_ST = seq_topic_probabilities_ST
        seq_label_prediction_S = self.repertoire_label_prediction_layer(
            label_features_ST[:, : self.n_topics_signal]
        ).flatten()

        decoded_x_SPA = self.latent_to_sequence(z_SL)

        return AIRRTM_ModelOutput(
            kl_divergences=kl_divergence_S,
            tm_likelihoods=tm_likelihoods_S,
            tm_log_scores=tm_log_scores_S,
            seq_topic_logits=seq_topic_logits_ST,
            seq_topic_probabilities=seq_topic_probabilities_ST,
            log_topic_proportions=log_topic_proportions_ST,
            label_likelihoods=seq_label_prediction_S,
            decoded_sequences=decoded_x_SPA,
        )

    # ------------------------------------------------------------------ latents

    def sequence_to_latent_distribution(
        self, x_sequence_SP: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_SE = self.encoder(x_sequence_SP)
        return self.z_mean_layer(x_SE), self.z_log_sigma_layer(x_SE)

    def sequence_to_latent(self, x_sequence_SP: torch.Tensor) -> torch.Tensor:
        """Deterministic latent (the posterior mean), used for scoring."""
        z_mean_SL, _ = self.sequence_to_latent_distribution(x_sequence_SP)
        return z_mean_SL

    def latent_to_sequence(self, z_SL: torch.Tensor) -> torch.Tensor:
        """Per-position logits over the alphabet."""
        return self.decoder(z_SL)

    def _sample_latent(
        self, z_mean_SL: torch.Tensor, z_log_sigma_SL: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.randn_like(z_mean_SL)
        return z_mean_SL + eps * torch.exp(z_log_sigma_SL * 0.5)

    def _with_vj(
        self,
        z_SL: torch.Tensor,
        v_ids: torch.Tensor | None,
        j_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not getattr(self, "use_vj", False):
            return z_SL
        if v_ids is None or j_ids is None:
            raise ValueError("Model was built with V/J genes but none were supplied")
        return torch.concatenate(
            [z_SL, self.v_gene_embedding(v_ids), self.j_gene_embedding(j_ids)], dim=1
        )

    # -------------------------------------------------------- topic proportions

    def log_topic_proportions(
        self,
        repertoire_ids: torch.Tensor,
        topic_input_ST: torch.Tensor,
        weights: torch.Tensor | None = None,
        context: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Per-sequence ``log theta`` of the repertoire each sequence came from.

        ``context`` is an optional ``(topic_input, repertoire_ids, weights)`` triple
        from a disjoint sample of the same repertoires; when given, Theta is inferred
        from it instead of from the scored sequences themselves.
        """
        if context is None:
            logits_RT, _, inverse_S = self.repertoire_topic_logits(
                repertoire_ids, topic_input_ST, weights
            )
            return torch.log_softmax(logits_RT, dim=1)[inverse_S]

        context_input_ST, context_ids, context_weights = context
        logits_RT, unique_ids_R, _ = self.repertoire_topic_logits(
            context_ids, context_input_ST, context_weights
        )
        # torch.unique returns sorted ids, so searchsorted maps each scored sequence
        # onto the row inferred for its repertoire.
        rows_S = torch.searchsorted(unique_ids_R, repertoire_ids)
        # searchsorted returns len(unique_ids_R) for ids past the end, so clamp before
        # indexing and let the equality check below report the real problem.
        in_range_S = rows_S < unique_ids_R.shape[0]
        rows_S = rows_S.clamp(max=unique_ids_R.shape[0] - 1)
        if not in_range_S.all() or not torch.equal(
            unique_ids_R[rows_S], repertoire_ids
        ):
            raise ValueError(
                "Every scored repertoire must also appear in the Theta context sample"
            )
        return torch.log_softmax(logits_RT, dim=1)[rows_S]

    def repertoire_topic_logits(
        self,
        repertoire_ids: torch.Tensor,
        topic_input_ST: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Topic logits for each distinct repertoire present in the batch.

        Returns ``(logits_RT, unique_repertoire_ids_R, inverse_S)`` where ``inverse_S``
        maps every sequence onto its row of ``logits_RT``.
        """
        unique_ids_R, inverse_S = torch.unique(repertoire_ids, return_inverse=True)
        n_unique = unique_ids_R.shape[0]

        logits_RT = torch.zeros(
            n_unique,
            self.n_topics,
            dtype=topic_input_ST.dtype,
            device=topic_input_ST.device,
        )
        if self.theta_mode in ("amortized", "both"):
            logits_RT = logits_RT + self._pool_topic_logits(
                topic_input_ST, inverse_S, n_unique, weights
            )
        if self.theta_mode in ("free", "both"):
            logits_RT = logits_RT + self.repertoire_topic_proportions(unique_ids_R)
        return logits_RT, unique_ids_R, inverse_S

    def _pool_topic_logits(
        self,
        topic_input_ST: torch.Tensor,
        inverse_S: torch.Tensor,
        n_groups: int,
        weights_S: torch.Tensor | None,
    ) -> torch.Tensor:
        """Collapse a repertoire's sequences into one topic-logit vector."""
        if self.theta_pooling == "attention":
            log_weights_S = (
                None if weights_S is None else torch.log(weights_S.clamp(min=EPS))
            )
            return self.theta_attention(
                topic_input_ST, inverse_S, n_groups, log_weights_S
            )
        return _grouped_mean(
            self.repertoire_topic_head(topic_input_ST), inverse_S, n_groups, weights_S
        )

    def infer_repertoire_topic_proportions(
        self,
        x_sequence_SP: torch.Tensor,
        v_ids: torch.Tensor | None = None,
        j_ids: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Topic proportions for a single, possibly unseen, repertoire.

        ``x_sequence_SP`` is a sample of that repertoire's sequences. Requires a
        ``theta_mode`` with an amortized component.
        """
        if self.theta_mode == "free":
            raise ValueError(
                "theta_mode='free' cannot infer proportions for an unseen repertoire; "
                "use 'amortized' or 'both'"
            )
        z_mean_SL = self.sequence_to_latent(x_sequence_SP)
        topic_input_ST = self._with_vj(z_mean_SL, v_ids, j_ids)
        inverse_S = torch.zeros(
            topic_input_ST.shape[0], dtype=torch.long, device=topic_input_ST.device
        )
        logits_1T = self._pool_topic_logits(topic_input_ST, inverse_S, 1, weights)
        return torch.softmax(logits_1T, dim=1).flatten()

    def get_topic_proportion_matrix(self) -> torch.Tensor:
        """Topic proportions of every *training* repertoire (free component only)."""
        if self.theta_mode == "amortized":
            raise ValueError(
                "theta_mode='amortized' keeps no per-repertoire parameters; "
                "use infer_repertoire_topic_proportions instead"
            )
        topic_logits = self.repertoire_topic_proportions(
            torch.arange(self.n_repertoires, device=self._get_device())
        )
        return torch.softmax(topic_logits, dim=1)

    # ---------------------------------------------------------------- inference

    def predict_topic_probabilities(
        self,
        x_sequence_SP: torch.Tensor,
        v_ids: torch.Tensor | None = None,
        j_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sequence topic probabilities, on the same scale used during training."""
        z_mean_SL = self.sequence_to_latent(x_sequence_SP)
        topic_input_ST = self._with_vj(z_mean_SL, v_ids, j_ids)
        return torch.sigmoid(
            self.latent_space_to_topic_proportions_layer(topic_input_ST)
        )

    def predict_label_logits(
        self,
        x_sequence_SP: torch.Tensor,
        v_ids: torch.Tensor | None = None,
        j_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sequence label logits from the signal topics: the MIL instance score.

        Always the per-sequence view, even when ``label_input="repertoire"`` -- this
        is what ranks individual sequences by signal intensity, which is the model's
        actual output. The repertoire-level prediction is a separate quantity.
        """
        topic_probabilities_ST = self.predict_topic_probabilities(
            x_sequence_SP, v_ids, j_ids
        )
        return self.repertoire_label_prediction_layer(
            topic_probabilities_ST[:, : self.n_topics_signal]
        ).flatten()

    def predict_signal_intensity(
        self,
        x_sequence_SP: torch.Tensor,
        repertoire_labels_R: torch.Tensor | None = None,
        label_of_interest: int = 1,
        v_ids: torch.Tensor | None = None,
        j_ids: torch.Tensor | None = None,
        repertoire_topic_proportions_RT: torch.Tensor | None = None,
        method: str = "label_head",
    ) -> torch.Tensor:
        """Score how strongly each sequence carries the immune signal.

        ``method="label_head"`` uses the trained label head directly (a probability in
        (0, 1)). ``method="topic_weights"`` reproduces the paper's formulation: weight
        each signal topic by how much more of it positive repertoires use than
        negative ones, then take the dot product with the sequence's topic
        probabilities. The latter needs ``repertoire_labels_R`` and, for an amortized
        model, ``repertoire_topic_proportions_RT``.
        """
        if method == "label_head":
            return torch.sigmoid(self.predict_label_logits(x_sequence_SP, v_ids, j_ids))
        if method != "topic_weights":
            raise ValueError(f"Unsupported signal intensity method {method}")
        if repertoire_labels_R is None:
            raise ValueError("method='topic_weights' requires repertoire_labels_R")
        topic_signed_weights_T = self._compute_signal_topic_weights(
            repertoire_labels_R, label_of_interest, repertoire_topic_proportions_RT
        )
        topic_probabilities_ST = self.predict_topic_probabilities(
            x_sequence_SP, v_ids, j_ids
        )
        return (
            topic_probabilities_ST[:, : self.n_topics_signal] * topic_signed_weights_T
        ).sum(dim=1)

    def reconstruct(
        self, x_sequence_SP: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Round-trip a batch of sequences through the autoencoder, returning tokens."""
        z_mean_SL, z_log_sigma_SL = self.sequence_to_latent_distribution(x_sequence_SP)
        z_SL = z_mean_SL if deterministic else self._sample_latent(z_mean_SL, z_log_sigma_SL)
        return torch.argmax(self.latent_to_sequence(z_SL), dim=2)

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def _compute_signal_topic_weights(
        self,
        repertoire_labels_R: torch.Tensor,
        label_of_interest: int,
        repertoire_topic_proportions_RT: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if repertoire_topic_proportions_RT is None:
            repertoire_topic_proportions_RT = self.get_topic_proportion_matrix()
        signal_RT = repertoire_topic_proportions_RT[:, : self.n_topics_signal]
        is_positive_R = repertoire_labels_R == label_of_interest
        topic_differences_T = (
            signal_RT[is_positive_R].mean(dim=0) - signal_RT[~is_positive_R].mean(dim=0)
        )
        magnitudes_T = topic_differences_T.abs()
        return topic_differences_T / magnitudes_T.sum().clamp(min=EPS)


class TopicAttentionPooling(torch.nn.Module):
    """Gated attention pooling (Ilse et al. 2018), with one attention map per topic.

    Mean pooling asks *what is the average topic composition of this repertoire*.
    That is the wrong question when the label is carried by a handful of rare
    clonotypes: a repertoire holding 30 discriminative sequences out of 140,000 has
    essentially the same average as one holding 6, so the information is destroyed at
    the pooling step and no bag size, temperature or loss weight downstream can
    recover it.

    Attention pooling instead asks *what is the strongest evidence for topic t here*.
    Each topic gets its own attention distribution over the repertoire's sequences,
    so a topic can act as a detector for a small set of sequences rather than as a
    component of an average. Sharp attention approaches a max, flat attention
    recovers the mean, and the network chooses where to sit.
    """

    def __init__(self, input_dim: int, n_topics: int, hidden_dim: int = 64):
        super().__init__()
        self.value = torch.nn.Linear(input_dim, n_topics)
        self.attend = torch.nn.Linear(input_dim, hidden_dim)
        self.gate = torch.nn.Linear(input_dim, hidden_dim)
        self.score = torch.nn.Linear(hidden_dim, n_topics)

    def forward(
        self,
        h_SD: torch.Tensor,
        inverse_S: torch.Tensor,
        n_groups: int,
        log_weights_S: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values_ST = self.value(h_SD)
        gated_SH = torch.tanh(self.attend(h_SD)) * torch.sigmoid(self.gate(h_SD))
        attention_ST = _grouped_softmax(
            self.score(gated_SH), inverse_S, n_groups, log_weights_S
        )
        pooled_GT = torch.zeros(
            n_groups, values_ST.shape[1], dtype=values_ST.dtype, device=values_ST.device
        )
        return pooled_GT.index_add_(0, inverse_S, attention_ST * values_ST)


def _grouped_softmax(
    scores_SD: torch.Tensor,
    inverse_S: torch.Tensor,
    n_groups: int,
    log_weights_S: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax over the sequences within each group, one distribution per column.

    ``log_weights_S`` adds clonal abundance into the attention logits, so an expanded
    clone draws attention mass in proportion to its size.
    """
    if log_weights_S is not None:
        scores_SD = scores_SD + log_weights_S.reshape(-1, 1)
    index_SD = inverse_S.unsqueeze(1).expand_as(scores_SD)
    max_GD = torch.full(
        (n_groups, scores_SD.shape[1]),
        float("-inf"),
        dtype=scores_SD.dtype,
        device=scores_SD.device,
    ).scatter_reduce_(0, index_SD, scores_SD, reduce="amax", include_self=False)
    exponentiated_SD = (scores_SD - max_GD[inverse_S]).exp()
    totals_GD = torch.zeros_like(max_GD).index_add_(0, inverse_S, exponentiated_SD)
    return exponentiated_SD / totals_GD[inverse_S].clamp(min=EPS)


def _grouped_mean(
    values_SD: torch.Tensor,
    inverse_S: torch.Tensor,
    n_groups: int,
    weights_S: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean of ``values_SD`` within each group, optionally weighted.

    Permutation invariant, so it does not care how the batch is ordered, and it
    tolerates unequal group sizes.
    """
    if weights_S is None:
        weights_S1 = torch.ones(
            values_SD.shape[0], 1, dtype=values_SD.dtype, device=values_SD.device
        )
    else:
        weights_S1 = weights_S.to(values_SD.dtype).reshape(-1, 1)

    sums_GD = torch.zeros(
        n_groups, values_SD.shape[1], dtype=values_SD.dtype, device=values_SD.device
    ).index_add_(0, inverse_S, values_SD * weights_S1)
    counts_G1 = torch.zeros(
        n_groups, 1, dtype=values_SD.dtype, device=values_SD.device
    ).index_add_(0, inverse_S, weights_S1)
    return sums_GD / counts_G1.clamp(min=EPS)
