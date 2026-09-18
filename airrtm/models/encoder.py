import torch

from x_transformers import TransformerWrapper, Encoder


class BaseEncoder(torch.nn.Module):
    """
    Suffix notation
    ---------------
    S: sequence in the dataset
    P: position in the sequence
    L: latent space dimension
    """

    def __init__(
        self,
        *,
        max_sequence_length: int,
        alphabet_length: int,
        **kwargs,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.alphabet_length = alphabet_length

    @property
    def pad_value(self) -> int:
        """Token id used for padding: one past the end of the alphabet."""
        return self.alphabet_length

    def padding_mask(self, x_SP: torch.Tensor) -> torch.Tensor:
        """True where the position holds a real residue."""
        return x_SP != self.pad_value

    def forward(self, x_SP: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerEncoder(BaseEncoder):
    """Transformer encoder over integer-encoded sequences.

    Parameters
    ----------
    pooling : str
        How the per-position embeddings are collapsed for the latent head.

        - ``"flatten"``: concatenate all positions (the original behaviour). Bakes
          absolute position into the latent and scales with ``max_sequence_length``.
        - ``"mean_max"``: concatenate the mask-aware mean and max over positions.
          Padding-invariant and length-invariant; preferred for new runs.
        - ``"mean"``: mask-aware mean only, half the output width of ``"mean_max"``.
          A simpler pooling for a smaller/cheaper encoder.
        - ``"attention"``: a single learned query attends over positions (mask-aware
          softmax), producing one weighted sum instead of an unweighted mean/max.
          Same output width as ``"mean"``.
    """

    def __init__(
        self,
        *,
        max_sequence_length: int,
        alphabet_length: int,
        attention_dim: int,
        attention_dim_head: int,
        attention_heads: int,
        depth: int,
        use_positional_encodings: bool = True,
        pooling: str = "flatten",
        mask_padding: bool = True,
    ):
        super().__init__(
            max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
        )
        self.attention_dim = attention_dim
        self.attention_dim_head = attention_dim_head
        self.attention_heads = attention_heads
        self.depth = depth
        self.use_positional_encodings = use_positional_encodings
        if pooling not in ("flatten", "mean_max", "mean", "attention"):
            raise ValueError(f"Unsupported pooling {pooling}")
        self.pooling = pooling
        self.mask_padding = mask_padding
        if pooling == "attention":
            self.attention_pool = _SequenceAttentionPooling(attention_dim)

        self.transformer = TransformerWrapper(
            num_tokens=self.alphabet_length + 1,  # +1 because pad value is a token too
            max_seq_len=self.max_sequence_length,
            attn_layers=Encoder(
                dim=self.attention_dim,
                attn_dim_head=self.attention_dim_head,
                depth=self.depth,
                heads=self.attention_heads,
                rotary_pos_emb=self.use_positional_encodings,
                attn_one_kv_head=False,
                unet_skips=(self.depth > 1),
                residual_attn=False,
            ),
            return_only_embed=True,
        )

    def forward(self, x_SP: torch.Tensor) -> torch.Tensor:
        # `getattr` keeps checkpoints pickled before these attributes existed loadable.
        mask = (
            self.padding_mask(x_SP) if getattr(self, "mask_padding", False) else None
        )
        x_SPE = self.transformer(x_SP, mask=mask)
        pooling = getattr(self, "pooling", "flatten")
        if pooling == "flatten":
            return x_SPE.reshape(x_SPE.shape[0], -1)
        if pooling == "mean":
            return _masked_mean(x_SPE, mask)
        if pooling == "attention":
            return self.attention_pool(x_SPE, mask)
        return _masked_mean_max(x_SPE, mask)

    def get_output_dim(self) -> int:
        pooling = getattr(self, "pooling", "flatten")
        if pooling == "flatten":
            return self.attention_dim * self.max_sequence_length
        if pooling in ("mean", "attention"):
            return self.attention_dim
        return self.attention_dim * 2


class _SequenceAttentionPooling(torch.nn.Module):
    """A single learned query attends over positions within one sequence.

    Unlike ``TopicAttentionPooling`` (one attention map per topic, over the
    variable-size set of sequences in a repertoire), this pools a fixed-length,
    padded position axis down to one vector per sequence -- so a plain masked
    softmax suffices, no grouped/ragged bookkeeping needed.
    """

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.score = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_SPE: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        scores_SP1 = self.score(x_SPE)
        if mask is not None:
            scores_SP1 = scores_SP1.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights_SP1 = torch.softmax(scores_SP1, dim=1)
        return (weights_SP1 * x_SPE).sum(dim=1)


def _masked_mean(x_SPE: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over positions, ignoring padded ones."""
    if mask is None:
        return x_SPE.mean(dim=1)
    mask_SP1 = mask.unsqueeze(-1).to(x_SPE.dtype)
    n_real_S1 = mask_SP1.sum(dim=1).clamp(min=1.0)
    return (x_SPE * mask_SP1).sum(dim=1) / n_real_S1


def _masked_mean_max(x_SPE: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Concatenate the mean and max over positions, ignoring padded ones."""
    mean_SE = _masked_mean(x_SPE, mask)
    if mask is None:
        max_SE = x_SPE.max(dim=1).values
    else:
        max_SE = x_SPE.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        # A row that is entirely padding would be all -inf; fall back to the mean there.
        max_SE = torch.where(torch.isfinite(max_SE), max_SE, mean_SE)
    return torch.concatenate([mean_SE, max_SE], dim=1)


class TransformerCNNEncoder(BaseEncoder):
    """A :class:`TransformerEncoder`, plus a single-layer CNN branch over the same
    sequence, concatenated.

    The CNN sees the same padded integer sequence through its own embedding and a
    single ``Conv1d`` (kernel ``cnn_kernel_size``, same-padding so sequence length is
    preserved), mask-aware mean+max pooled the same way as the transformer's
    ``"mean_max"`` pooling. A local n-gram-style filter is a different inductive bias
    than self-attention -- useful to try where a planted motif is a short, fixed-width
    pattern (e.g. the synthetic S1/S2 datasets) rather than requiring the transformer
    branch alone to discover it.
    """

    def __init__(
        self,
        *,
        max_sequence_length: int,
        alphabet_length: int,
        attention_dim: int,
        attention_dim_head: int,
        attention_heads: int,
        depth: int,
        use_positional_encodings: bool = True,
        pooling: str = "mean_max",
        mask_padding: bool = True,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 5,
    ):
        super().__init__(
            max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
        )
        self.transformer_encoder = TransformerEncoder(
            max_sequence_length=max_sequence_length,
            alphabet_length=alphabet_length,
            attention_dim=attention_dim,
            attention_dim_head=attention_dim_head,
            attention_heads=attention_heads,
            depth=depth,
            use_positional_encodings=use_positional_encodings,
            pooling=pooling,
            mask_padding=mask_padding,
        )
        self.mask_padding = mask_padding
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_embedding = torch.nn.Embedding(self.alphabet_length + 1, cnn_channels)
        self.cnn = torch.nn.Conv1d(
            cnn_channels, cnn_channels, kernel_size=cnn_kernel_size, padding="same"
        )

    def forward(self, x_SP: torch.Tensor) -> torch.Tensor:
        transformer_out_SE = self.transformer_encoder(x_SP)

        mask = self.padding_mask(x_SP) if self.mask_padding else None
        embedded_SPC = self.cnn_embedding(x_SP)
        conv_SPC = torch.relu(self.cnn(embedded_SPC.transpose(1, 2)).transpose(1, 2))
        cnn_out_SE = _masked_mean_max(conv_SPC, mask)

        return torch.concatenate([transformer_out_SE, cnn_out_SE], dim=1)

    def get_output_dim(self) -> int:
        return self.transformer_encoder.get_output_dim() + self.cnn_channels * 2


class LSTMEncoder(BaseEncoder):
    def __init__(
        self,
        *,
        max_sequence_length: int,
        alphabet_length: int,
        output_dim: int,
        depth: int,
    ):
        super().__init__(
            max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
        )
        self.output_dim = output_dim
        self.num_layers = depth
        self.time_distributed_linear = torch.nn.Linear(
            self.alphabet_length + 1,
            self.output_dim,
        )
        self.datatype = torch.float32

    def forward(self, x_SP):
        mask_SP = self.padding_mask(x_SP)
        x_SPA = torch.nn.functional.one_hot(x_SP, self.alphabet_length + 1).to(
            dtype=self.datatype
        )
        x_SPL = torch.nn.functional.relu(self.time_distributed_linear(x_SPA))
        return _masked_mean_max(x_SPL, mask_SP)

    def get_output_dim(self) -> int:
        return self.output_dim * 2
