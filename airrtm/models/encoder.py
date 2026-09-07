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
        if pooling not in ("flatten", "mean_max"):
            raise ValueError(f"Unsupported pooling {pooling}")
        self.pooling = pooling
        self.mask_padding = mask_padding

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
        return _masked_mean_max(x_SPE, mask)

    def get_output_dim(self) -> int:
        if getattr(self, "pooling", "flatten") == "flatten":
            return self.attention_dim * self.max_sequence_length
        return self.attention_dim * 2


def _masked_mean_max(x_SPE: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Concatenate the mean and max over positions, ignoring padded ones."""
    if mask is None:
        return torch.concatenate([x_SPE.mean(dim=1), x_SPE.max(dim=1).values], dim=1)
    mask_SP1 = mask.unsqueeze(-1).to(x_SPE.dtype)
    n_real_S1 = mask_SP1.sum(dim=1).clamp(min=1.0)
    mean_SE = (x_SPE * mask_SP1).sum(dim=1) / n_real_S1
    max_SE = x_SPE.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
    # A row that is entirely padding would be all -inf; fall back to the mean there.
    max_SE = torch.where(torch.isfinite(max_SE), max_SE, mean_SE)
    return torch.concatenate([mean_SE, max_SE], dim=1)


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
