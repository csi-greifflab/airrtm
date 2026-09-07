import torch

from x_transformers import ContinuousTransformerWrapper, Encoder


class BaseDecoder(torch.nn.Module):
    """
    Suffix notation
    ---------------
    S: sequence in the dataset
    P: position in the sequence
    A: index of the amino acid
    L: latent space dimension
    E: attention (embedding) dimension

    Decoders return **logits** over the alphabet, not probabilities: the
    reconstruction loss applies its own ``log_softmax``, and applying softmax here
    too would flatten the gradients.
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

    def forward(self, x_SL: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerDecoder(BaseDecoder):
    def __init__(
        self,
        *,
        max_sequence_length: int,
        input_dim: int,
        alphabet_length: int,
        attention_dim: int,
        attention_dim_head: int,
        attention_heads: int,
        depth: int,
        use_positional_encodings: bool = True,
    ):
        super().__init__(
            max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
        )
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attention_dim_head = attention_dim_head
        self.attention_heads = attention_heads
        self.depth = depth
        self.use_positional_encodings = use_positional_encodings

        self.starting_linear_layer = torch.nn.Linear(
            in_features=input_dim,
            out_features=self.max_sequence_length * self.attention_dim,
        )

        self.transformer = ContinuousTransformerWrapper(
            dim_in=self.attention_dim,
            dim_out=self.alphabet_length + 1,  # +1 because pad value is a token too
            max_seq_len=self.max_sequence_length,
            attn_layers=Encoder(
                dim=self.attention_dim,
                attn_dim_head=self.attention_dim_head,
                depth=self.depth,
                heads=self.attention_heads,
                rotary_pos_emb=self.use_positional_encodings,
                attn_one_kv_head=False,
                unet_skips=False,
                residual_attn=False,
                cross_attend=False,
            ),
        )

    def forward(
        self,
        x_SL: torch.Tensor,
    ) -> torch.Tensor:
        starting_sequence_SPE = self.starting_linear_layer(x_SL).reshape(
            -1, self.max_sequence_length, self.attention_dim
        )
        # Logits, deliberately: see the note on BaseDecoder.
        return self.transformer(starting_sequence_SPE)

    def generate_empty_starting_sequence(self, batch_size) -> torch.Tensor:
        return torch.full(
            fill_value=self.alphabet_length,
            size=(batch_size, self.max_sequence_length),
        )


class LSTMDecoder(BaseDecoder):
    def __init__(
        self,
        *,
        max_sequence_length: int,
        alphabet_length: int,
        input_dim: int,
        depth: int,
    ):
        super().__init__(
            max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
        )
        self.input_dim = input_dim
        self.output_size = alphabet_length
        self.num_layers = depth
        self.time_distributed_linear = torch.nn.Linear(
            self.input_dim // self.max_sequence_length, self.alphabet_length + 1
        )

    def forward(self, x_SL):
        x_SPL = x_SL.reshape(
            -1, self.max_sequence_length, self.input_dim // self.max_sequence_length
        )
        return self.time_distributed_linear(x_SPL)
