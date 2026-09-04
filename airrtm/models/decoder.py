import torch

# from x_transformers import TransformerWrapper, Decoder

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
            # num_tokens=self.alphabet_length + 1,  # +1 because pad value is a token too
            dim_in=self.attention_dim,  # VAE latent dim
            dim_out=self.alphabet_length + 1,  # +1 because pad value is a token too
            max_seq_len=self.max_sequence_length,
            attn_layers=Encoder(
                dim=self.attention_dim,
                attn_dim_head=self.attention_dim_head,
                depth=self.depth,
                heads=self.attention_heads,
                rotary_pos_emb=self.use_positional_encodings,
                attn_one_kv_head=False,
                unet_skips=False,  # (self.depth > 1),
                residual_attn=False,
                # cross_attend=True,
                cross_attend=False,
                # # layer_dropout=0,
                # # attn_dropout=self.dropout_rate,
                # # ff_dropout=self.dropout_rate,
            ),
            # return_only_embed=False,
        )
        # self.transformer = TransformerWrapper(
        #     num_tokens=self.alphabet_length + 1,  # +1 because pad value is a token too
        #     max_seq_len=self.max_sequence_length,
        #     attn_layers=Decoder(
        #         dim=self.attention_dim,
        #         attn_dim_head=self.attention_dim_head,
        #         depth=self.depth,
        #         heads=self.attention_heads,
        #         rotary_pos_emb=self.use_positional_encodings,
        #         attn_one_kv_head=False,
        #         unet_skips=False,  # (self.depth > 1),
        #         residual_attn=False,
        #         # cross_attend=True,
        #         cross_attend=False,
        #         # # layer_dropout=0,
        #         # # attn_dropout=self.dropout_rate,
        #         # # ff_dropout=self.dropout_rate,
        #     ),
        #     # return_only_embed=False,
        # )

    def forward(
        self,
        x_SL: torch.Tensor,
    ) -> torch.Tensor:
        # aa_probabilities_SPA = torch.nn.functional.softmax(x_SL, dim=2)
        # return aa_probabilities_SPA

        # starting_sequence_SP = self.generate_empty_starting_sequence(x_SL.shape[0]).to(
        #     x_SL.device
        # )
        # context_SPE = self.starting_linear_layer(x_SL).reshape(
        #     -1, self.max_sequence_length, self.attention_dim
        # )
        # return self.transformer(starting_sequence_SP, context=context_SPE)

        starting_sequence_SPE = self.starting_linear_layer(x_SL).reshape(
            -1, self.max_sequence_length, self.attention_dim
        )
        # starting_sequence_SPE = torch.nn.functional.relu(starting_sequence_SPE)
        transformer_output_SPA = self.transformer(starting_sequence_SPE)
        aa_probabilities_SPA = torch.nn.functional.softmax(
            transformer_output_SPA, dim=2
        )
        return aa_probabilities_SPA

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
        # self.lstm_model = torch.nn.LSTM(
        #     input_size=self.input_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     batch_first=True,
        #     bidirectional=True,
        # )

    # def forward(self, x_SLE, hidden):
    def forward(self, x_SL):
        x_SPL = x_SL.reshape(
            -1, self.max_sequence_length, self.input_dim // self.max_sequence_length
        )
        x_SPA = self.time_distributed_linear(x_SPL)
        return x_SPA
        # lstm_output, (hidden, cell) = self.lstm_model(x_SLE, hidden)
        # output = self.linear_output(lstm_output)
        # return output, (hidden, cell)
