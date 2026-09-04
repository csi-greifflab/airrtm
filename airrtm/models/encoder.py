import torch

from x_transformers import TransformerWrapper, Encoder
from xlstm import xLSTM


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

    def forward(self, x_SP: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerEncoder(BaseEncoder):
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
    ):
        super().__init__(
            max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
        )
        self.attention_dim = attention_dim
        self.attention_dim_head = attention_dim_head
        self.attention_heads = attention_heads
        self.depth = depth
        self.use_positional_encodings = use_positional_encodings

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
                # layer_dropout=0,
                # attn_dropout=self.dropout_rate,
                # ff_dropout=self.dropout_rate,
            ),
            return_only_embed=True,
        )

    def forward(self, x_SP: torch.Tensor) -> torch.Tensor:
        return self.transformer(x_SP)

    def get_output_dim(self) -> int:
        return self.attention_dim * self.max_sequence_length


# class CNNEncoder(BaseEncoder):
#     def __init__(
#         self,
#         *,
#         max_sequence_length: int,
#         alphabet_length: int,
#         attention_dim: int,
#         attention_dim_head: int,
#         attention_heads: int,
#         encoder_depth: int,
#         use_positional_encodings: bool = True,
#     ):
#         super().__init__(
#             max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
#         )


# class XLSTMncoder(BaseEncoder):
#     def __init__(
#         self,
#         *,
#         max_sequence_length: int,
#         alphabet_length: int,
#         dim: int,
#         dim_head: int,
#         n_heads: int,
#         depth: int,
#         kernel_size: int,
#     ):
#         super().__init__(
#             max_sequence_length=max_sequence_length, alphabet_length=alphabet_length
#         )

#         self.depth = depth
#         self.dim = dim
#         self.dim_head = dim_head
#         self.n_heads = n_heads
#         self.kernel_size = kernel_size
#         self.signature = (7, 1)
#         self.p_factor = (2, 4 / 3)

#         self.xlstm_model = xLSTM(
#             vocab_size=alphabet_length + 1,
#             num_layers=self.depth,
#             signature=self.signature,
#             inp_dim=self.dim,
#             head_dim=self.dim_head,
#             head_num=self.n_heads,
#             p_factor=self.p_factor,
#             ker_size=self.kernel_size,
#         )

#     def forward(self, x_SP: torch.Tensor) -> torch.Tensor:
#         return self.xlstm_model(x_SP)


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
            self.output_dim // self.max_sequence_length,
        )

        self.datatype = torch.float32
        # self.lstm_model = torch.nn.LSTM(
        #     input_size=alphabet_length,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     batch_first=True,
        #     bidirectional=True,
        # )

    def forward(self, x_SP):
        x_SPA = torch.nn.functional.one_hot(x_SP, self.alphabet_length + 1).to(
            dtype=self.datatype
        )
        x_SPL = torch.nn.functional.relu(self.time_distributed_linear(x_SPA))
        # x_SL = x_SPL.reshape(-1, self.output_dim)
        x_SL_mean = x_SPL.mean(dim=1)
        x_SL_max = x_SPL.max(dim=1)
        x_SL = torch.concatenate(x_SL_mean, x_SL_max, dim=1)

        return x_SL
        # output, (hidden, cell) = self.lstm_model(x_SPA)
        # return (hidden, cell)

    def get_output_dim(self) -> int:
        return self.max_sequence_length * self.output_dim

    # # Parameters for the inference
    # token_lim = 16
    # use_top_k = 50
    # temperature = 0.7

    # # Generate text
    # stream = model.generate(
    #     # We can provide more than one prompt!
    #     prompt=[
    #         "Once upon a time",
    #         "In a galaxy far far away",
    #     ],
    #     tokenizer=tokenizer,
    #     token_lim=token_lim,
    #     use_top_k=use_top_k,
    #     temperature=temperature,
    # )

    # for token in stream:
    #     # Each token is a dictionary indexed by the
    #     # batch-id and contains the produced string
    #     # as value, so we can print the first batch as:
    #     print(token[0], end="")
