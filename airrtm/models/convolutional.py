import math
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute masked mean over given dim.
    x: [B, N, D]
    mask: [B, N] bool or 0/1
    """
    mask = mask.float()
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    x = x * mask.unsqueeze(-1)
    return x.sum(dim=dim) / denom


def kl_divergence_diag_gaussians(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """KL(q(z|x) || p(z)) for diagonal Gaussians. Returns mean over batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()


class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise separable 1D conv residual block with optional causal padding and FiLM conditioning.

    Input/Output shape: [B, L, C] (channels-last for ease with LayerNorm)
    - Pre-LN
    - Optional FiLM: y = gamma * y + beta, where gamma,beta are [B, C] broadcast over L
    - Depthwise conv (causal or same), kernel_size=3 by default, dilation configurable
    - SiLU activation
    - Pointwise conv (1x1)
    - Dropout
    - Residual add
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = False,
        dropout: float = 0.0,
        use_film: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel size for 'same' padding"
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.use_film = use_film

        self.ln = nn.LayerNorm(dim)

        # depthwise conv (groups=dim)
        self.dw_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
            bias=True,
            padding=0,  # we manually pad in forward
        )
        # pointwise conv
        self.pw_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=True)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        # x: [B, L, C]
        residual = x
        y = self.ln(x)

        if self.use_film and film is not None:
            gamma, beta = film  # [B, C], [B, C]
            y = y * gamma.unsqueeze(1) + beta.unsqueeze(1)

        # to [B, C, L]
        y = y.transpose(1, 2)

        # padding for depthwise conv
        if self.causal:
            pad_left = (self.kernel_size - 1) * self.dilation
            y = F.pad(y, (pad_left, 0))
        else:
            pad_total = (self.kernel_size - 1) * self.dilation
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            y = F.pad(y, (pad_left, pad_right))

        y = self.dw_conv(y)  # [B, C, L]
        y = self.act(y)
        y = self.pw_conv(y)
        y = self.dropout(y)

        # back to [B, L, C]
        y = y.transpose(1, 2)
        return y + residual


class FiLMConditioner(nn.Module):
    """
    Map latent z -> per-layer FiLM parameters (gamma, beta) for a stack of decoder blocks.
    Produces [B, n_layers, 2*dim], split along last dimension to gamma/beta per layer.
    """

    def __init__(self, latent_dim: int, dim: int, n_layers: int, hidden_mult: int = 2):
        super().__init__()
        hidden = max(dim, latent_dim) * hidden_mult
        self.n_layers = n_layers
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_layers * 2 * dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [B, latent_dim]
        params = self.net(z)  # [B, n_layers * 2 * dim]
        params = params.view(z.size(0), self.n_layers, 2, self.dim)  # [B, L, 2, C]
        gamma = params[:, :, 0, :]  # [B, n_layers, C]
        beta = params[:, :, 1, :]  # [B, n_layers, C]
        return gamma, beta


class ConvSeqVAE(nn.Module):
    """
    Sequence VAE with CNN encoder and CNN autoregressive decoder (causal dilated convs + FiLM conditioning).

    - Encoder: non-causal depthwise-separable residual Conv1d blocks with dilations.
               Token + positional embeddings -> conv stack -> masked mean pool -> mu, logvar.
    - Latent: reparameterization z ~ N(mu, diag(sigma^2)).
    - Decoder: causal depthwise-separable residual Conv1d blocks with dilations.
               Teacher forcing during training: predict tokens[:, 1:] from tokens[:, :-1].
               FiLM conditioning with z applied at each decoder block.
    - Loss: recon CE + beta * KL.

    Args:
        vocab_size: number of tokens (e.g., 22)
        seq_len: maximum sequence length (e.g., 26)
        pad_token_id: padding token id
        stop_token_id: optional EOS token id for generation stopping
        dim: channel width
        enc_layers: number of encoder blocks
        dec_layers: number of decoder blocks
        kernel_size: convolution kernel size (odd)
        enc_dilations: list of dilations for encoder (len must equal enc_layers, or will be cycled)
        dec_dilations: list of dilations for decoder (len must equal dec_layers, or will be cycled)
        latent_dim: latent z dimension
        beta: KL weight
        dropout: dropout rate
        tie_decoder_embeddings: tie decoder token embedding to output projection
    """

    def __init__(
        self,
        vocab_size: int = 22,
        seq_len: int = 26,
        pad_token_id: int = 21,
        stop_token_id: Optional[int] = 20,
        dim: int = 256,
        enc_layers: int = 6,
        dec_layers: int = 8,
        kernel_size: int = 3,
        enc_dilations: Optional[List[int]] = None,
        dec_dilations: Optional[List[int]] = None,
        latent_dim: int = 128,
        beta: float = 0.0,
        dropout: float = 0.1,
        tie_decoder_embeddings: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.stop_token_id = stop_token_id
        self.dim = dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.tie_decoder_embeddings = tie_decoder_embeddings

        # Embeddings
        self.encoder_tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.decoder_tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.pos_emb = nn.Parameter(
            torch.zeros(seq_len, dim)
        )  # shared pos emb for simplicity

        # Encoder stack (non-causal)
        if enc_dilations is None:
            enc_dilations = [1, 2, 4, 8]
        enc_dilations = (
            enc_dilations
            * ((enc_layers + len(enc_dilations) - 1) // len(enc_dilations))
        )[:enc_layers]
        self.encoder_blocks = nn.ModuleList(
            [
                DepthwiseSeparableConvBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    causal=False,
                    dropout=dropout,
                    use_film=False,
                )
                for d in enc_dilations
            ]
        )

        # Latent heads
        self.enc_ln = nn.LayerNorm(dim)
        self.to_mu = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, latent_dim)
        )
        self.to_logvar = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, latent_dim)
        )

        # Decoder conditioning from z
        self.film = FiLMConditioner(
            latent_dim=latent_dim, dim=dim, n_layers=dec_layers, hidden_mult=2
        )

        # Decoder stack (causal)
        if dec_dilations is None:
            dec_dilations = [1, 2, 4, 8]
        dec_dilations = (
            dec_dilations
            * ((dec_layers + len(dec_dilations) - 1) // len(dec_dilations))
        )[:dec_layers]
        self.decoder_blocks = nn.ModuleList(
            [
                DepthwiseSeparableConvBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    causal=True,
                    dropout=dropout,
                    use_film=True,
                )
                for d in dec_dilations
            ]
        )
        self.dec_ln = nn.LayerNorm(dim)

        # Output projection; tie weights with decoder embedding if requested
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        if self.tie_decoder_embeddings:
            # tie by making forward use decoder_tok_emb.weight
            # we keep out_proj for shape/reference, but will use F.linear with tied weight in forward
            nn.init.zeros_(self.out_proj.weight)  # not used if tied

        # init
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        # Embeddings default init is fine; convs are initialized by PyTorch

    def encode(
        self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode tokens to pooled representation and VAE posterior parameters.
        tokens: [B, L]
        attn_mask: [B, L] bool, True for valid (non-pad)
        """
        B, L = tokens.shape
        if attn_mask is None:
            attn_mask = tokens != self.pad_token_id

        # embeddings + pos
        x = self.encoder_tok_emb(
            tokens
        )  # [B, L, C]; padding_idx -> zeros for pad tokens
        pos = self.pos_emb[:L].unsqueeze(0)  # [1, L, C]
        x = x + pos

        # non-causal conv stack
        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.enc_ln(x)  # [B, L, C]

        # masked mean pool over length
        pooled = masked_mean(x, attn_mask, dim=1)  # [B, C]

        mu = self.to_mu(pooled)  # [B, latent_dim]
        logvar = self.to_logvar(pooled)  # [B, latent_dim]

        return {"pooled": pooled, "mu": mu, "logvar": logvar}

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        if not sample:
            return mu
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_forward(
        self,
        tokens_in: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the causal CNN decoder with teacher forcing inputs.
        tokens_in: [B, L_in] (e.g., tokens[:, :-1])
        z: [B, latent_dim]
        Returns logits over vocab: [B, L_in, vocab]
        """
        B, L = tokens_in.shape

        h = self.decoder_tok_emb(tokens_in) + self.pos_emb[:L].unsqueeze(0)  # [B, L, C]

        # get FiLM params for each decoder block
        gammas, betas = self.film(z)  # [B, n_layers, C] each

        for i, blk in enumerate(self.decoder_blocks):
            film = (gammas[:, i, :], betas[:, i, :])
            h = blk(h, film=film)

        h = self.dec_ln(h)  # [B, L, C]

        if self.tie_decoder_embeddings:
            logits = F.linear(h, self.decoder_tok_emb.weight)  # [B, L, vocab]
        else:
            logits = self.out_proj(h)
        return logits

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kl_weight: Optional[float] = None,
        sample_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss = recon CE + beta * KL.
        tokens: [B, L]
        """
        if attn_mask is None:
            attn_mask = tokens != self.pad_token_id

        # encode -> mu, logvar
        enc = self.encode(tokens, attn_mask=attn_mask)
        mu, logvar = enc["mu"], enc["logvar"]

        # sample or use mean
        z = self.reparameterize(mu, logvar, sample=sample_posterior)

        # teacher forcing: predict next token
        tokens_in = tokens[:, :-1]  # [B, L-1]
        targets = tokens[:, 1:]  # [B, L-1]

        logits = self.decoder_forward(tokens_in, z)  # [B, L-1, vocab]

        # cross-entropy with ignore_index for pads in targets
        loss_recon = F.cross_entropy(
            logits.transpose(1, 2),  # [B, vocab, L-1]
            targets,
            ignore_index=self.pad_token_id,
        )

        acc_recon = (torch.argmax(logits, dim=2) == targets).to(torch.float64).mean()

        # KL
        kl = kl_divergence_diag_gaussians(mu, logvar)
        beta = self.beta if kl_weight is None else kl_weight
        total = loss_recon + beta * kl

        return {
            "total_loss": total,
            "recon_loss": loss_recon.detach(),
            "recon_acc": acc_recon.detach(),
            "kl_loss": kl.detach(),
            "mu": mu.detach(),
            "logvar": logvar.detach(),
        }

    @torch.no_grad()
    def reconstruct(
        self,
        tokens: torch.Tensor,
        deterministic: bool = True,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode -> z -> decode autoregressively.
        If deterministic, use mu (no sampling).
        """
        device = tokens.device
        B, L = tokens.shape
        length = L if max_len is None else max_len

        attn_mask = tokens != self.pad_token_id
        enc = self.encode(tokens, attn_mask=attn_mask)
        z = self.reparameterize(enc["mu"], enc["logvar"], sample=not deterministic)

        # start with a single pad token as BOS surrogate
        cur = torch.full(
            (B, 1), fill_value=self.pad_token_id, dtype=torch.long, device=device
        )

        eos = eos_token_id if eos_token_id is not None else self.stop_token_id

        for _ in range(length - 1):
            logits = self.decoder_forward(cur, z)  # [B, t, vocab]
            next_logits = logits[:, -1, :]  # [B, vocab]
            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-8)

            probs = next_logits.softmax(dim=-1)
            next_tok = probs.argmax(dim=-1)  # greedy by default
            # For stochastic sampling:
            # next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            cur = torch.cat([cur, next_tok.unsqueeze(1)], dim=1)

            if eos is not None:
                # stop early if all sequences emitted EOS
                if ((cur[:, -1] == eos) | (cur.size(1) >= length)).all():
                    break

        # pad to fixed length if needed
        if cur.size(1) < length:
            pad = torch.full(
                (B, length - cur.size(1)),
                self.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            cur = torch.cat([cur, pad], dim=1)

        return cur[:, :length]

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        deterministic: bool = False,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample z ~ N(0, I) then decode autoregressively.
        """
        if device is None:
            device = next(self.parameters()).device

        length = self.seq_len if max_len is None else max_len
        z = (
            torch.zeros(batch_size, self.latent_dim, device=device)
            if deterministic
            else torch.randn(batch_size, self.latent_dim, device=device)
        )

        # start with pad as BOS surrogate
        cur = torch.full(
            (batch_size, 1), self.pad_token_id, dtype=torch.long, device=device
        )
        eos = eos_token_id if eos_token_id is not None else self.stop_token_id

        for _ in range(length - 1):
            logits = self.decoder_forward(cur, z)  # [B, t, vocab]
            next_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-8)
            probs = next_logits.softmax(dim=-1)
            next_tok = probs.argmax(
                dim=-1
            )  # greedy; switch to multinomial for sampling
            # next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            cur = torch.cat([cur, next_tok.unsqueeze(1)], dim=1)

            if eos is not None:
                if ((cur[:, -1] == eos) | (cur.size(1) >= length)).all():
                    break

        if cur.size(1) < length:
            pad = torch.full(
                (batch_size, length - cur.size(1)),
                self.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            cur = torch.cat([cur, pad], dim=1)

        return cur[:, :length]


# ---------------------------
# Example usage / training
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 22  # 20 AAs + stop + pad
    seq_len = 26
    pad_token_id = 21
    stop_token_id = 20

    model = ConvSeqVAE(
        vocab_size=vocab_size,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        stop_token_id=stop_token_id,
        dim=256,
        enc_layers=6,
        dec_layers=8,
        kernel_size=3,
        enc_dilations=[1, 2, 4, 8],
        dec_dilations=[1, 2, 4, 8],
        latent_dim=128,
        beta=0.0,  # start at 0, anneal later
        dropout=0.1,
        tie_decoder_embeddings=True,
    ).to(device)

    # Dummy batch [B, L]
    B = 16
    x = torch.randint(
        0, vocab_size - 2, (B, seq_len), device=device
    )  # avoid pad/stop in random init
    x[0, -3:] = pad_token_id
    x[1, -5:] = pad_token_id

    opt = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1500))

    model.train()
    for step in range(1500):
        out = model(x, sample_posterior=False)
        loss = out["loss"]

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(
                f"step {step} | loss {loss.item():.3f} | recon {out['recon_loss'].item():.3f} | kl {out['kl_loss'].item():.4f}"
            )

    # Inference
    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(x, deterministic=True)
        samp = model.sample(
            batch_size=4, deterministic=False, temperature=1.0, device=device
        )

    print("Reconstruction shape:", recon.shape)
    print("Sampled shape:", samp.shape)
