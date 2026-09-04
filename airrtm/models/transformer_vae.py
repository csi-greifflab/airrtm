from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import TransformerWrapper, Encoder, Decoder, AutoregressiveWrapper


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute masked mean over given dim.
    x: [B, N, D]
    mask: [B, N] with 1 for valid tokens and 0 for pad
    """
    mask = mask.float()
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    x = x * mask.unsqueeze(-1)
    return x.sum(dim=dim) / denom


def kl_divergence_diag_gaussians(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) where q is N(mu, diag(sigma^2)) and p is N(0, I).
    Returns mean KL per batch element.
    """
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # [B]
    return kl.mean()


class SeqVAE(nn.Module):
    """
    Sequence VAE using x-transformers for encoder and decoder.

    - Encoder: non-causal transformer encoder over input tokens, masked on pad.
    - Latent: masked pooled encoder representation -> mu, logvar -> z.
    - Decoder: causal transformer decoder with cross-attention to latent "memory tokens".
               Wrapped with AutoregressiveWrapper for next-token prediction loss and generation.

    Args:
        vocab_size: number of discrete tokens (e.g., 22)
        seq_len: fixed maximum sequence length (e.g., 26)
        pad_token_id: id used for padding tokens
        stop_token_id: optional id for a stop token (can be used for early stopping at generation)
        dim: transformer model dimension
        depth_enc: number of encoder layers
        depth_dec: number of decoder layers
        heads: attention heads
        ff_mult: feedforward expansion multiplier
        latent_dim: VAE latent dimension
        n_latent_tokens: number of latent "memory tokens" to feed to decoder via cross-attn
        beta: default KL weight
        dropout: dropout for embeddings/attn if desired
    """

    def __init__(
        self,
        vocab_size: int = 22,
        seq_len: int = 26,
        pad_token_id: int = 21,
        stop_token_id: Optional[int] = 20,
        dim: int = 256,
        depth_enc: int = 4,
        depth_dec: int = 4,
        heads: int = 8,
        ff_mult: int = 4,
        latent_dim: int = 64,
        n_latent_tokens: int = 4,
        beta: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.stop_token_id = stop_token_id
        self.dim = dim
        self.latent_dim = latent_dim
        self.n_latent_tokens = n_latent_tokens
        self.beta = beta

        # Encoder: non-causal transformer
        self.encoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=seq_len,
            emb_dim=dim,
            attn_layers=Encoder(
                dim=dim,
                depth=depth_enc,
                heads=heads,
                # ff_mult=ff_mult,
                attn_dropout=dropout,
                ff_dropout=dropout,
            ),
            emb_dropout=dropout,
        )

        # Decoder: causal transformer with cross-attention
        self.decoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=seq_len,
            emb_dim=dim,
            attn_layers=Decoder(
                dim=dim,
                depth=depth_dec,
                heads=heads,
                # ff_mult=ff_mult,
                cross_attend=True,
                attn_dropout=dropout,
                ff_dropout=dropout,
            ),
            emb_dropout=dropout,
        )

        # Autoregressive wrapper handles teacher forcing loss and generation
        # ignore_index ensures pad positions are not contributing to CE loss
        self.ar_decoder = AutoregressiveWrapper(
            self.decoder, ignore_index=self.pad_token_id
        )

        # Latent heads: pooled encoder -> mu, logvar
        self.to_mu = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, latent_dim),
        )
        self.to_logvar = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, latent_dim),
        )

        # Project latent to memory tokens consumed by decoder cross-attention
        self.latent_to_mem = nn.Sequential(
            nn.Linear(latent_dim, n_latent_tokens * dim),
            nn.SiLU(),
        )

    def encode(
        self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode tokens to posterior parameters (mu, logvar) and pooled embedding.

        tokens: [B, N] long
        attn_mask: [B, N] bool, True for valid tokens. If None, uses tokens != pad.
        """
        if attn_mask is None:
            attn_mask = tokens != self.pad_token_id  # [B, N], bool

        # Get encoder embeddings; return_embeddings=True yields [B, N, D]
        enc_emb = self.encoder(
            tokens, mask=attn_mask, return_embeddings=True
        )  # [B, N, D]

        # Masked mean pooling over sequence length
        pooled = masked_mean(enc_emb, attn_mask, dim=1)  # [B, D]

        mu = self.to_mu(pooled)  # [B, latent_dim]
        logvar = self.to_logvar(pooled)  # [B, latent_dim]

        return {"pooled": pooled, "mu": mu, "logvar": logvar}

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps
        """
        if not sample:
            return mu
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_to_context(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map z to a set of cross-attendable memory tokens for the decoder.
        z: [B, latent_dim]
        returns memory: [B, n_latent_tokens, D]
        """
        mem = self.latent_to_mem(z)  # [B, n_latent_tokens * D]
        mem = mem.view(z.size(0), self.n_latent_tokens, self.dim)  # [B, Lm, D]
        return mem

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kl_weight: Optional[float] = None,
        sample_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss = recon + beta * KL.

        tokens: [B, N] long
        attn_mask: [B, N] bool, True for valid tokens
        kl_weight: override default beta if provided
        sample_posterior: whether to sample z or use mu (useful for eval)

        Returns dict with total_loss, recon_loss, kl_loss, mu, logvar
        """
        if attn_mask is None:
            attn_mask = tokens != self.pad_token_id

        enc_out = self.encode(tokens, attn_mask=attn_mask)
        mu, logvar = enc_out["mu"], enc_out["logvar"]

        z = self.reparameterize(mu, logvar, sample=sample_posterior)
        context = self.latent_to_context(z)  # [B, Lm, D]

        # AutoregressiveWrapper computes cross-entropy loss with teacher forcing
        # It uses tokens[:, :-1] -> inputs, tokens[:, 1:] -> targets under the hood.
        # ignore_index=pad_token_id means pad in targets is ignored.
        recon_loss = self.ar_decoder(tokens, context=context)

        kl_loss = kl_divergence_diag_gaussians(mu, logvar)

        beta = self.beta if kl_weight is None else kl_weight
        total = recon_loss + beta * kl_loss

        return {
            "total_loss": total,
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
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
        Encode an input sequence to z and decode/generate a reconstruction.
        If deterministic=True, uses mu (no sampling). Otherwise samples z.

        Returns generated tokens [B, L]
        """
        device = tokens.device
        attn_mask = tokens != self.pad_token_id

        enc_out = self.encode(tokens, attn_mask=attn_mask)
        z = self.reparameterize(
            enc_out["mu"], enc_out["logvar"], sample=not deterministic
        )
        context = self.latent_to_context(z)

        # Start tokens for generation; we can start with a single pad token as a BOS surrogate
        B = tokens.size(0)
        start = torch.full(
            (B, 1), fill_value=self.pad_token_id, dtype=torch.long, device=device
        )

        gen_len = max_len if max_len is not None else self.seq_len

        # If you have a valid stop_token_id and want early stopping:
        eos = eos_token_id if eos_token_id is not None else self.stop_token_id

        generated = self.ar_decoder.generate(
            start,
            seq_len=gen_len,
            context=context,
            temperature=temperature,
            eos_token=eos,
        )
        return generated

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
        Sample sequences from the prior p(z) ~ N(0, I) and decode.

        Returns [B, L]
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample z from the prior
        if deterministic:
            z = torch.zeros(batch_size, self.latent_dim, device=device)
        else:
            z = torch.randn(batch_size, self.latent_dim, device=device)

        context = self.latent_to_context(z)

        start = torch.full(
            (batch_size, 1),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        gen_len = max_len if max_len is not None else self.seq_len
        eos = eos_token_id if eos_token_id is not None else self.stop_token_id

        samples = self.ar_decoder.generate(
            start,
            seq_len=gen_len,
            context=context,
            temperature=temperature,
            eos_token=eos,
        )
        return samples


# # ---------------------------
# # Example usage / training
# # ---------------------------
# if __name__ == "__main__":
#     # Hyperparameters
#     vocab_size = 22       # 20 AAs + stop + pad
#     seq_len = 26
#     pad_token_id = 21
#     stop_token_id = 20

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = SeqVAE(
#         vocab_size=vocab_size,
#         seq_len=seq_len,
#         pad_token_id=pad_token_id,
#         stop_token_id=stop_token_id,
#         dim=256,
#         depth_enc=4,
#         depth_dec=4,
#         heads=8,
#         ff_mult=4,
#         latent_dim=64,
#         n_latent_tokens=4,
#         beta=0.5,    # you may anneal this from 0 -> 1 over training
#         dropout=0.1,
#     ).to(device)

#     # Dummy batch of tokenized protein sequences [B, N]
#     B = 8
#     x = torch.randint(low=0, high=vocab_size - 2, size=(B, seq_len), device=device)  # avoid using pad & stop randomly
#     # Add some padding at the tail for a few sequences
#     x[0, -3:] = pad_token_id
#     x[1, -5:] = pad_token_id

#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

#     model.train()
#     for step in range(100):
#         optimizer.zero_grad()

#         # Optional KL annealing
#         # e.g., linear anneal from 0 -> model.beta over first 10k steps
#         # kl_w = min(1.0, step / 10000) * model.beta
#         kl_w = None  # use model.beta

#         out = model(tokens=x, kl_weight=kl_w, sample_posterior=True)
#         loss = out["loss"]
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         if step % 10 == 0:
#             print(f"step {step} | loss {loss.item():.4f} | recon {out['recon_loss'].item():.4f} | kl {out['kl_loss'].item():.4f}")

#     # Inference: reconstruct and sample
#     model.eval()
#     with torch.no_grad():
#         recon = model.reconstruct(x, deterministic=True)   # use mu for z
#         samp = model.sample(batch_size=4, temperature=1.0)

#     print("Reconstruction shape:", recon.shape)
#     print("Sampled shape:", samp.shape)
