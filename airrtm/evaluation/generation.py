"""Generate novel signal-associated sequences (paper, Methods 5).

Topics are directions in the VAE latent space, but the KL term keeps all embeddings
near the origin, so walking far along a topic direction decodes to nothing sequence-
like. The paper's approach, reproduced here: take the top-scoring identified
sequences, fit a diagonal Gaussian to their latents, and sample from it with a
temperature that widens the covariance.
"""

import numpy as np
import torch

from airrtm.models import AIRRTM_Model
from airrtm.utils import SequenceDataset


__all__ = ["signal_latent_distribution", "generate_sequences", "generation_report"]


@torch.no_grad()
def signal_latent_distribution(
    model: AIRRTM_Model,
    dataset: SequenceDataset,
    scores: torch.Tensor,
    k: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and per-dimension std of the latents of the top-``k`` scoring sequences."""
    device = device or next(model.parameters()).device
    model.eval()
    top_indices = torch.topk(scores.flatten(), k=min(k, scores.numel())).indices
    latents = model.sequence_to_latent(
        dataset.data[top_indices].to(torch.long).to(device)
    )
    return latents.mean(dim=0), latents.std(dim=0)


@torch.no_grad()
def generate_sequences(
    model: AIRRTM_Model,
    mean_L: torch.Tensor,
    std_L: torch.Tensor,
    n_sequences: int = 100_000,
    temperature: float = 1.0,
    batch_size: int = 8192,
    device: torch.device | None = None,
) -> list[str]:
    """Decode samples from ``N(mean, diag((std * temperature)^2))`` into strings.

    Higher temperatures give more diverse sequences at lower precision -- the paper
    sweeps ``temperature`` in [0, 2] and reports the trade-off.
    """
    device = device or next(model.parameters()).device
    model.eval()
    alphabet = model.decoder.alphabet_length
    pad_value = alphabet  # one past the alphabet, as everywhere else

    generated = []
    for start in range(0, n_sequences, batch_size):
        size = min(batch_size, n_sequences - start)
        z_SL = mean_L.unsqueeze(0) + torch.randn(
            size, mean_L.shape[0], device=device
        ) * (std_L.unsqueeze(0) * temperature)
        tokens_SP = torch.argmax(model.latent_to_sequence(z_SL), dim=2).cpu()
        generated.extend(_tokens_to_strings(tokens_SP, pad_value))
    return generated


def _tokens_to_strings(tokens_SP: torch.Tensor, pad_value: int) -> list[str]:
    from airrtm.utils.constants import AA_ALPHABET

    strings = []
    for row in tokens_SP.tolist():
        symbols = []
        for token in row:
            if token == pad_value:
                break
            symbols.append(AA_ALPHABET[token])
        strings.append("".join(symbols))
    return strings


def generation_report(
    generated: list[str],
    training_sequences: set[str],
    is_signal=None,
) -> dict[str, float]:
    """Novelty, uniqueness and (when a signal oracle is available) precision.

    ``is_signal`` is a callable mapping a sequence string to a bool -- for Emerson
    there is no ground-truth oracle, so pass ``None`` and read novelty/uniqueness
    only; for the synthetic datasets it is the k-mer rule.
    """
    non_empty = [s for s in generated if s]
    unique = set(non_empty)
    report = {
        "n_generated": len(generated),
        "n_non_empty": len(non_empty),
        "uniqueness": len(unique) / max(len(non_empty), 1),
        "novelty": (
            len(unique - training_sequences) / max(len(unique), 1)
            if training_sequences
            else float("nan")
        ),
        "mean_length": float(np.mean([len(s) for s in non_empty])) if non_empty else 0.0,
    }
    if is_signal is not None and non_empty:
        report["signal_precision"] = float(
            np.mean([bool(is_signal(s)) for s in non_empty])
        )
    return report
