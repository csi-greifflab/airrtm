"""Turn a trained AIRRTM model into per-sequence scores and per-repertoire features."""

import numpy as np
import torch

from tqdm import tqdm

from airrtm.models import AIRRTM_Model
from airrtm.utils import SequenceDataset


EPS = 1e-12


__all__ = [
    "DEFAULT_QUANTILES",
    "score_sequences",
    "score_repertoires",
    "repertoire_features",
    "infer_topic_proportions",
    "fold_in_topic_proportions",
    "topic_proportion_features",
]

#: Upper tail of the signal-intensity distribution. A repertoire is called positive
#: because a small fraction of its sequences carry the signal, so the discriminative
#: statistic lives in the extreme upper quantiles -- the quantile at which the classes
#: separate is itself an estimate of the witness rate.
DEFAULT_QUANTILES = (
    0.5,
    0.9,
    0.99,
    0.995,
    0.998,
    0.999,
    0.9995,
    0.9999,
    0.99995,
    0.99999,
)


@torch.no_grad()
def score_sequences(
    model: AIRRTM_Model,
    dataset: SequenceDataset,
    batch_size: int = 8192,
    device: torch.device | None = None,
    method: str = "label_head",
    repertoire_labels: torch.Tensor | None = None,
    repertoire_topic_proportions_RT: torch.Tensor | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Signal intensity for every sequence of one repertoire.

    ``method`` is passed through to :meth:`AIRRTM_Model.predict_signal_intensity`.
    """
    device = device or next(model.parameters()).device
    model.eval()
    scores = []
    n = dataset.size()
    steps = range(0, n, batch_size)
    for start in tqdm(steps, disable=not show_progress, leave=False):
        stop = min(start + batch_size, n)
        x_SP = dataset.data[start:stop].to(torch.long).to(device)
        v_ids = dataset.v_ids[start:stop].to(device) if dataset.v_ids is not None else None
        j_ids = dataset.j_ids[start:stop].to(device) if dataset.j_ids is not None else None
        scores.append(
            model.predict_signal_intensity(
                x_SP,
                repertoire_labels_R=repertoire_labels,
                v_ids=v_ids,
                j_ids=j_ids,
                repertoire_topic_proportions_RT=repertoire_topic_proportions_RT,
                method=method,
            ).detach().to("cpu")
        )
    return torch.concatenate(scores, dim=0)


def score_repertoires(
    model: AIRRTM_Model,
    datasets: list[SequenceDataset],
    show_progress: bool = True,
    **kwargs,
) -> list[torch.Tensor]:
    return [
        score_sequences(model, dataset, **kwargs)
        for dataset in tqdm(datasets, disable=not show_progress, desc="scoring")
    ]


def repertoire_features(
    scores_by_repertoire: list[torch.Tensor],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> np.ndarray:
    """Represent each repertoire by upper quantiles of its signal intensities.

    This is the representation the paper uses for repertoire classification (Fig. 4A):
    the mean is dominated by the signal-negative majority, so the discriminative
    information is in the tail.
    """
    quantile_tensor = torch.tensor(quantiles, dtype=torch.float32)
    rows = []
    for scores in scores_by_repertoire:
        scores = scores.to(torch.float32).flatten()
        rows.append(
            torch.quantile(scores, q=quantile_tensor).numpy().tolist()
            + [float(scores.mean()), float(scores.std())]
        )
    return np.asarray(rows, dtype=np.float64)


@torch.no_grad()
def infer_topic_proportions(
    model: AIRRTM_Model,
    dataset: SequenceDataset,
    n_sequences: int = 16384,
    n_repeats: int = 4,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Topic proportions of a repertoire the model may never have seen.

    Averages the amortized head over ``n_repeats`` independent samples of the
    repertoire to damp the sampling noise. Requires ``theta_mode`` with an amortized
    component -- a model trained with ``theta_mode="free"`` has no way to place an
    unseen repertoire in topic space, which is why the earlier evaluation could only
    ever look at training repertoires.
    """
    device = device or next(model.parameters()).device
    model.eval()
    probabilities = dataset.sampling_probabilities()
    size = dataset.size()
    accumulated = None
    for _ in range(n_repeats):
        if probabilities is None:
            indices = torch.randint(
                low=0, high=size, size=(min(n_sequences, size),), generator=generator
            )
        else:
            indices = torch.multinomial(
                probabilities,
                num_samples=min(n_sequences, size),
                replacement=True,
                generator=generator,
            )
        theta = model.infer_repertoire_topic_proportions(
            dataset.data[indices].to(torch.long).to(device),
            v_ids=dataset.v_ids[indices].to(device) if dataset.v_ids is not None else None,
            j_ids=dataset.j_ids[indices].to(device) if dataset.j_ids is not None else None,
            weights=dataset.weights[indices].to(device) if dataset.weights is not None else None,
        ).detach().to("cpu")
        accumulated = theta if accumulated is None else accumulated + theta
    return accumulated / n_repeats


@torch.no_grad()
def fold_in_topic_proportions(
    model: AIRRTM_Model,
    dataset: SequenceDataset,
    n_sequences: int = 16384,
    n_repeats: int = 4,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    use_weights: bool = False,
) -> torch.Tensor:
    """Theta for an unseen repertoire, by one EM E-step -- the LDA "fold-in".

    ``infer_topic_proportions`` needs an amortized head, so a ``theta_mode="free"``
    model -- v1's parameterisation, and every unopposed-TM diagnostic -- has no way
    to place a held-out repertoire in topic space, and those runs have had to fall
    back on ``repertoire_features`` (12 quantiles of a scalar per-sequence score).
    That fallback is weak twice over: it summarises a *scalar projection* of phi
    rather than the latent the model actually defines, and its quantiles are rank
    statistics over the whole repertoire, so with Emerson depths spanning
    50k-590k clonotypes the 0.99999 quantile is the top 0.5 sequences in the
    smallest repertoire and the top 5.9 in the largest -- the feature itself carries
    sequencing depth.

    Fold-in is the standard answer: freeze phi (which is amortized and applies to any
    sequence) and infer only Theta for the new repertoire. Under a uniform prior over
    topics the posterior of a single sequence is

        p(t | s) = softmax_t(phi_ts)

    because ``exp(phi_ts)`` is the model's unnormalised ``p(s | t)``. One E-step then
    aggregates those posteriors into a repertoire-level mixture::

        theta_t  ∝  sum_s  w_s * softmax_t(phi_ts)

    which is the first iteration of PLSA/LDA's EM from a uniform start. Iterating
    would refine it, but the single step is free, has no learning rate, and already
    returns the model's own latent in the model's own units -- so a free-Theta run
    becomes directly comparable to an amortized one on ``theta_features``, against
    the same untrained-network baseline.

    ``use_weights`` defaults to False on purpose. When the dataset carries clonal
    abundances the sampling here is already multinomial in ``duplicate_count``, so
    the draw is abundance-weighted; applying the weights again in the aggregation
    would square them, which is the exact defect ``theta_pooling_weights`` exists to
    control on the amortized path.
    """
    device = device or next(model.parameters()).device
    model.eval()
    probabilities = dataset.sampling_probabilities()
    size = dataset.size()
    n_draw = min(n_sequences, size)
    accumulated = None
    for _ in range(n_repeats):
        if probabilities is None:
            indices = torch.randint(
                low=0, high=size, size=(n_draw,), generator=generator
            )
        else:
            indices = torch.multinomial(
                probabilities, num_samples=n_draw, replacement=True, generator=generator
            )
        phi_ST = model.predict_topic_logits(
            dataset.data[indices].to(torch.long).to(device),
            v_ids=dataset.v_ids[indices].to(device) if dataset.v_ids is not None else None,
            j_ids=dataset.j_ids[indices].to(device) if dataset.j_ids is not None else None,
        )
        posterior_ST = torch.softmax(phi_ST, dim=1)
        if use_weights and dataset.weights is not None:
            weights_S = dataset.weights[indices].to(device).to(posterior_ST.dtype)
            theta_T = (posterior_ST * weights_S.unsqueeze(1)).sum(dim=0)
        else:
            theta_T = posterior_ST.sum(dim=0)
        theta_T = theta_T / theta_T.sum().clamp(min=EPS)
        theta_T = theta_T.detach().to("cpu")
        accumulated = theta_T if accumulated is None else accumulated + theta_T
    return accumulated / n_repeats


def topic_proportion_features(
    model: AIRRTM_Model,
    datasets: list[SequenceDataset],
    show_progress: bool = True,
    theta_method: str = "auto",
    **kwargs,
) -> np.ndarray:
    """Represent each repertoire directly by its inferred topic proportions.

    ``theta_method="amortized"`` runs the trained pooling head, ``"fold_in"`` runs the
    one-step EM in ``fold_in_topic_proportions``, and ``"auto"`` (the default) picks
    the former when the model has an amortized head and the latter when it does not --
    so a ``theta_mode="free"`` model now produces theta_features instead of being
    skipped.
    """
    if theta_method not in ("auto", "amortized", "fold_in"):
        raise ValueError(
            f"theta_method must be 'auto', 'amortized' or 'fold_in', got {theta_method!r}"
        )
    if theta_method == "auto":
        theta_method = "amortized" if model.theta_mode != "free" else "fold_in"
    infer = infer_topic_proportions if theta_method == "amortized" else fold_in_topic_proportions
    return np.stack(
        [
            infer(model, dataset, **kwargs).numpy()
            for dataset in tqdm(datasets, disable=not show_progress, desc="theta")
        ]
    )
