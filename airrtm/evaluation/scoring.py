"""Turn a trained AIRRTM model into per-sequence scores and per-repertoire features."""

import numpy as np
import torch

from tqdm import tqdm

from airrtm.models import AIRRTM_Model
from airrtm.utils import SequenceDataset


__all__ = [
    "DEFAULT_QUANTILES",
    "score_sequences",
    "score_repertoires",
    "repertoire_features",
    "infer_topic_proportions",
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


def topic_proportion_features(
    model: AIRRTM_Model,
    datasets: list[SequenceDataset],
    show_progress: bool = True,
    **kwargs,
) -> np.ndarray:
    """Represent each repertoire directly by its inferred topic proportions."""
    return np.stack(
        [
            infer_topic_proportions(model, dataset, **kwargs).numpy()
            for dataset in tqdm(datasets, disable=not show_progress, desc="theta")
        ]
    )
