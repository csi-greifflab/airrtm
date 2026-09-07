"""Inspect what the learned topics actually contain.

The point of using a topic model rather than a plain MIL classifier is that the
intermediate representation is meant to be readable. These helpers make that
checkable: which sequences load on a topic, whether it has a V-gene or length bias,
and whether positive and negative repertoires use it differently.
"""

import numpy as np
import pandas as pd
import torch

from airrtm.models import AIRRTM_Model
from airrtm.utils import SequenceDataset


__all__ = [
    "top_sequences_per_topic",
    "topic_composition",
    "topic_separation",
    "position_weight_matrix",
]


@torch.no_grad()
def top_sequences_per_topic(
    model: AIRRTM_Model,
    dataset: SequenceDataset,
    n_top: int = 50,
    topics: tuple[int, ...] | None = None,
    batch_size: int = 8192,
    device: torch.device | None = None,
) -> dict[int, pd.DataFrame]:
    """The ``n_top`` sequences with the highest probability under each topic."""
    device = device or next(model.parameters()).device
    model.eval()
    topics = tuple(range(model.n_topics)) if topics is None else topics

    probabilities = []
    n = dataset.size()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        probabilities.append(
            model.predict_topic_probabilities(
                dataset.data[start:stop].to(torch.long).to(device),
                v_ids=dataset.v_ids[start:stop].to(device) if dataset.v_ids is not None else None,
                j_ids=dataset.j_ids[start:stop].to(device) if dataset.j_ids is not None else None,
            ).detach().to("cpu")
        )
    probabilities_ST = torch.concatenate(probabilities, dim=0)

    strings = dataset.to_strings()
    result = {}
    for topic in topics:
        scores = probabilities_ST[:, topic]
        order = torch.topk(scores, k=min(n_top, scores.shape[0])).indices
        result[topic] = pd.DataFrame(
            {
                "cdr3_aa": [strings[i] for i in order.tolist()],
                "topic_probability": scores[order].numpy(),
                "length": [len(strings[i]) for i in order.tolist()],
                "weight": (
                    dataset.weights[order].cpu().numpy()
                    if dataset.weights is not None
                    else np.ones(len(order))
                ),
                "v_gene": (
                    [dataset.v_genes[i] for i in dataset.v_ids[order].tolist()]
                    if dataset.has_vj and dataset.v_genes is not None
                    else None
                ),
            }
        )
    return result


def topic_composition(top_sequences: pd.DataFrame) -> dict[str, object]:
    """Summarise one topic's top sequences: length profile and V-gene usage."""
    composition = {
        "n": int(len(top_sequences)),
        "mean_length": float(top_sequences["length"].mean()),
        "length_std": float(top_sequences["length"].std()),
    }
    if top_sequences["v_gene"].notna().any():
        composition["top_v_genes"] = (
            top_sequences["v_gene"].value_counts(normalize=True).head(5).to_dict()
        )
    return composition


def topic_separation(
    topic_proportions_RT: np.ndarray,
    repertoire_labels_R: np.ndarray,
) -> pd.DataFrame:
    """Per-topic difference in usage between positive and negative repertoires.

    This is the quantity ``AIRRTM_Model._compute_signal_topic_weights`` turns into
    signal-intensity weights, exposed here so it can be looked at directly. A model
    where the TM branch did nothing shows near-zero differences across all topics.
    """
    labels = np.asarray(repertoire_labels_R).astype(bool)
    positive = topic_proportions_RT[labels]
    negative = topic_proportions_RT[~labels]
    difference = positive.mean(axis=0) - negative.mean(axis=0)
    pooled_std = np.sqrt(
        (positive.var(axis=0, ddof=1) + negative.var(axis=0, ddof=1)) / 2
    )
    return pd.DataFrame(
        {
            "topic": np.arange(topic_proportions_RT.shape[1]),
            "mean_positive": positive.mean(axis=0),
            "mean_negative": negative.mean(axis=0),
            "difference": difference,
            "cohens_d": difference / np.maximum(pooled_std, 1e-12),
        }
    ).sort_values("cohens_d", key=np.abs, ascending=False)


def position_weight_matrix(
    sequences: list[str], alphabet: tuple[str, ...], max_length: int
) -> pd.DataFrame:
    """Positional amino-acid frequencies of a set of sequences, for motif inspection."""
    counts = np.zeros((max_length, len(alphabet)))
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    for sequence in sequences:
        for position, symbol in enumerate(sequence[:max_length]):
            if symbol in index:
                counts[position, index[symbol]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        counts / np.maximum(totals, 1), columns=list(alphabet)
    ).rename_axis("position")
