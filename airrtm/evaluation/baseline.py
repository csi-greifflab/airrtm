"""The Emerson et al. 2017 burden-score baseline.

Any claim that AIRRTM has learned something about CMV status has to be measured
against this: a purely statistical selection of label-associated clonotypes,
with no sequence model at all.

Method: for every clonotype, count how many positive and negative *training*
repertoires it appears in, take a one-sided Fisher exact test for enrichment in the
positive ones, keep the clonotypes below a p-value threshold, and score a repertoire
by the fraction of its clonotypes that are in that set.
"""

import numpy as np
import torch

from scipy.stats import hypergeom
from tqdm import tqdm

from airrtm.utils import SequenceDataset


__all__ = ["BurdenScoreClassifier", "clonotype_keys"]


def clonotype_keys(dataset: SequenceDataset, use_vj: bool = True) -> np.ndarray:
    """Stable integer keys for the distinct clonotypes of one repertoire.

    Each sequence is hashed from its token ids, optionally mixed with the V/J
    indices, so set operations over hundreds of repertoires stay cheap.
    """
    tokens = dataset.data.to(torch.int64).cpu().numpy().astype(np.uint64)
    # 26 positions x 22 symbols does not fit in 64 bits, so hash instead of pack.
    # Unsigned arithmetic, because the mixing constants overflow a signed int64.
    alphabet_size = np.uint64(dataset.alphabet_length + 1)
    mix_constant = np.uint64(0xFF51AFD7ED558CCD)
    shift = np.uint64(33)
    keys = np.zeros(tokens.shape[0], dtype=np.uint64)
    for position in range(tokens.shape[1]):
        keys = keys * alphabet_size + tokens[:, position]
        keys ^= keys >> shift
        keys *= mix_constant
    if use_vj and dataset.v_ids is not None:
        prime = np.uint64(1000003)
        keys = keys * prime + dataset.v_ids.cpu().numpy().astype(np.uint64)
        keys = keys * prime + dataset.j_ids.cpu().numpy().astype(np.uint64)
    return np.unique(keys)


class BurdenScoreClassifier:
    """Fisher-exact clonotype selection followed by a burden score.

    Parameters
    ----------
    p_threshold : float
        One-sided Fisher p-value below which a clonotype counts as label-associated.
    min_repertoires : int
        Ignore clonotypes seen in fewer repertoires than this; they cannot reach
        significance and testing them only inflates the multiple-testing burden.
    """

    def __init__(self, p_threshold: float = 1e-4, min_repertoires: int = 4):
        self.p_threshold = p_threshold
        self.min_repertoires = min_repertoires
        self.selected_keys_: np.ndarray | None = None

    def fit(
        self,
        datasets: list[SequenceDataset],
        labels: np.ndarray,
        use_vj: bool = True,
        show_progress: bool = True,
    ) -> "BurdenScoreClassifier":
        labels = np.asarray(labels).astype(bool)
        n_positive, n_negative = int(labels.sum()), int((~labels).sum())

        positive_counts: dict[int, int] = {}
        total_counts: dict[int, int] = {}
        for dataset, is_positive in tqdm(
            list(zip(datasets, labels)), disable=not show_progress, desc="burden fit"
        ):
            for key in clonotype_keys(dataset, use_vj=use_vj).tolist():
                total_counts[key] = total_counts.get(key, 0) + 1
                if is_positive:
                    positive_counts[key] = positive_counts.get(key, 0) + 1

        keys = np.fromiter(total_counts.keys(), dtype=np.uint64, count=len(total_counts))
        totals = np.fromiter(total_counts.values(), dtype=np.int64, count=len(total_counts))
        positives = np.array([positive_counts.get(int(k), 0) for k in keys])

        keep = totals >= self.min_repertoires
        keys, totals, positives = keys[keep], totals[keep], positives[keep]

        # One-sided Fisher exact == hypergeometric survival function.
        p_values = hypergeom.sf(
            positives - 1, n_positive + n_negative, n_positive, totals
        )
        self.selected_keys_ = np.sort(keys[p_values < self.p_threshold])
        self.n_tested_ = int(len(keys))
        return self

    def score(
        self,
        datasets: list[SequenceDataset],
        use_vj: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        if self.selected_keys_ is None:
            raise RuntimeError("Call fit() before score()")
        scores = []
        for dataset in tqdm(datasets, disable=not show_progress, desc="burden score"):
            keys = clonotype_keys(dataset, use_vj=use_vj)
            hits = np.isin(keys, self.selected_keys_, assume_unique=True).sum()
            scores.append(hits / max(len(keys), 1))
        return np.asarray(scores)
