"""Experiment 2(a): how much of the Fisher signal survives bag subsampling?

PLAN_v4, 2026-09-16. Every AIRRTM run reads a repertoire through a bag of
``n_sequences_per_repertoire_in_batch`` sequences (8192 in every Emerson config,
4 x 16384 at evaluation time), while the Fisher burden baseline reads *all* of a
repertoire's distinct clonotypes. This script measures the resulting ceiling
directly: it gives an oracle -- the Fisher-selected clonotype set itself -- the
same bag the model gets, and reports the ROC-AUC that oracle achieves.

If the oracle scores ~0.70 from an 8192 bag, then 0.740 is already at the
*sampling* ceiling and no modelling change that reads an 8192 bag can beat it;
the lever is bag size or vocabulary restriction, not architecture.

It also reports the effective sample size of the three pooling-weight schemes
(uniform / count / count**2) behind PLAN_v4 diagnosis section 3 -- count**2 is
what the shipped code applies, because abundance enters both the sampler and
Theta's attention pooling.

Usage:
    python analysis/ceiling_sampling.py --input_dir /path/to/processed_data_vj
"""

import argparse
import json
import pathlib as pl

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import airrtm.utils as au
from airrtm.evaluation.baseline import BurdenScoreClassifier, clonotype_keys


#: Bag sizes to probe. 8192 is every training config's per-repertoire draw;
#: 65536 is 4 x 16384, what `infer_topic_proportions` uses at evaluation time.
BAG_SIZES = (1024, 4096, 8192, 16384, 32768, 65536, 131072)
N_DRAWS = 20


def effective_sample_size(weights: np.ndarray) -> float:
    """1 / sum(q**2) for the normalised weights -- how many items actually count."""
    total = weights.sum()
    if total <= 0:
        return float(len(weights))
    q = weights / total
    return float(1.0 / np.square(q).sum())


def repertoire_summary(dataset, selected_keys: np.ndarray) -> dict:
    """Per-clonotype keys, abundances and Fisher-hit flags for one repertoire."""
    # clonotype_keys() de-duplicates; we need the per-row mapping to pair each
    # clonotype with its abundance, so redo the hash and aggregate here.
    keys_all = _per_row_keys(dataset)
    keys, inverse = np.unique(keys_all, return_inverse=True)
    if dataset.weights is None:
        counts = np.bincount(inverse).astype(np.float64)
    else:
        counts = np.bincount(
            inverse, weights=dataset.weights.cpu().numpy().astype(np.float64)
        )
    hits = np.isin(keys, selected_keys)
    return {"keys": keys, "counts": counts, "hits": hits}


def _per_row_keys(dataset) -> np.ndarray:
    """clonotype_keys() without the final np.unique -- one key per row."""
    tokens = dataset.data.to(torch.int64).cpu().numpy().astype(np.uint64)
    alphabet_size = np.uint64(dataset.alphabet_length + 1)
    mix_constant = np.uint64(0xFF51AFD7ED558CCD)
    shift = np.uint64(33)
    keys = np.zeros(tokens.shape[0], dtype=np.uint64)
    for position in range(tokens.shape[1]):
        keys = keys * alphabet_size + tokens[:, position]
        keys ^= keys >> shift
        keys *= mix_constant
    if dataset.v_ids is not None:
        prime = np.uint64(1000003)
        keys = keys * prime + dataset.v_ids.cpu().numpy().astype(np.uint64)
        keys = keys * prime + dataset.j_ids.cpu().numpy().astype(np.uint64)
    return keys


def oracle_scores(summary: dict, bag_size: int, scheme: str, rng) -> float:
    """Mean over N_DRAWS of (distinct Fisher hits in bag) / (distinct in bag).

    This is exactly ``BurdenScoreClassifier.score``'s statistic, but computed on
    a bag rather than on the whole repertoire.
    """
    counts, hits = summary["counts"], summary["hits"]
    n_clonotypes = counts.shape[0]
    values = []
    for _ in range(N_DRAWS):
        if scheme == "abundance":
            # What the shipped sampler does: multinomial proportional to
            # duplicate_count, WITH replacement.
            probabilities = counts / counts.sum()
            drawn = rng.choice(n_clonotypes, size=bag_size, replace=True, p=probabilities)
        elif scheme == "uniform":
            # Uniform over distinct clonotypes, without replacement where possible.
            size = min(bag_size, n_clonotypes)
            drawn = rng.choice(n_clonotypes, size=size, replace=False)
        else:
            raise ValueError(scheme)
        distinct = np.unique(drawn)
        values.append(hits[distinct].sum() / max(distinct.shape[0], 1))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=pl.Path)
    parser.add_argument("--output", type=pl.Path, default=pl.Path("ceiling_sampling.json"))
    parser.add_argument("--p_threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=239)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    train_meta = au.load_metadata(args.input_dir, split="train")
    test_meta = au.load_metadata(args.input_dir, split="test")
    print(f"{len(train_meta)} train / {len(test_meta)} test repertoires")

    datasets_train, labels_train = au.load_repertoires(args.input_dir, train_meta)
    classifier = BurdenScoreClassifier(p_threshold=args.p_threshold, min_repertoires=4)
    classifier.fit(datasets_train, labels_train.numpy())
    selected = classifier.selected_keys_
    print(f"Fisher selected {len(selected)} clonotypes at p<{args.p_threshold}")

    summaries_train = [
        repertoire_summary(d, selected) for d in tqdm(datasets_train, desc="train stats")
    ]
    del datasets_train

    datasets_test, labels_test = au.load_repertoires(args.input_dir, test_meta)
    summaries_test = [
        repertoire_summary(d, selected) for d in tqdm(datasets_test, desc="test stats")
    ]
    del datasets_test

    y_test = labels_test.numpy()

    report: dict = {
        "n_selected": int(len(selected)),
        "p_threshold": args.p_threshold,
        "n_draws": N_DRAWS,
        "full_repertoire": {},
        "bags": {},
        "effective_sample_size": {},
    }

    # --- the un-subsampled reference: what Fisher itself scores --------------
    full = np.array(
        [s["hits"].sum() / max(s["keys"].shape[0], 1) for s in summaries_test]
    )
    report["full_repertoire"]["roc_auc"] = float(roc_auc_score(y_test, full))
    report["full_repertoire"]["median_n_distinct"] = float(
        np.median([s["keys"].shape[0] for s in summaries_test])
    )
    report["full_repertoire"]["median_hits"] = float(
        np.median([s["hits"].sum() for s in summaries_test])
    )
    print(f"whole-repertoire oracle AUC: {report['full_repertoire']['roc_auc']:.4f}")

    # --- the ceiling curve ---------------------------------------------------
    for scheme in ("abundance", "uniform"):
        for bag_size in BAG_SIZES:
            scores = np.array(
                [oracle_scores(s, bag_size, scheme, rng) for s in summaries_test]
            )
            auc = float(roc_auc_score(y_test, scores))
            report["bags"].setdefault(scheme, {})[str(bag_size)] = auc
            print(f"  {scheme:10s} bag={bag_size:7d}  oracle AUC={auc:.4f}")

    # --- effective sample size of the three pooling schemes ------------------
    for name, power in (("uniform", 0.0), ("count", 1.0), ("count_squared", 2.0)):
        values = [effective_sample_size(s["counts"] ** power) for s in summaries_test]
        report["effective_sample_size"][name] = {
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        print(f"  ESS {name:14s} median={np.median(values):10.1f}")

    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
