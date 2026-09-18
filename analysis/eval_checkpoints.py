"""Score `theta_features` ROC-AUC across a run's periodic checkpoints.

PLAN_v4, 2026-09-16. Training stops on `val_metrics["total_loss"]`, which on
Emerson is dominated by reconstruction, and the restored "best epoch" therefore
has no particular relationship to the metric every result in FINDINGS.md is
reported on. Observed best epochs across one batch of runs span **5 to 57** on
the same 60-epoch budget, which is enough on its own to invalidate a comparison
between arms.

This script answers both questions the spread raises, without retraining:

1. **Is Theta undertrained?** -- is held-out AUC still rising at the last
   checkpoint, or has it plateaued?
2. **Does early stopping pick the right checkpoint?** -- how far is the
   total_loss-selected epoch from the AUC-best one?

It rebuilds the model from the run's own config yaml (periodic checkpoints carry
only `{"epoch", "state_dict"}`, not `model_config`) and reuses the exact
functions `evaluate-model` uses internally, so the numbers are comparable to
`report.json`.

Usage:
    python analysis/eval_checkpoints.py \
        --run_dir /Warehouse/Andrei/emerson/model/warmstart_full_medium \
        --config emerson_run/config_warmstart_full_medium.yaml \
        --input_dir /Warehouse/Andrei/emerson/processed_data_vj \
        --device cuda:0
"""

import argparse
import json
import pathlib as pl
import re

import numpy as np
import torch
import yaml

import airrtm.utils as au
from airrtm.evaluation.metrics import classify_repertoires
from airrtm.evaluation.scoring import (
    repertoire_features,
    score_repertoires,
    topic_proportion_features,
)
from airrtm.models import model_factory


def checkpoint_epochs(run_dir: pl.Path) -> list[tuple[int, pl.Path]]:
    found = []
    for path in (run_dir / "checkpoints").glob("*.pt"):
        match = re.search(r"(\d+)", path.stem)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def build_model(config: dict, n_repertoires: int, reference, device) -> torch.nn.Module:
    """Rebuild the trained architecture the way cli/train.py does."""
    model_config = json.loads(json.dumps(config["model_config"]))
    data_config = config.get("data_config", {})
    for key in ("encoder_params", "decoder_params"):
        model_config[key]["max_sequence_length"] = reference.max_length
        model_config[key]["alphabet_length"] = reference.alphabet_length
    model_config["airrtm_params"]["n_repertoires"] = n_repertoires
    use_vj = data_config.get("use_vj", True) and reference.has_vj
    model_config["airrtm_params"]["n_v_genes"] = reference.n_v_genes if use_vj else 0
    model_config["airrtm_params"]["n_j_genes"] = reference.n_j_genes if use_vj else 0
    return model_factory(**model_config).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=pl.Path)
    parser.add_argument("--config", required=True, type=pl.Path)
    parser.add_argument("--input_dir", required=True, type=pl.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--features",
        default="theta",
        choices=("theta", "quantile"),
        help="theta (default) reports theta_features, folding Theta in for a "
        "free-Theta model; quantile reports the older per-sequence summary.",
    )
    parser.add_argument(
        "--score_method",
        default="label_head",
        choices=("label_head", "topic_weights"),
        help="Per-sequence readout for theta_mode=free runs. The default reads the "
        "trained label head -- but a tm_likelihood_coef=1.0 diagnostic never trains "
        "that head (label_likelihood_coef = 1 - tm = 0), so label_head is a random "
        "projection there and understates phi. topic_weights is the paper's own "
        "formulation and is the fair readout for those runs.",
    )
    parser.add_argument(
        "--random_init_repeats",
        type=int,
        default=1,
        help="How many independently initialised untrained models to score. One "
        "draw is not a distribution, and initialisation spread is exactly what "
        "this control has to measure.",
    )
    parser.add_argument("--output", type=pl.Path, default=None)
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Also score the freshly built, UNTRAINED model (reported as epoch -1). "
        "The control for whether theta_features AUC reflects learning at all: the "
        "attention-pooled Theta is a 30-dim projection of a repertoire's sequence "
        "composition, and a random projection of that may already separate the "
        "classes.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(args.config.read_text())

    metadata_train = au.load_metadata(args.input_dir, split="train")
    metadata_test = au.load_metadata(args.input_dir, split="test")
    datasets_train, labels_train = au.load_repertoires(args.input_dir, metadata_train)
    datasets_test, labels_test = au.load_repertoires(args.input_dir, metadata_test)
    y_train, y_test = labels_train.numpy(), labels_test.numpy()

    model = build_model(config, len(datasets_train), datasets_train[0], device)

    rows = []
    todo = checkpoint_epochs(args.run_dir)
    if args.random_init:
        todo = [(-1 - i, None) for i in range(args.random_init_repeats)] + todo
    for epoch, path in todo:
        if path is None:
            # Re-draw the weights so each random-init row is an independent sample.
            model = build_model(config, len(datasets_train), datasets_train[0], device)
        else:
            payload = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(payload["state_dict"])
        model.eval()
        if args.features == "quantile":
            score_kwargs = dict(
                batch_size=8192, device=device, method=args.score_method
            )
            if args.score_method == "topic_weights":
                score_kwargs["repertoire_labels"] = labels_train.to(device)
            train_features = repertoire_features(
                score_repertoires(model, datasets_train, **score_kwargs)
            )
            test_features = repertoire_features(
                score_repertoires(model, datasets_test, **score_kwargs)
            )
        else:
            # "auto" folds in a free Theta and runs the amortized head otherwise.
            train_features = topic_proportion_features(model, datasets_train)
            test_features = topic_proportion_features(model, datasets_test)
        result = classify_repertoires(train_features, y_train, test_features, y_test)
        rows.append({"epoch": epoch, **{k: v for k, v in result.items()}})
        metric = "quantile_features" if args.features == "quantile" else "theta_features"
        print(
            f"  epoch {epoch:4d}  {metric} roc_auc={result['roc_auc']:.4f}  "
            f"train={result['train_roc_auc']:.4f}",
            flush=True,
        )

    if rows:
        best = max(rows, key=lambda r: r["roc_auc"])
        print(f"\nbest checkpoint: epoch {best['epoch']}  roc_auc={best['roc_auc']:.4f}")
        print(f"last checkpoint: epoch {rows[-1]['epoch']}  roc_auc={rows[-1]['roc_auc']:.4f}")
    output = args.output or (args.run_dir / "checkpoint_auc_curve.json")
    output.write_text(json.dumps(rows, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
