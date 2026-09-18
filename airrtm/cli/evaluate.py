import json
import pathlib as pl

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

import airrtm.utils as au

from airrtm.cli.train import load_model
from airrtm.evaluation import (
    BurdenScoreClassifier,
    classify_repertoires,
    cross_validated_report,
    repertoire_features,
    score_repertoires,
    topic_proportion_features,
    topic_separation,
    top_sequences_per_topic,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Evaluate a trained AIRRTM model on held-out repertoires"
    )
    parser.add_argument("--input_dir", required=True, type=pl.Path)
    parser.add_argument("--model", required=True, type=pl.Path)
    parser.add_argument("--output_dir", required=True, type=pl.Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--score_method",
        type=str,
        default="label_head",
        choices=("label_head", "topic_weights"),
    )
    parser.add_argument(
        "--repertoire_slice",
        type=str,
        default=None,
        help="Optional START:STOP slice of the training repertoires, matching the "
        "slice the model was trained on.",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip the Emerson burden-score baseline (it is the slow part).",
    )
    parser.add_argument("--n_top_sequences", type=int, default=100)
    return parser


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = pl.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, device=device)
    model.eval()

    metadata_train = au.load_metadata(args.input_dir, split="train")
    if args.repertoire_slice is not None:
        start, stop = (int(x) for x in args.repertoire_slice.split(":"))
        metadata_train = metadata_train.iloc[start:stop].reset_index(drop=True)
    metadata_test = au.load_metadata(args.input_dir, split="test")

    datasets_train, labels_train = au.load_repertoires(args.input_dir, metadata_train)
    datasets_test, labels_test = au.load_repertoires(args.input_dir, metadata_test)
    labels_train_np = labels_train.numpy()
    labels_test_np = labels_test.numpy()

    report: dict[str, object] = {
        "n_train_repertoires": len(datasets_train),
        "n_test_repertoires": len(datasets_test),
        "train_label_mean": float(labels_train_np.mean()),
        "test_label_mean": float(labels_test_np.mean()),
        "score_method": args.score_method,
    }

    # --- per-sequence scores -> repertoire quantile features -----------------
    # Needs phi (the per-sequence topic-logit layer), which use_topic_model=False
    # drops entirely -- skip rather than crash, same style as the theta_mode='free'
    # skip below.
    scores_test = None
    if model.use_topic_model:
        score_kwargs = dict(
            batch_size=args.batch_size, device=device, method=args.score_method
        )
        if args.score_method == "topic_weights":
            score_kwargs["repertoire_labels"] = labels_train.to(device)
            if model.theta_mode != "free":
                score_kwargs["repertoire_topic_proportions_RT"] = torch.tensor(
                    topic_proportion_features(model, datasets_train), device=device
                )

        scores_train = score_repertoires(model, datasets_train, **score_kwargs)
        scores_test = score_repertoires(model, datasets_test, **score_kwargs)

        features_train = repertoire_features(scores_train)
        features_test = repertoire_features(scores_test)
        report["quantile_features"] = classify_repertoires(
            features_train, labels_train_np, features_test, labels_test_np
        )
        report["quantile_features_cv"] = cross_validated_report(
            np.concatenate([features_train, features_test]),
            np.concatenate([labels_train_np, labels_test_np]),
        )
    else:
        report["quantile_features"] = (
            "skipped: use_topic_model=False dropped phi, no per-sequence score exists"
        )
        report["quantile_features_cv"] = report["quantile_features"]

    # --- topic proportions as features ---------------------------------------
    # theta_mode="free" has no amortized head, so held-out repertoires used to be
    # skipped here entirely. topic_proportion_features now folds them in instead
    # (one EM E-step over frozen phi), which puts free-Theta runs on the same
    # feature space, classifier and baseline as the amortized ones.
    if True:
        theta_train = topic_proportion_features(model, datasets_train)
        theta_test = topic_proportion_features(model, datasets_test)
        report["theta_features"] = classify_repertoires(
            theta_train, labels_train_np, theta_test, labels_test_np
        )
        report["theta_features_cv"] = cross_validated_report(
            np.concatenate([theta_train, theta_test]),
            np.concatenate([labels_train_np, labels_test_np]),
        )
        separation = topic_separation(
            np.concatenate([theta_train, theta_test]),
            np.concatenate([labels_train_np, labels_test_np]),
        )
        separation.to_csv(output_dir / "topic_separation.csv", index=False)
        report["top_topic_cohens_d"] = float(separation["cohens_d"].abs().max())
        np.save(output_dir / "theta_train.npy", theta_train)
        np.save(output_dir / "theta_test.npy", theta_test)
    else:
        report["theta_features"] = (
            "skipped: theta_mode='free' has no proportions for unseen repertoires"
        )
        report["theta_features_cv"] = report["theta_features"]

    # --- baseline ------------------------------------------------------------
    if not args.skip_baseline:
        baseline = BurdenScoreClassifier().fit(datasets_train, labels_train_np)
        baseline_train = baseline.score(datasets_train).reshape(-1, 1)
        baseline_test = baseline.score(datasets_test).reshape(-1, 1)
        report["burden_baseline"] = classify_repertoires(
            baseline_train, labels_train_np, baseline_test, labels_test_np
        )
        report["burden_n_selected"] = int(len(baseline.selected_keys_))

    # --- what the topics contain (needs phi) ---------------------------------
    if model.use_topic_model:
        top_by_topic = top_sequences_per_topic(
            model, datasets_train[0], n_top=args.n_top_sequences, device=device
        )
        pd.concat(
            [frame.assign(topic=topic) for topic, frame in top_by_topic.items()]
        ).to_csv(output_dir / "top_sequences_per_topic.csv", index=False)

    # --- top-ranked candidate sequences (needs phi) --------------------------
    if scores_test is not None:
        _write_top_candidates(
            datasets_test, scores_test, metadata_test, output_dir, n_top=1000
        )

    with open(output_dir / "report.json", "w") as otp:
        json.dump(report, otp, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


def _write_top_candidates(datasets, scores, metadata_df, output_dir, n_top: int):
    """The highest-scoring sequences across the held-out repertoires."""
    rows = []
    for dataset, score, (_, meta) in zip(datasets, scores, metadata_df.iterrows()):
        k = min(n_top, score.numel())
        top = torch.topk(score.flatten(), k=k)
        strings = dataset.to_strings()
        rows.append(
            pd.DataFrame(
                {
                    "filename": meta["filename"],
                    "label": meta["label"],
                    "cdr3_aa": [strings[i] for i in top.indices.tolist()],
                    "signal_intensity": top.values.numpy(),
                }
            )
        )
    pd.concat(rows).sort_values("signal_intensity", ascending=False).to_csv(
        output_dir / "top_candidate_sequences.csv", index=False
    )
