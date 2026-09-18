"""Per-sequence PR-AUC of a checkpoint against the Fisher-selected clonotypes.

PLAN_v5 stage 4, 2026-09-18. Repertoire classification is what every number in
FINDINGS.md reports, but the payoff the architecture was built for is *sequence
attribution*: ranking the CDR3s that actually drive the label. Every checkpoint
tested so far sits at chance on that task (`pr_auc_over_chance` 0.94-1.11x,
precision@100 exactly 0), and that was measured with a scratch script that was
never committed -- this is the committed version.

Ground truth is the Fisher-selected clonotype set: `BurdenScoreClassifier` fit on
the training split only, so the labels never touch the test repertoires. Scoring
is restricted to the positive test repertoires that hold at least one selected
clonotype, because a negative repertoire has no signal sequence to rank.

Read `pr_auc_over_chance` first. At a witness rate of ~1e-4, ROC-AUC barely moves
even for a ranking that is clearly good or clearly bad -- see the docstring of
`signal_enrichment`.

Usage:
    python analysis/sequence_attribution.py \
        --run_dir /Warehouse/Andrei/emerson/model/normloss_tm03_nowarm_seed239 \
        --config emerson_run/config_normloss_tm03_nowarm_seed239.yaml \
        --input_dir /Warehouse/Andrei/emerson/processed_data_vj \
        --epoch 40 --device cuda:7
"""

import argparse
import json
import pathlib as pl

import numpy as np
import torch

import airrtm.utils as au
from airrtm.evaluation.baseline import BurdenScoreClassifier
from airrtm.evaluation.metrics import signal_enrichment
from airrtm.evaluation.scoring import topic_proportion_features
from airrtm.models.airrtm_model import EPS, _grouped_softmax

from eval_checkpoints import build_model, checkpoint_epochs



def _per_row_keys(dataset) -> np.ndarray:
    """clonotype_keys() without its final np.unique -- one key per row."""
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


@torch.no_grad()
def score_dataset(
    model, dataset, method, device, batch_size=8192, labels=None,
    theta_train_RT=None,
):
    """Per-sequence signal intensity for every sequence in one repertoire."""
    out = []
    for start in range(0, dataset.size(), batch_size):
        stop = min(start + batch_size, dataset.size())
        kwargs = {}
        if dataset.v_ids is not None:
            kwargs["v_ids"] = dataset.v_ids[start:stop].to(device)
            kwargs["j_ids"] = dataset.j_ids[start:stop].to(device)
        if method == "topic_weights":
            kwargs["repertoire_labels_R"] = labels
            if theta_train_RT is not None:
                kwargs["repertoire_topic_proportions_RT"] = theta_train_RT
        out.append(
            model.predict_signal_intensity(
                dataset.data[start:stop].to(torch.long).to(device),
                method=method,
                **kwargs,
            ).flatten().cpu()
        )
    return torch.cat(out).numpy()



@torch.no_grad()
def score_attention_value(model, dataset, device, batch_size=8192) -> np.ndarray:
    """The ``attention * value`` pathway of TopicAttentionPooling, summed over signal topics.

    This is the readout that has ever shown real sequence-ranking signal:
    `warmstart_full_medium` scored pr_auc_over_chance 16.9x and enrichment
    166x/66x/43x at the finest fractions, reproducing on 2 of 3 seeds
    (FINDINGS.md, 2026-09-10). The `label_head` and `topic_weights` readouts both
    go through phi, which has never beaten ~1.1x on any checkpoint -- so comparing
    a new checkpoint on those alone says nothing about this pathway.

    The attention weights are a softmax over the sequences WITHIN a repertoire, so
    the whole repertoire has to be normalised together; batching is only over the
    encoder forward, and the softmax is applied once at the end.
    """
    if getattr(model, "theta_pooling", None) != "attention":
        raise ValueError(
            f"attention*value needs theta_pooling='attention', got "
            f"{getattr(model, 'theta_pooling', None)!r}"
        )
    pooling = model.theta_attention
    scores, values = [], []
    for start in range(0, dataset.size(), batch_size):
        stop = min(start + batch_size, dataset.size())
        z_mean_SL = model.sequence_to_latent(
            dataset.data[start:stop].to(torch.long).to(device)
        )
        h_SD = model._with_vj(
            z_mean_SL,
            dataset.v_ids[start:stop].to(device) if dataset.v_ids is not None else None,
            dataset.j_ids[start:stop].to(device) if dataset.j_ids is not None else None,
        )
        gated_SH = torch.tanh(pooling.attend(h_SD)) * torch.sigmoid(pooling.gate(h_SD))
        scores.append(pooling.score(gated_SH).cpu())
        values.append(pooling.value(h_SD).cpu())
    score_ST = torch.cat(scores)
    values_ST = torch.cat(values)

    # Clonal abundance enters the attention logits exactly as it does in training.
    log_weights_S = None
    if dataset.weights is not None and getattr(model, "theta_pooling_weights", True):
        weights_S = dataset.weights.cpu().to(torch.float32).clamp(min=EPS)
        power = getattr(model, "theta_pooling_weight_power", 1.0)
        if power != 1.0:
            weights_S = weights_S ** power
        log_weights_S = torch.log(weights_S.clamp(min=EPS))

    inverse_S = torch.zeros(score_ST.shape[0], dtype=torch.long)
    attention_ST = _grouped_softmax(score_ST, inverse_S, 1, log_weights_S)
    signal = model.n_topics_signal
    return (attention_ST[:, :signal] * values_ST[:, :signal]).sum(dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=pl.Path)
    parser.add_argument("--config", required=True, type=pl.Path)
    parser.add_argument("--input_dir", required=True, type=pl.Path)
    parser.add_argument("--epoch", type=int, default=None,
                        help="periodic checkpoint to score")
    parser.add_argument("--model_pt", type=pl.Path, default=None,
                        help="score a final model.pt (the restored best epoch) instead. "
                        "The 16.9x reference number was measured on one of these, not on a "
                        "periodic checkpoint, and this metric is very epoch-sensitive.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--p_threshold", type=float, default=1e-3)
    parser.add_argument("--output", type=pl.Path, default=None)
    args = parser.parse_args()

    import yaml
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(args.config.read_text())

    meta_train = au.load_metadata(args.input_dir, split="train")
    meta_test = au.load_metadata(args.input_dir, split="test")
    datasets_train, labels_train = au.load_repertoires(args.input_dir, meta_train)
    datasets_test, labels_test = au.load_repertoires(args.input_dir, meta_test)

    classifier = BurdenScoreClassifier(p_threshold=args.p_threshold, min_repertoires=4)
    classifier.fit(datasets_train, labels_train.numpy())
    selected = classifier.selected_keys_
    print(f"Fisher selected {len(selected)} clonotypes at p<{args.p_threshold}")

    model = build_model(config, len(datasets_train), datasets_train[0], device)
    if args.model_pt is not None:
        payload = torch.load(args.model_pt, map_location=device, weights_only=False)
        state = payload["state_dict"] if "state_dict" in payload else payload
    else:
        matches = [p for e, p in checkpoint_epochs(args.run_dir) if e == args.epoch]
        if not matches:
            raise SystemExit(
                f"no checkpoint at epoch {args.epoch}; have "
                f"{[e for e, _ in checkpoint_epochs(args.run_dir)]}"
            )
        state = torch.load(matches[0], map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state)
    model.eval()
    datasets_train_kept = datasets_train

    theta_train_RT = None
    if model.theta_mode != "free":
        theta_train_RT = torch.tensor(
            topic_proportion_features(model, datasets_train_kept), device=device
        )
    report = {"epoch": args.epoch if args.model_pt is None else "model.pt", "n_selected": int(len(selected))}
    for method in ("label_head", "topic_weights", "attention_value"):
        all_scores, all_flags = [], []
        for dataset, label in zip(datasets_test, labels_test.tolist()):
            if label != 1:
                continue  # a negative repertoire has no signal sequence to rank
            flags = np.isin(_per_row_keys(dataset), selected)
            if not flags.any():
                continue
            try:
                if method == "attention_value":
                    scores = score_attention_value(model, dataset, device)
                else:
                    scores = score_dataset(
                        model, dataset, method, device,
                        labels=labels_train.to(device), theta_train_RT=theta_train_RT,
                    )
            except (ValueError, TypeError) as exc:
                report[method] = f"unavailable: {exc}"
                break
            all_scores.append(scores)
            all_flags.append(flags)
        else:
            if all_scores:
                s = np.concatenate(all_scores)
                f = np.concatenate(all_flags)
                report[method] = signal_enrichment(s, f)
                report[method]["n_repertoires"] = len(all_scores)
                print(f"\n{method}:")
                for k, v in report[method].items():
                    print(f"    {k:<22} {v}")
    output = args.output or (args.run_dir / f"sequence_attribution_epoch{args.epoch}.json")
    output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
