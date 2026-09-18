import json
import pathlib as pl

from argparse import ArgumentParser

import torch
import yaml

import airrtm.utils as au

from airrtm.losses import CompositeLoss
from airrtm.models import model_factory
from airrtm.training.train import train_model


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train an AIRRTM model")
    parser.add_argument("--input_dir", required=True, type=pl.Path)
    parser.add_argument("--output_dir", required=True, type=pl.Path)
    parser.add_argument("--config", required=True, type=pl.Path)
    parser.add_argument(
        "--repertoire_slice",
        type=str,
        default=None,
        help="Optional START:STOP slice of the training repertoires, for quick runs "
        "(e.g. '0:32'). Applied after the train/test split.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    with open(args.config) as inp:
        config = yaml.safe_load(inp)
    model_config = config["model_config"]
    training_config = config["training_config"]
    data_config = config.get("data_config", {})

    device = torch.device(training_config["device"])
    seed = training_config["seed"]
    torch.manual_seed(seed)

    input_dir = pl.Path(args.input_dir)
    metadata_df = au.load_metadata(input_dir, split="train")
    if args.repertoire_slice is not None:
        start, stop = (int(x) for x in args.repertoire_slice.split(":"))
        metadata_df = metadata_df.iloc[start:stop].reset_index(drop=True)

    output_dir = pl.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir = output_dir / training_config["log_dir_suffix"]
    checkpoint_dir = output_dir / "checkpoints"

    sequence_datasets, labels = au.load_repertoires(input_dir, metadata_df)
    generator = torch.Generator().manual_seed(seed)
    sequence_datasets_train, sequence_datasets_val = au.split_sequences(
        sequence_datasets, training_config["val_size"], generator=generator
    )

    model = build_model(model_config, sequence_datasets, data_config)
    criterion = build_criterion(training_config, sequence_datasets[0])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )
    scheduler = None
    if training_config.get("lr_decay_gamma"):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_config["lr_decay_gamma"]
        )

    history = train_model(
        model=model,
        sequence_datasets_train=sequence_datasets_train,
        sequence_datasets_val=sequence_datasets_val,
        repertoire_labels=labels,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=training_config["n_epochs"],
        patience=training_config["patience"],
        n_sequences_per_repertoire_in_batch=training_config[
            "n_sequences_per_repertoire_in_batch"
        ],
        n_repertoires_in_batch=training_config.get("n_repertoires_in_batch", 4),
        min_repertoire_size=training_config.get("min_repertoire_size", 32768),
        n_batches_per_repertoire=training_config.get("n_batches_per_repertoire"),
        n_batches_per_repertoire_val=training_config.get(
            "n_batches_per_repertoire_val"
        ),
        n_theta_sequences_per_repertoire=training_config.get(
            "n_theta_sequences_per_repertoire", 0
        ),
        log_every=training_config.get("log_every", 20),
        tau=training_config.get("tau", 1.0),
        tau_start=training_config.get("tau_start"),
        tau_anneal_epochs=training_config.get("tau_anneal_epochs"),
        tm_likelihood_coef_start=training_config.get("tm_likelihood_coef_start"),
        theta_entropy_coef_start=training_config.get("theta_entropy_coef_start"),
        topic_usage_coef_start=training_config.get("topic_usage_coef_start"),
        coef_anneal_epochs=training_config.get("coef_anneal_epochs"),
        vae_coef_start=training_config.get("vae_coef_start"),
        reconstruction_loss_coef_start=training_config.get(
            "reconstruction_loss_coef_start"
        ),
        vae_anneal_epochs=training_config.get("vae_anneal_epochs"),
        abundance_weighted_sampling=data_config.get(
            "abundance_weighted_sampling", True
        ),
        grad_clip=training_config.get("grad_clip"),
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=training_config.get("checkpoint_every", 1),
        keep_best_model=True,
    )

    save_model(model, model_config, output_dir / "model.pt")
    with open(output_dir / "config.yaml", "w") as otp:
        yaml.dump(config, otp)
    with open(output_dir / "history.json", "w") as otp:
        json.dump(history, otp, indent=2)
    print(f"Wrote {output_dir / 'model.pt'}")


def build_model(model_config: dict, sequence_datasets: list, data_config: dict):
    """Fill in the data-dependent fields of the model config, then build the model."""
    reference = sequence_datasets[0]
    max_sequence_length = au.check_consistent_max_length(sequence_datasets)
    alphabet_length = reference.alphabet_length

    model_config["airrtm_params"]["n_repertoires"] = len(sequence_datasets)
    for key in ("encoder_params", "decoder_params"):
        model_config[key]["max_sequence_length"] = max_sequence_length
        model_config[key]["alphabet_length"] = alphabet_length

    use_vj = data_config.get("use_vj", True) and reference.has_vj
    model_config["airrtm_params"]["n_v_genes"] = reference.n_v_genes if use_vj else 0
    model_config["airrtm_params"]["n_j_genes"] = reference.n_j_genes if use_vj else 0
    if use_vj:
        print(
            f"Using V/J genes: {reference.n_v_genes} V, {reference.n_j_genes} J"
        )
    return model_factory(**model_config)


def build_criterion(training_config: dict, reference_dataset) -> CompositeLoss:
    loss_config = dict(training_config["loss_config"])
    loss_config.setdefault(
        "n_sequences_per_repertoire",
        training_config["n_sequences_per_repertoire_in_batch"],
    )
    # Padding is excluded from the reconstruction loss and from the accuracy it
    # reports; on Emerson roughly half of all positions are padding.
    loss_config.setdefault("pad_value", reference_dataset.pad_value)
    loss_config.setdefault("default_tau", training_config.get("tau", 1.0))
    return CompositeLoss(**loss_config)


def save_model(model, model_config: dict, path: pl.Path) -> None:
    """Save weights plus the config needed to rebuild the module.

    Deliberately not ``torch.save(model)``: a pickled module is tied to the exact
    class definition, so every edit to the model code silently invalidates every
    existing checkpoint.
    """
    torch.save({"state_dict": model.state_dict(), "model_config": model_config}, path)


def load_model(path: pl.Path, device: torch.device | None = None):
    """Rebuild a model saved by :func:`save_model`."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = model_factory(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device or "cpu")
