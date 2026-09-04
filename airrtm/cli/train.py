import pathlib as pl

from argparse import ArgumentParser

import pandas as pd
import torch
import yaml

import airrtm.utils as au

from airrtm.cli.preprocess_sequences import METADATA_FILENAME
from airrtm.losses import CompositeLoss
from airrtm.models import model_factory
from airrtm.training.train import train_model


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=pl.Path)
    parser.add_argument("--output_dir", required=True, type=pl.Path)
    parser.add_argument("--config", required=True, type=pl.Path)
    args = parser.parse_args()

    with open(args.config) as inp:
        config = yaml.safe_load(inp)
    model_config = config["model_config"]
    training_config = config["training_config"]

    device = torch.device(training_config["device"])
    seed = training_config["seed"]
    torch.manual_seed(seed)

    input_dir = pl.Path(args.input_dir)
    metadata_df = pd.read_csv(input_dir / METADATA_FILENAME)
    metadata_df = metadata_df.loc[metadata_df["split"] == "train"]

    output_dir = pl.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    log_dir = output_dir / training_config["log_dir_suffix"]
    checkpoint_dir = output_dir / "checkpoints"

    labels = torch.Tensor(metadata_df["label"].to_numpy()).to(torch.uint8)
    sequence_datasets = [
        au.SequenceDataset.load(input_dir / f) for f in metadata_df["filename"]
    ]
    shuffled_indices = [torch.randperm(d.size()) for d in sequence_datasets]
    indices_train = [
        indices[int(indices.shape[0] * training_config["val_size"]) :]
        for indices in shuffled_indices
    ]
    indices_val = [
        indices[: int(indices.shape[0] * training_config["val_size"])]
        for indices in shuffled_indices
    ]
    sequence_datasets_train = [
        d[indices] for d, indices in zip(sequence_datasets, indices_train)
    ]
    sequence_datasets_val = [
        d[indices] for d, indices in zip(sequence_datasets, indices_val)
    ]

    # update config with the information from the training set
    model_config["airrtm_params"]["n_repertoires"] = len(sequence_datasets)
    max_sequence_length = sequence_datasets[0].max_length
    alphabet_length = sequence_datasets[0].alphabet_length
    model_config["encoder_params"]["max_sequence_length"] = max_sequence_length
    model_config["encoder_params"]["alphabet_length"] = alphabet_length
    model_config["decoder_params"]["max_sequence_length"] = max_sequence_length
    model_config["decoder_params"]["alphabet_length"] = alphabet_length
    model = model_factory(**model_config)

    criterion = CompositeLoss(**training_config["loss_config"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    train_model(
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
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        keep_best_model=True,
    )
    torch.save(model, output_dir / "model.pt")
    with open(output_dir / "config.yaml", "w") as otp:
        yaml.dump(config, otp)
