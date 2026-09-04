import pathlib as pl

import torch

from torch.utils.tensorboard import SummaryWriter

torch.nn.attention._SDPBackend = torch.nn.attention.SDPBackend  # don't ask

from tqdm import tqdm

from airrtm.losses import CompositeLoss
from airrtm.models import AIRRTM_Model
from airrtm.types import AIRRTM_ModelOutput, AIRRTM_ModelTarget
from airrtm.utils import SequenceDataset

LOSS_KEY_TO_PRINT = {
    "total_loss": "total",
    "reconstruction_accuracy": "rec_acc",
    "kl_divergence": "kl_d",
    "tm_loss": "tm",
    "label_accuracy": "label_acc",
}


def train_model(
    *,
    model: AIRRTM_Model,
    sequence_datasets_train: list[SequenceDataset],
    sequence_datasets_val: list[SequenceDataset],  #  from the same repertoires as train
    repertoire_labels: torch.IntTensor,  # assumed to be 1d and consist of 1s and 0s
    criterion: CompositeLoss,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    patience: int,
    n_sequences_per_repertoire_in_batch: int,
    device: torch.device | None = None,
    log_dir: pl.Path | None = None,
    checkpoint_dir: pl.Path | None = None,
    keep_best_model: bool = True,
) -> None:
    device = device or torch.tensor(0.0).device

    writer = None
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

    n_repertoires = len(sequence_datasets_train)
    total_batch_size = n_repertoires * n_sequences_per_repertoire_in_batch
    repertoire_sizes_train = [d.size() for d in sequence_datasets_train]
    minimum_repertoire_size_train = min(repertoire_sizes_train)
    repertoire_sizes_val = [d.size() for d in sequence_datasets_val]
    minimum_repertoire_size_val = min(repertoire_sizes_val)
    n_batches_train = (
        minimum_repertoire_size_train // n_sequences_per_repertoire_in_batch
    )
    n_batches_val = minimum_repertoire_size_val // n_sequences_per_repertoire_in_batch

    print(
        f"min rep size train: {minimum_repertoire_size_train}, min rep size val: {minimum_repertoire_size_val}"
    )
    print(
        f"max rep size train: {max(repertoire_sizes_train)}, max rep size val: {max(repertoire_sizes_val)}"
    )
    print(f"{n_batches_train=}, {n_batches_val=}")
    print(f"{total_batch_size=}")
    assert n_batches_train >= 1, n_batches_val >= 1

    model = model.to(device)
    early_stopping_counter = 0
    best_val_loss = torch.inf
    best_state_dict = model.state_dict().copy()
    best_epoch = 0

    loss_keys = ["total_loss"]
    if criterion.return_individual_compotents:
        loss_keys.extend(
            ["reconstruction_accuracy", "kl_divergence", "tm_loss", "label_accuracy"]
        )
    train_loss_list = []
    val_loss_list = []

    sequence_repertoire_labels = torch.concatenate(
        [
            torch.full(fill_value=label, size=(n_sequences_per_repertoire_in_batch,))
            for label in repertoire_labels
        ]
    ).to(device)
    sequence_repertoire_indicators = torch.ones_like(sequence_repertoire_labels).to(
        device
    )

    self_repertoire_ids = torch.concatenate(
        [
            torch.full(fill_value=i, size=(n_sequences_per_repertoire_in_batch,))
            for i in range(n_repertoires)
        ]
    ).to(device)
    for epoch in tqdm(range(n_epochs)):
        shuffled_sequence_indices = [
            torch.randperm(size) for size in repertoire_sizes_train
        ]
        other_repertoire_ids = _sample_other_repertoire_ids(
            repertoire_labels, n_sequences_per_repertoire_in_batch
        ).to(device)

        model.train()
        running_loss = {key: 0.0 for key in loss_keys}

        with tqdm(total=n_batches_train, leave=False) as pbar:
            for batch_id in range(n_batches_train):
                x_batch_SP = (
                    torch.concatenate(
                        [
                            dataset.data[
                                indices[
                                    batch_id
                                    * n_sequences_per_repertoire_in_batch : (
                                        batch_id + 1
                                    )
                                    * n_sequences_per_repertoire_in_batch
                                ]
                            ]
                            for dataset, indices in zip(
                                sequence_datasets_train, shuffled_sequence_indices
                            )
                        ],
                        dim=0,
                    )
                    .to(torch.long)
                    .to(device)
                )
                self_predictions: AIRRTM_ModelOutput = model(
                    self_repertoire_ids, x_batch_SP
                )
                self_targets = AIRRTM_ModelTarget(
                    sequence_repertoire_indicators=sequence_repertoire_indicators,
                    sequence_repertoire_labels=sequence_repertoire_labels,
                    sequences=x_batch_SP,
                )
                self_loss = criterion(self_predictions, self_targets)

                other_predictions: AIRRTM_ModelOutput = model(
                    other_repertoire_ids, x_batch_SP
                )
                other_targets = AIRRTM_ModelTarget(
                    sequence_repertoire_indicators=1 - sequence_repertoire_indicators,
                    sequence_repertoire_labels=1 - sequence_repertoire_labels,
                    sequences=x_batch_SP,
                )
                other_loss = criterion(other_predictions, other_targets)

                loss = {
                    key: (self_loss[key] + other_loss[key]) / 2 for key in loss_keys
                }
                loss["total_loss"].backward()
                optimizer.step()
                for key in loss_keys:
                    running_loss[key] += loss[key].item()  # the loss is normalized here
                averaged_loss = {
                    key: running_loss[key] / (batch_id + 1) for key in loss_keys
                }
                if batch_id % 5 == 1:
                    pbar.set_postfix(
                        batch=batch_id,
                        **{
                            LOSS_KEY_TO_PRINT[key]: value
                            for key, value in averaged_loss.items()
                        },
                    )
                    if writer is not None:
                        for key in loss_keys:
                            writer.add_scalars(
                                key,
                                {
                                    "Train": averaged_loss[key],
                                },
                                epoch * n_batches_train + batch_id,
                            )
                pbar.update(1)
            train_loss_list.append(
                {key: running_loss[key] / n_batches_train for key in loss_keys}
            )

        model.eval()
        with torch.no_grad():
            shuffled_sequence_indices_val = [
                torch.randperm(size) for size in repertoire_sizes_val
            ]
            other_repertoire_ids = _sample_other_repertoire_ids(
                repertoire_labels, n_sequences_per_repertoire_in_batch
            ).to(device)
            running_loss = {key: 0.0 for key in loss_keys}
            for batch_id in range(n_batches_val):
                x_batch_SP = (
                    torch.concatenate(
                        [
                            dataset.data[
                                indices[
                                    batch_id
                                    * n_sequences_per_repertoire_in_batch : (
                                        batch_id + 1
                                    )
                                    * n_sequences_per_repertoire_in_batch
                                ]
                            ]
                            for dataset, indices in zip(
                                sequence_datasets_val, shuffled_sequence_indices_val
                            )
                        ]
                    )
                    .to(torch.long)
                    .to(device)
                )
                self_predictions: AIRRTM_ModelOutput = model(
                    self_repertoire_ids, x_batch_SP
                )
                self_targets = AIRRTM_ModelTarget(
                    sequence_repertoire_indicators=sequence_repertoire_indicators,
                    sequence_repertoire_labels=sequence_repertoire_labels,
                    sequences=x_batch_SP,
                )
                self_loss = criterion(self_predictions, self_targets)

                other_predictions: AIRRTM_ModelOutput = model(
                    other_repertoire_ids, x_batch_SP
                )
                other_targets = AIRRTM_ModelTarget(
                    sequence_repertoire_indicators=1 - sequence_repertoire_indicators,
                    sequence_repertoire_labels=1 - sequence_repertoire_labels,
                    sequences=x_batch_SP,
                )
                other_loss = criterion(other_predictions, other_targets)

                loss = {
                    key: (self_loss[key] + other_loss[key]) / 2 for key in loss_keys
                }
                for key in loss_keys:
                    running_loss[key] += loss[key].item()  # the loss is normalized here
            val_loss_list.append(
                {key: running_loss[key] / n_batches_val for key in loss_keys}
            )

            # Write to tensorboard
            if writer is not None:
                for key in loss_keys:
                    writer.add_scalars(
                        key,
                        {
                            "Val": val_loss_list[-1][key],
                        },
                        (epoch + 1) * n_batches_train - 1,
                    )

            if val_loss_list[-1]["total_loss"] < best_val_loss:
                best_val_loss = val_loss_list[-1]["total_loss"]
                best_state_dict = model.state_dict().copy()
                best_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Stopping at epoch {epoch}")
                    break

        if checkpoint_dir is not None:
            torch.save(model, checkpoint_dir / f"checkpoint_epoch_{epoch}.py")
    if writer is not None:
        writer.close()

    if keep_best_model:
        model.load_state_dict(best_state_dict)
        print(f"Saving the model state at the best epoch ({best_epoch})")


def _sample_other_repertoire_ids(
    repertoire_labels: torch.Tensor,
    n_sequences_per_batch: int,
) -> torch.Tensor:
    positive_repertoire_ids = torch.where(repertoire_labels == 1)[0]
    negative_repertoire_ids = torch.where(repertoire_labels == 0)[0]
    other_repertoire_ids = [
        negative_repertoire_ids if label == 1 else positive_repertoire_ids
        for label in repertoire_labels
    ]
    other_repertoire_ids = [
        repertoire_ids[
            torch.randint(
                low=0, high=repertoire_ids.shape[0], size=(n_sequences_per_batch,)
            )
        ]
        for repertoire_ids in other_repertoire_ids
    ]
    other_repertoire_ids = torch.concatenate(other_repertoire_ids)
    return other_repertoire_ids
