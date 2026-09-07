import pathlib as pl

import pandas as pd
import torch

from tqdm import tqdm

from airrtm.utils.constants import GENE_VOCABULARY_FILENAME, METADATA_FILENAME
from airrtm.utils.sequence_dataset import SequenceDataset


__all__ = [
    "load_metadata",
    "load_repertoires",
    "load_gene_vocabulary",
    "split_sequences",
    "check_consistent_max_length",
]


def load_metadata(input_dir: pl.Path, split: str | None = None) -> pd.DataFrame:
    """Read the dataset metadata, optionally restricted to one split."""
    metadata_df = pd.read_csv(pl.Path(input_dir) / METADATA_FILENAME)
    if split is not None:
        metadata_df = metadata_df.loc[metadata_df["split"] == split]
    return metadata_df.reset_index(drop=True)


def load_repertoires(
    input_dir: pl.Path,
    metadata_df: pd.DataFrame,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> tuple[list[SequenceDataset], torch.Tensor]:
    """Load every repertoire named in ``metadata_df``, with its label."""
    input_dir = pl.Path(input_dir)
    datasets = [
        SequenceDataset.load(input_dir / filename, device=device)
        for filename in tqdm(
            metadata_df["filename"], disable=not show_progress, desc="loading"
        )
    ]
    labels = torch.tensor(metadata_df["label"].to_numpy(), dtype=torch.long)
    check_consistent_max_length(datasets)
    return datasets, labels


def load_gene_vocabulary(
    input_dir: pl.Path,
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    path = pl.Path(input_dir) / GENE_VOCABULARY_FILENAME
    if not path.exists():
        return None, None
    payload = torch.load(path, weights_only=False)
    return payload.get("v_genes"), payload.get("j_genes")


def check_consistent_max_length(datasets: list[SequenceDataset]) -> int:
    """Fail loudly when repertoires disagree on the padded sequence length.

    ``SequenceDataset.pad`` silently declines to shorten data that is already longer
    than the requested length, so a preprocessing run can leave repertoires with
    different widths -- which then fails deep inside the training loop's
    ``torch.concatenate`` with an unhelpful message.
    """
    lengths = {d.max_length for d in datasets}
    if len(lengths) > 1:
        raise ValueError(
            f"Repertoires have inconsistent max_length values: {sorted(lengths)}. "
            "Re-run preprocessing with a single --max_len."
        )
    return lengths.pop() if lengths else 0


def split_sequences(
    datasets: list[SequenceDataset],
    val_size: float,
    generator: torch.Generator | None = None,
) -> tuple[list[SequenceDataset], list[SequenceDataset]]:
    """Hold out a fraction of each repertoire's sequences.

    This is a *sequence*-level split within the same repertoires: it measures whether
    the model generalises to unseen sequences, not to unseen individuals. Repertoire
    generalisation is measured on the held-out split of ``metadata.csv`` instead.
    """
    train, val = [], []
    for dataset in datasets:
        indices = torch.randperm(dataset.size(), generator=generator)
        n_val = int(indices.shape[0] * val_size)
        val.append(dataset[indices[:n_val]])
        train.append(dataset[indices[n_val:]])
    return train, val
