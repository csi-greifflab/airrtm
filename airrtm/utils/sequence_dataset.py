import logging
import pathlib as pl

import torch
from torch.utils.data import Dataset

from airrtm.utils.constants import ALPHABETS


__all__ = ["SequenceDataset"]


class SequenceDataset(Dataset):
    """A dataset of nucleotide or amino acid sequences.

    Parameters
    ----------
    sequence_data : list[str] | SequenceDataset | tuple[torch.Tensor, torch.Tensor]
    alphabet_name : str, optional
        "aa" or "nt", default="aa"
    max_length : int, optional
        Length to pads the sequences to.
    device : torch.device, optional
        Torch device
    """

    # TODO change type option to SequenceDataset
    def __init__(
        self,
        sequence_data: list[str] | Dataset | tuple[torch.Tensor, torch.Tensor],
        alphabet_name: str = "aa",
        max_length: int = -1,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.tensor(0.0).device

        self.alphabet_name = alphabet_name
        if alphabet_name not in ALPHABETS:
            raise ValueError(
                f"Unknown alphabet key: {alphabet_name}, supported options: {ALPHABETS.keys()}"
            )
        alphabet = ALPHABETS[alphabet_name]
        self.alphabet = alphabet
        self.alphabet_length = len(alphabet)
        self.pad_value = self.alphabet_length

        # Load or convert data.
        if isinstance(sequence_data, list):
            self.data, self.lengths = SequenceDataset.from_strings(
                sequences=sequence_data,
                pad_value=self.pad_value,
                alphabet=self.alphabet,
            )
        elif isinstance(sequence_data, tuple):
            self.data, self.lengths = sequence_data
        elif isinstance(sequence_data, SequenceDataset):
            self.data, self.lengths = (
                sequence_data.data.detach().clone(),
                sequence_data.lengths.detach().clone(),
            )
        else:
            raise ValueError("Invalid data format.")

        self.data = self.data.type(torch.uint8)
        self.lengths = self.lengths.type(torch.int64)

        self.max_length = self.data.shape[1]
        if max_length != -1:
            self.pad(max_length)

        self.data = self.data.to(self.device)
        self.lengths = self.lengths.to(self.device)

    def __len__(self):
        return self.size()

    def size(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, indices):
        return SequenceDataset(
            sequence_data=(self.data[indices], self.lengths[indices]),
            alphabet_name=self.alphabet_name,
            max_length=self.max_length,
            device=self.device,
        )

    def to_strings(self) -> list[str]:
        return [
            "".join(self.alphabet[x] for x in s[:l])
            for s, l in zip(self.data.detach().cpu(), self.lengths.detach().cpu())
        ]

    @staticmethod
    def from_strings(
        sequences: list[str], pad_value: int, alphabet: tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alphabet_map = {c: i for i, c in enumerate(alphabet)}

        max_length = max([len(s) for s in sequences])

        data = torch.full((len(sequences), max_length), pad_value, dtype=torch.int8)
        lengths = torch.zeros(len(sequences), dtype=torch.int64)

        for i, s in enumerate(sequences):
            lengths[i] = len(s)
            for j, c in enumerate(s):
                data[i, j] = alphabet_map[c]

        return data, lengths

    def pad(self, length: int) -> None:
        actual_data_length = self.data.shape[1]
        if length < actual_data_length:
            logging.warning(
                f"Actual maximum sequence length ({actual_data_length}) is higher than the received length"
                + f"({length}), ignoring the argument"
            )
            return
        if length == actual_data_length:
            return
        self.data = torch.nn.functional.pad(
            self.data, (0, length - actual_data_length), value=self.pad_value
        )
        self.max_length = length

    def save(self, path: pl.Path) -> None:
        torch.save(
            {
                "alphabet_name": self.alphabet_name,
                "alphabet": self.alphabet,
                "data": self.data,
                "lengths": self.lengths,
            },
            path,
        )

    @classmethod
    def load(cls, path: pl.Path, device: torch.device = None) -> "SequenceDataset":
        device = device or torch.tensor(0.0).device
        loaded_dict = torch.load(path, map_location=device, weights_only=True)
        return cls(
            sequence_data=(loaded_dict["data"], loaded_dict["lengths"]),
            alphabet_name=loaded_dict["alphabet_name"],
            device=device,
        )
