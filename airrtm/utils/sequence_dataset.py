import logging
import pathlib as pl

import torch
from torch.utils.data import Dataset

from airrtm.utils.constants import ALPHABETS


__all__ = ["SequenceDataset"]


class SequenceDataset(Dataset):
    """A dataset of nucleotide or amino acid sequences.

    Optionally carries per-sequence clonal abundances (``weights``) and V/J gene
    assignments (``v_ids``, ``j_ids``). All three are ``None`` for datasets written
    before those columns were supported, so old ``.pt`` files still load.

    Parameters
    ----------
    sequence_data : list[str] | SequenceDataset | tuple[torch.Tensor, torch.Tensor] | dict
        A list of sequence strings, another dataset to copy, a ``(data, lengths)``
        tuple, or a dict with keys ``data``, ``lengths`` and optionally ``weights``,
        ``v_ids``, ``j_ids``.
    alphabet_name : str, optional
        "aa" or "nt", default="aa"
    max_length : int, optional
        Length to pads the sequences to.
    device : torch.device, optional
        Torch device
    weights : torch.Tensor, optional
        Per-sequence clonal abundance (e.g. ``duplicate_count``). Ignored when
        ``sequence_data`` already carries weights.
    v_ids, j_ids : torch.Tensor, optional
        Integer V/J gene indices. Ignored when ``sequence_data`` already carries them.
    v_genes, j_genes : tuple[str, ...], optional
        The vocabulary the ids index into, kept so predictions can be mapped back.
    """

    def __init__(
        self,
        sequence_data: list[str] | Dataset | tuple[torch.Tensor, torch.Tensor] | dict,
        alphabet_name: str = "aa",
        max_length: int = -1,
        device: torch.device = None,
        weights: torch.Tensor | None = None,
        v_ids: torch.Tensor | None = None,
        j_ids: torch.Tensor | None = None,
        v_genes: tuple[str, ...] | None = None,
        j_genes: tuple[str, ...] | None = None,
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

        self.v_genes = v_genes
        self.j_genes = j_genes

        # Load or convert data.
        if isinstance(sequence_data, list):
            self.data, self.lengths = SequenceDataset.from_strings(
                sequences=sequence_data,
                pad_value=self.pad_value,
                alphabet=self.alphabet,
            )
            self.weights, self.v_ids, self.j_ids = weights, v_ids, j_ids
        elif isinstance(sequence_data, dict):
            self.data, self.lengths = sequence_data["data"], sequence_data["lengths"]
            self.weights = sequence_data.get("weights", weights)
            self.v_ids = sequence_data.get("v_ids", v_ids)
            self.j_ids = sequence_data.get("j_ids", j_ids)
            self.v_genes = sequence_data.get("v_genes", v_genes)
            self.j_genes = sequence_data.get("j_genes", j_genes)
        elif isinstance(sequence_data, tuple):
            self.data, self.lengths = sequence_data
            self.weights, self.v_ids, self.j_ids = weights, v_ids, j_ids
        elif isinstance(sequence_data, SequenceDataset):
            self.data, self.lengths = (
                sequence_data.data.detach().clone(),
                sequence_data.lengths.detach().clone(),
            )
            self.weights = _clone_or_none(sequence_data.weights)
            self.v_ids = _clone_or_none(sequence_data.v_ids)
            self.j_ids = _clone_or_none(sequence_data.j_ids)
            self.v_genes = sequence_data.v_genes
            self.j_genes = sequence_data.j_genes
        else:
            raise ValueError("Invalid data format.")

        self.data = self.data.type(torch.uint8)
        self.lengths = self.lengths.type(torch.int64)
        if self.weights is not None:
            self.weights = self.weights.type(torch.float32)
        if self.v_ids is not None:
            self.v_ids = self.v_ids.type(torch.int64)
        if self.j_ids is not None:
            self.j_ids = self.j_ids.type(torch.int64)

        self.max_length = self.data.shape[1]
        if max_length != -1:
            self.pad(max_length)

        self.data = self.data.to(self.device)
        self.lengths = self.lengths.to(self.device)
        for name in ("weights", "v_ids", "j_ids"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(self.device))

    def __len__(self):
        return self.size()

    def size(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_v_genes(self) -> int:
        return len(self.v_genes) if self.v_genes is not None else 0

    @property
    def n_j_genes(self) -> int:
        return len(self.j_genes) if self.j_genes is not None else 0

    @property
    def has_vj(self) -> bool:
        return self.v_ids is not None and self.j_ids is not None

    def sampling_probabilities(self) -> torch.Tensor | None:
        """Normalised clonal abundances, for abundance-weighted batch sampling.

        Returns ``None`` when the dataset carries no weights, in which case callers
        should fall back to uniform sampling.
        """
        if self.weights is None:
            return None
        weights = self.weights.to(torch.float64).clamp(min=0)
        total = weights.sum()
        if total <= 0:
            return None
        return (weights / total).to(torch.float32)

    def __getitem__(self, indices):
        return SequenceDataset(
            sequence_data={
                "data": self.data[indices],
                "lengths": self.lengths[indices],
                "weights": _index_or_none(self.weights, indices),
                "v_ids": _index_or_none(self.v_ids, indices),
                "j_ids": _index_or_none(self.j_ids, indices),
                "v_genes": self.v_genes,
                "j_genes": self.j_genes,
            },
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

        data = torch.full((len(sequences), max_length), pad_value, dtype=torch.int16)
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

    def truncate(self, length: int) -> None:
        """Drop positions beyond ``length``, shortening ``lengths`` accordingly.

        Together with :meth:`pad` this makes ``max_length`` an exact guarantee rather
        than a suggestion, so datasets from different repertoires can be concatenated.
        """
        if length >= self.data.shape[1]:
            return
        self.data = self.data[:, :length].contiguous()
        self.lengths = self.lengths.clamp(max=length)
        self.max_length = length

    def save(self, path: pl.Path) -> None:
        payload = {
            "alphabet_name": self.alphabet_name,
            "alphabet": self.alphabet,
            "data": self.data,
            "lengths": self.lengths,
        }
        if self.weights is not None:
            payload["weights"] = self.weights
        if self.v_ids is not None:
            payload["v_ids"] = self.v_ids
            payload["v_genes"] = self.v_genes
        if self.j_ids is not None:
            payload["j_ids"] = self.j_ids
            payload["j_genes"] = self.j_genes
        torch.save(payload, path)

    @classmethod
    def load(cls, path: pl.Path, device: torch.device = None) -> "SequenceDataset":
        device = device or torch.tensor(0.0).device
        # v_genes/j_genes are tuples of str, which weights_only=True refuses to unpickle.
        loaded_dict = torch.load(path, map_location=device, weights_only=False)
        return cls(
            sequence_data={
                "data": loaded_dict["data"],
                "lengths": loaded_dict["lengths"],
                "weights": loaded_dict.get("weights"),
                "v_ids": loaded_dict.get("v_ids"),
                "j_ids": loaded_dict.get("j_ids"),
                "v_genes": loaded_dict.get("v_genes"),
                "j_genes": loaded_dict.get("j_genes"),
            },
            alphabet_name=loaded_dict["alphabet_name"],
            device=device,
        )


def _clone_or_none(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().clone()


def _index_or_none(tensor: torch.Tensor | None, indices) -> torch.Tensor | None:
    return None if tensor is None else tensor[indices]
