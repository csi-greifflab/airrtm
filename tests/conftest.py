import random

import pytest
import torch

from airrtm.losses import CompositeLoss
from airrtm.models import model_factory
from airrtm.types import AIRRTM_ModelTarget
from airrtm.utils import AA_ALPHABET, SequenceDataset


MAX_LENGTH = 12
N_TOPICS_SIGNAL = 3
N_TOPICS_NONSIGNAL = 4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    random.seed(0)


@pytest.fixture
def alphabet_length() -> int:
    return len(AA_ALPHABET)


def make_model_config(
    n_repertoires: int,
    alphabet_length: int,
    theta_mode: str = "amortized",
    n_v_genes: int = 0,
    n_j_genes: int = 0,
) -> dict:
    transformer_params = {
        "max_sequence_length": MAX_LENGTH,
        "alphabet_length": alphabet_length,
        "attention_dim": 32,
        "attention_dim_head": 32,
        "attention_heads": 2,
        "depth": 2,
        "use_positional_encodings": True,
    }
    return {
        "encoder_type": "transformer",
        "encoder_params": {**transformer_params, "pooling": "mean_max"},
        "decoder_type": "transformer",
        "decoder_params": dict(transformer_params),
        "airrtm_params": {
            "n_repertoires": n_repertoires,
            "latent_dim": 8,
            "n_topics_signal": N_TOPICS_SIGNAL,
            "n_topics_nonsignal": N_TOPICS_NONSIGNAL,
            "theta_mode": theta_mode,
            "n_v_genes": n_v_genes,
            "n_j_genes": n_j_genes,
            "vj_embedding_dim": 4,
        },
    }


@pytest.fixture
def model(alphabet_length):
    return model_factory(**make_model_config(4, alphabet_length))


@pytest.fixture
def criterion(alphabet_length):
    return CompositeLoss(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        topic_l1_coef=0.1,
        theta_entropy_coef=0.1,
        topic_decorrelation_coef=0.1,
        n_sequences_per_repertoire=5,
        pad_value=alphabet_length,
    )


def random_sequences(n: int, min_len: int = 5, max_len: int = MAX_LENGTH) -> list[str]:
    return [
        "".join(random.choice(AA_ALPHABET) for _ in range(random.randint(min_len, max_len)))
        for _ in range(n)
    ]


def make_dataset(n: int = 40, with_extras: bool = False) -> SequenceDataset:
    sequences = random_sequences(n)
    dataset = SequenceDataset(
        sequence_data=sequences,
        alphabet_name="aa",
        max_length=MAX_LENGTH,
        weights=torch.randint(1, 20, (n,)).float() if with_extras else None,
        v_ids=torch.randint(0, 5, (n,)) if with_extras else None,
        j_ids=torch.randint(0, 3, (n,)) if with_extras else None,
        v_genes=tuple(f"V{i}" for i in range(5)) if with_extras else None,
        j_genes=tuple(f"J{i}" for i in range(3)) if with_extras else None,
    )
    dataset.truncate(MAX_LENGTH)
    return dataset


def make_batch(n_repertoires: int = 4, n_per_repertoire: int = 5, alphabet_length: int = 21):
    """A batch laid out the way the trainer builds one: contiguous per-repertoire blocks."""
    total = n_repertoires * n_per_repertoire
    sequences = torch.randint(0, alphabet_length + 1, (total, MAX_LENGTH))
    repertoire_ids = torch.arange(n_repertoires).repeat_interleave(n_per_repertoire)
    labels = (torch.arange(n_repertoires) % 2).repeat_interleave(n_per_repertoire)
    return sequences, repertoire_ids, labels


def make_target(sequences, repertoire_ids, labels) -> AIRRTM_ModelTarget:
    return AIRRTM_ModelTarget(
        sequence_repertoire_indicators=torch.ones_like(labels),
        sequence_repertoire_labels=labels,
        sequences=sequences,
        repertoire_ids=repertoire_ids,
    )
