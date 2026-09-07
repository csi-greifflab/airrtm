import pytest
import torch

from airrtm.models import model_factory
from airrtm.training.train import _make_batch, _sampling_probabilities, train_model
from airrtm.utils import SequenceDataset, check_consistent_max_length, split_sequences

from conftest import MAX_LENGTH, make_dataset, make_model_config


def test_dataset_roundtrip(tmp_path):
    dataset = make_dataset(30, with_extras=True)
    path = tmp_path / "repertoire.pt"
    dataset.save(path)
    loaded = SequenceDataset.load(path)

    assert loaded.size() == dataset.size()
    assert torch.equal(loaded.data, dataset.data)
    assert torch.equal(loaded.weights, dataset.weights)
    assert torch.equal(loaded.v_ids, dataset.v_ids)
    assert loaded.v_genes == dataset.v_genes
    assert loaded.to_strings() == dataset.to_strings()


def test_dataset_without_extras_still_loads(tmp_path):
    """Datasets written before weights/V/J existed must keep working."""
    dataset = make_dataset(10, with_extras=False)
    path = tmp_path / "old.pt"
    dataset.save(path)
    loaded = SequenceDataset.load(path)
    assert loaded.weights is None
    assert loaded.v_ids is None
    assert loaded.sampling_probabilities() is None
    assert not loaded.has_vj


def test_indexing_carries_extras():
    dataset = make_dataset(20, with_extras=True)
    subset = dataset[torch.arange(5)]
    assert subset.size() == 5
    assert subset.weights.shape == (5,)
    assert subset.v_ids.shape == (5,)
    assert subset.v_genes == dataset.v_genes


def test_truncate_makes_max_length_exact():
    dataset = make_dataset(10, with_extras=False)
    dataset.truncate(6)
    assert dataset.max_length == 6
    assert dataset.data.shape[1] == 6
    assert int(dataset.lengths.max()) <= 6
    assert all(len(s) <= 6 for s in dataset.to_strings())


def test_inconsistent_max_length_is_rejected():
    """Regression: `pad` silently ignores a shorter target, so repertoires could end
    up different widths and only fail later inside torch.concatenate."""
    a, b = make_dataset(5), make_dataset(5)
    b.truncate(MAX_LENGTH - 2)
    with pytest.raises(ValueError, match="inconsistent max_length"):
        check_consistent_max_length([a, b])


def test_sampling_probabilities_follow_abundance():
    dataset = make_dataset(4, with_extras=False)
    dataset.weights = torch.tensor([1.0, 1.0, 1.0, 97.0])
    probabilities = dataset.sampling_probabilities()
    assert probabilities.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert probabilities[3] > 0.9

    draws = torch.multinomial(probabilities, num_samples=2000, replacement=True)
    assert (draws == 3).float().mean() > 0.9


def test_split_sequences_partitions_each_repertoire():
    datasets = [make_dataset(50) for _ in range(3)]
    train, val = split_sequences(datasets, val_size=0.2)
    for original, tr, va in zip(datasets, train, val):
        assert tr.size() + va.size() == original.size()
        assert va.size() == 10


def test_make_batch_layout(alphabet_length):
    datasets = [make_dataset(40, with_extras=True) for _ in range(3)]
    probabilities = _sampling_probabilities(datasets, abundance_weighted=True)
    batch = _make_batch(
        datasets=datasets,
        probabilities=probabilities,
        repertoire_indices=torch.tensor([0, 2]),
        repertoire_labels=torch.tensor([1, 0, 1]),
        n_sequences=7,
        device=torch.device("cpu"),
    )
    assert batch.sequences.shape == (14, MAX_LENGTH)
    assert batch.repertoire_ids.tolist() == [0] * 7 + [2] * 7
    assert batch.labels.tolist() == [1] * 7 + [1] * 7
    assert batch.v_ids.shape == (14,)
    assert batch.weights.shape == (14,)


def test_train_model_smoke(tmp_path, criterion, alphabet_length):
    """The whole loop must run end to end; the previous version raised TypeError."""
    n_repertoires = 4
    datasets = [make_dataset(60, with_extras=True) for _ in range(n_repertoires)]
    train, val = split_sequences(datasets, val_size=0.2)
    labels = torch.tensor([1, 0, 1, 0])

    model = model_factory(**make_model_config(n_repertoires, alphabet_length))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = model.z_mean_layer.weight.detach().clone()
    history = train_model(
        model=model,
        sequence_datasets_train=train,
        sequence_datasets_val=val,
        repertoire_labels=labels,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=2,
        patience=10,
        n_sequences_per_repertoire_in_batch=5,
        n_repertoires_in_batch=2,
        min_repertoire_size=16,
        tau=1.0,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=None,
    )

    assert len(history["train"]) == 2
    assert len(history["val"]) == 2
    assert all(torch.isfinite(torch.tensor(m["total_loss"])) for m in history["train"])
    assert not torch.equal(before, model.z_mean_layer.weight)
    assert list((tmp_path / "checkpoints").glob("checkpoint_epoch_*.pt"))


def test_optimizer_gradients_do_not_accumulate_across_steps(
    tmp_path, criterion, alphabet_length
):
    """Regression: the old loop called backward() and step() but never zero_grad()."""
    datasets = [make_dataset(40, with_extras=False) for _ in range(2)]
    train, val = split_sequences(datasets, val_size=0.25)
    model = model_factory(**make_model_config(2, alphabet_length))

    seen = []

    class RecordingAdam(torch.optim.Adam):
        def step(self, *args, **kwargs):
            seen.append(
                float(
                    torch.norm(
                        torch.stack(
                            [
                                p.grad.norm()
                                for p in model.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                )
            )
            return super().step(*args, **kwargs)

    train_model(
        model=model,
        sequence_datasets_train=train,
        sequence_datasets_val=val,
        repertoire_labels=torch.tensor([1, 0]),
        criterion=criterion,
        optimizer=RecordingAdam(model.parameters(), lr=1e-4),
        n_epochs=1,
        patience=10,
        n_sequences_per_repertoire_in_batch=5,
        n_repertoires_in_batch=2,
        min_repertoire_size=16,
        tau=1.0,
        log_dir=None,
    )
    assert len(seen) >= 2
    # Accumulating gradients would make the norm grow roughly linearly with the step
    # count; with zero_grad in place it stays on the same scale.
    assert max(seen) < 20 * min(seen) + 1.0


def test_model_save_load_roundtrip(tmp_path, alphabet_length):
    from airrtm.cli.train import load_model, save_model

    config = make_model_config(3, alphabet_length)
    model = model_factory(**config)
    path = tmp_path / "model.pt"
    save_model(model, config, path)

    reloaded = load_model(path)
    sequences = torch.randint(0, alphabet_length + 1, (8, MAX_LENGTH))
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(
            model.predict_topic_probabilities(sequences),
            reloaded.predict_topic_probabilities(sequences),
            atol=1e-6,
        )


def test_make_batch_theta_sample_is_disjoint():
    """With uniform sampling the scored and Theta draws must not overlap."""
    from airrtm.training.train import _sample_indices

    for _ in range(20):
        scored, theta = _sample_indices(1000, 64, None, 64)
        assert scored.shape == (64,) and theta.shape == (64,)
        assert len(set(scored.tolist()) & set(theta.tolist())) == 0

    datasets = [make_dataset(200, with_extras=True) for _ in range(2)]
    batch = _make_batch(
        datasets=datasets,
        probabilities=_sampling_probabilities(datasets, abundance_weighted=False),
        repertoire_indices=torch.tensor([0, 1]),
        repertoire_labels=torch.tensor([1, 0]),
        n_sequences=16,
        device=torch.device("cpu"),
        n_theta_sequences=32,
    )
    assert batch.sequences.shape[0] == 32
    assert batch.theta_sequences.shape[0] == 64
    assert batch.theta_repertoire_ids.tolist() == [0] * 32 + [1] * 32
    assert batch.theta_weights.shape == (64,)


def test_no_theta_sample_when_disabled():
    datasets = [make_dataset(60) for _ in range(2)]
    batch = _make_batch(
        datasets=datasets,
        probabilities=[None, None],
        repertoire_indices=torch.tensor([0, 1]),
        repertoire_labels=torch.tensor([1, 0]),
        n_sequences=8,
        device=torch.device("cpu"),
    )
    assert batch.theta_sequences is None
    assert batch.theta_repertoire_ids is None


def test_stratified_groups_mix_classes():
    """A uniformly random group of 4 is single-class ~14% of the time at 44% positives."""
    from airrtm.training.train import _stratified_groups

    torch.manual_seed(0)
    labels = torch.tensor([1] * 44 + [0] * 56)
    single_class = 0
    n_groups = 0
    for _ in range(50):
        groups = _stratified_groups(labels, 4)
        for g in groups:
            g_labels = labels[g]
            n_groups += 1
            if g_labels.min() == g_labels.max():
                single_class += 1
    assert n_groups > 0
    assert single_class / n_groups < 0.02, f"{single_class}/{n_groups} groups single-class"


def test_stratified_groups_cover_repertoires_without_repeats():
    from airrtm.training.train import _stratified_groups

    labels = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1])
    groups = _stratified_groups(labels, 2)
    flat = torch.cat(groups).tolist()
    assert len(flat) == len(set(flat)), "a repertoire appears twice in one epoch"
    assert set(flat) <= set(range(len(labels)))
    assert len(groups) == len(labels) // 2


def test_stratified_groups_handle_single_class_input():
    """Must not hang or crash when one class is absent."""
    from airrtm.training.train import _stratified_groups

    groups = _stratified_groups(torch.tensor([1, 1, 1, 1, 1, 1]), 2)
    assert len(groups) == 3
    assert len(torch.cat(groups)) == 6
