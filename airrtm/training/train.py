import pathlib as pl

from dataclasses import dataclass

import numpy as np
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
    "reconstruction_loss": "rec",
    "reconstruction_accuracy": "rec_acc",
    "kl_divergence": "kl_d",
    "tm_loss": "tm",
    "label_loss": "label",
    "label_accuracy": "label_acc",
    "topic_l1": "l1",
    "theta_entropy": "H_theta",
    "topic_usage_entropy": "H_usage",
    "topic_decorrelation": "decorr",
    "phi_l2": "phi_l2",
    "predicted_witness_rate": "wr",
}

#: Repertoires smaller than this are sampled with replacement, so that every
#: repertoire contributes the same number of sequences per epoch regardless of size.
DEFAULT_MIN_REPERTOIRE_SIZE = 32768


def _tm_random_baseline(
    n_repertoires_in_batch: int,
    n_sequences_per_repertoire_in_batch: int,
    tm_max_negatives: int | None,
    tm_likelihood_family: str = "batch_softmax",
) -> float:
    """Expected ``tm_loss`` with a completely uninformative theta/phi.

    ``tm_likelihood_family="batch_softmax"``: ``tm_loss`` is a softmax negative
    log-likelihood over the batch's sequences (``CompositeLoss._tm_loss``): with no
    real signal, every candidate scores about equally, so the loss sits at
    ``log(N)`` for ``N`` candidates -- not ``-log(0.5)``, which is the *label*
    loss's chance value, not this one. ``tm_loss`` barely moves off this ceiling
    even when the TM branch is doing real work (on the order of 1e-2 to 1e-3 nats),
    which is why ``tm_gain`` below rescales the gap rather than the raw loss.

    ``tm_likelihood_family="raw_bce"``: ``tm_loss`` is a plain binary
    cross-entropy (``CompositeLoss._tm_loss_raw_bce``), not a softmax over a
    candidate pool -- its chance value *is* ``-log(0.5)``, same as the label loss.
    """
    if tm_likelihood_family == "raw_bce":
        return float(np.log(2.0))
    pool_size = n_repertoires_in_batch * n_sequences_per_repertoire_in_batch
    if tm_max_negatives is not None:
        pool_size = min(pool_size, tm_max_negatives)
    return float(np.log(pool_size))


@dataclass
class Batch:
    """One training step's worth of sequences, drawn from several repertoires.

    The ``theta_*`` fields hold a second, disjoint draw from the same repertoires,
    used only to infer their topic proportions. They are ``None`` when Theta is
    pooled from the scored sequences themselves.
    """

    sequences: torch.Tensor  # (S, P) long
    repertoire_ids: torch.Tensor  # (S,) long, index into the full repertoire list
    labels: torch.Tensor  # (S,) repertoire label, broadcast per sequence
    v_ids: torch.Tensor | None
    j_ids: torch.Tensor | None
    weights: torch.Tensor | None  # (S,) clonal abundance, or None
    theta_sequences: torch.Tensor | None = None
    theta_repertoire_ids: torch.Tensor | None = None
    theta_v_ids: torch.Tensor | None = None
    theta_j_ids: torch.Tensor | None = None
    theta_weights: torch.Tensor | None = None


def train_model(
    *,
    model: AIRRTM_Model,
    sequence_datasets_train: list[SequenceDataset],
    sequence_datasets_val: list[SequenceDataset],  #  from the same repertoires as train
    repertoire_labels: torch.Tensor,  # assumed to be 1d and consist of 1s and 0s
    criterion: CompositeLoss,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    patience: int,
    n_sequences_per_repertoire_in_batch: int,
    n_repertoires_in_batch: int = 4,
    min_repertoire_size: int = DEFAULT_MIN_REPERTOIRE_SIZE,
    n_batches_per_repertoire: int | None = None,
    n_batches_per_repertoire_val: int | None = None,
    n_theta_sequences_per_repertoire: int = 0,
    tau: float = 1.0,
    tau_start: float | None = None,
    tau_anneal_epochs: int | None = None,
    tm_likelihood_coef_start: float | None = None,
    theta_entropy_coef_start: float | None = None,
    topic_usage_coef_start: float | None = None,
    coef_anneal_epochs: int | None = None,
    vae_coef_start: float | None = None,
    reconstruction_loss_coef_start: float | None = None,
    vae_anneal_epochs: int | None = None,
    abundance_weighted_sampling: bool = True,
    grad_clip: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
    log_dir: pl.Path | None = None,
    log_every: int = 20,
    checkpoint_dir: pl.Path | None = None,
    checkpoint_every: int = 1,
    keep_best_model: bool = True,
    tm_gain_scale: float = 100.0,
) -> dict[str, list[dict[str, float]]]:
    """Train an AIRRTM model with repertoire mini-batching.

    Each optimizer step sees ``n_repertoires_in_batch`` repertoires contributing
    ``n_sequences_per_repertoire_in_batch`` sequences each. Both the label loss (a
    per-bag MIL term) and the TM loss (normalised over the batch) need several
    repertoires per step, so the repertoire axis is genuinely batched rather than
    taken whole -- taking all 597 Emerson repertoires at once, as the previous
    implementation did, needs ~1.2M sequences in a single forward pass.

    ``tau_start`` optionally anneals the MIL pooling temperature geometrically from
    ``tau_start`` to ``tau`` across epochs: a low temperature pools towards the mean
    (a stable but weak signal), a high one towards the max (sharp, but easy to get
    stuck on a single sequence).

    ``tm_likelihood_coef_start``, ``theta_entropy_coef_start`` and
    ``topic_usage_coef_start`` optionally ramp those three ``criterion`` coefficients
    linearly from their ``_start`` value to the value already set on ``criterion``,
    over ``coef_anneal_epochs`` (default: the whole run). Unlike ``tau``'s geometric
    ramp, this one is linear so it can start at exactly 0.0 (entropy/usage) or end at
    exactly 1.0 (tm_likelihood_coef, which correspondingly drives
    ``criterion.label_likelihood_coef`` to ``1 - tm_likelihood_coef`` each epoch, same
    as ``CompositeLoss.__init__``). Intended for warm-starting the TM likelihood
    before the label loss and entropy regularisers compete for the same shared
    encoder/Theta: set ``tm_likelihood_coef_start=1.0`` and the entropy/usage starts
    to ``0.0`` to give TM an unopposed window, then anneal down/up to the run's real
    target coefficients.

    ``vae_coef_start`` and ``reconstruction_loss_coef_start`` are the same idea one
    level up: they ramp ``criterion.vae_coef`` and ``criterion.reconstruction_loss_coef``
    (with ``criterion.kld_coef`` kept as ``1 - reconstruction_loss_coef`` each epoch,
    same as ``CompositeLoss.__init__``) over ``vae_anneal_epochs`` (default:
    ``coef_anneal_epochs``, so one window covers everything unless a separate one is
    wanted). ``vae_coef`` gates the *entire* non-VAE branch
    (``total = vae_coef * VAE + (1 - vae_coef) * non_vae``), so either direction is a
    real, distinct experiment, unlike the TM-vs-label warm start above, which only
    reshuffles weight *within* the non-VAE branch:

    - ``vae_coef_start`` near ``0.0`` (ramping *up* to the run's real, usually small,
      target) gives TM/label/entropy an unopposed window with literally zero VAE
      gradient reaching the shared encoder at all, regardless of ``vae_coef``'s
      target -- the same regime as the standalone "unopposed TM" diagnostic
      (``theta_mode="free"``, everything but ``tm_likelihood_coef`` zeroed,
      ``vae_coef=0.0`` throughout), except VAE fades back in afterward instead of
      staying off for the whole run.
    - ``vae_coef_start`` near ``1.0`` (ramping *down*) is the opposite: the model
      spends its first epochs as a plain sequence autoencoder, and topic modelling
      and classification are only attached once VAE's weight has faded down.

    Pair either with ``reconstruction_loss_coef_start`` near ``1.0`` for classic KL
    annealing (near-zero KL weight at first, to avoid posterior collapse while the
    encoder is still finding a useful representation) on top.
    """
    device = device or torch.tensor(0.0).device

    writer = SummaryWriter(log_dir=log_dir, flush_secs=1) if log_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir = pl.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_repertoires = len(sequence_datasets_train)
    if n_repertoires < n_repertoires_in_batch:
        raise ValueError(
            f"n_repertoires_in_batch={n_repertoires_in_batch} exceeds the number of "
            f"repertoires ({n_repertoires})"
        )
    repertoire_labels = repertoire_labels.to(device)

    sizes_train = [d.size() for d in sequence_datasets_train]
    sizes_val = [d.size() for d in sequence_datasets_val]
    # How many batches each repertoire contributes per epoch. Sampling is with
    # replacement, so this is a free choice of epoch length rather than a property of
    # the data: with 597 repertoires an uncapped epoch is thousands of steps, which
    # makes validation and early stopping far too coarse to watch.
    if n_batches_per_repertoire is None:
        n_batches_per_repertoire = max(
            min(sizes_train) // n_sequences_per_repertoire_in_batch, 1
        )
    if n_batches_per_repertoire_val is None:
        n_batches_per_repertoire_val = max(
            max(min(sizes_val), min_repertoire_size)
            // n_sequences_per_repertoire_in_batch,
            1,
        )
    n_batches_per_repertoire_train = n_batches_per_repertoire
    n_groups = n_repertoires // n_repertoires_in_batch

    print(f"min rep size train: {min(sizes_train)}, min rep size val: {min(sizes_val)}")
    print(f"max rep size train: {max(sizes_train)}, max rep size val: {max(sizes_val)}")
    print(
        f"steps/epoch: {n_batches_per_repertoire_train * n_groups} train, "
        f"{n_batches_per_repertoire_val * n_groups} val "
        f"({n_batches_per_repertoire_train} x {n_groups} repertoire groups)"
    )
    print(
        f"batch: {n_repertoires_in_batch} repertoires x "
        f"{n_sequences_per_repertoire_in_batch} sequences = "
        f"{n_repertoires_in_batch * n_sequences_per_repertoire_in_batch} sequences"
    )
    tm_random_baseline = _tm_random_baseline(
        n_repertoires_in_batch,
        n_sequences_per_repertoire_in_batch,
        criterion.tm_max_negatives,
        criterion.tm_likelihood_family,
    )
    if model.use_topic_model:
        print(
            f"tm_loss random baseline: {tm_random_baseline:.4f} nats "
            f"(tm_gain = (baseline - tm_loss) * {tm_gain_scale:g})"
        )
    if n_theta_sequences_per_repertoire:
        print(
            f"theta inferred from a disjoint sample of "
            f"{n_theta_sequences_per_repertoire} sequences per repertoire"
        )

    probabilities_train = _sampling_probabilities(
        sequence_datasets_train, abundance_weighted_sampling
    )
    probabilities_val = _sampling_probabilities(
        sequence_datasets_val, abundance_weighted_sampling
    )

    model = model.to(device)
    early_stopping_counter = 0
    best_val_loss = torch.inf
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    loss_keys = _loss_keys(criterion)
    history = {"train": [], "val": []}

    taus = _tau_schedule(tau_start, tau, n_epochs, tau_anneal_epochs)
    tm_coefs = _linear_schedule(
        tm_likelihood_coef_start, criterion.tm_likelihood_coef, n_epochs, coef_anneal_epochs
    )
    entropy_coefs = _linear_schedule(
        theta_entropy_coef_start, criterion.theta_entropy_coef, n_epochs, coef_anneal_epochs
    )
    usage_coefs = _linear_schedule(
        topic_usage_coef_start, criterion.topic_usage_coef, n_epochs, coef_anneal_epochs
    )
    vae_anneal_epochs = vae_anneal_epochs if vae_anneal_epochs is not None else coef_anneal_epochs
    vae_coefs = _linear_schedule(
        vae_coef_start, criterion.vae_coef, n_epochs, vae_anneal_epochs
    )
    rec_coefs = _linear_schedule(
        reconstruction_loss_coef_start,
        criterion.reconstruction_loss_coef,
        n_epochs,
        vae_anneal_epochs,
    )
    global_step = 0
    recent = {key: 0.0 for key in loss_keys}
    for epoch in range(n_epochs):
        epoch_tau = taus[epoch]
        criterion.tm_likelihood_coef = float(tm_coefs[epoch])
        criterion.label_likelihood_coef = 1.0 - float(tm_coefs[epoch])
        criterion.theta_entropy_coef = float(entropy_coefs[epoch])
        criterion.topic_usage_coef = float(usage_coefs[epoch])
        criterion.vae_coef = float(vae_coefs[epoch])
        criterion.reconstruction_loss_coef = float(rec_coefs[epoch])
        criterion.kld_coef = 1.0 - float(rec_coefs[epoch])
        model.train()
        running = {key: 0.0 for key in loss_keys}
        n_steps = 0

        with tqdm(range(n_batches_per_repertoire_train), leave=False) as pbar:
            for _ in pbar:
                for group in _stratified_groups(
                    repertoire_labels, n_repertoires_in_batch
                ):
                    batch = _make_batch(
                        datasets=sequence_datasets_train,
                        probabilities=probabilities_train,
                        repertoire_indices=group,
                        repertoire_labels=repertoire_labels,
                        n_sequences=n_sequences_per_repertoire_in_batch,
                        device=device,
                        n_theta_sequences=n_theta_sequences_per_repertoire,
                    )
                    loss = _step(
                        model=model,
                        criterion=criterion,
                        batch=batch,
                        tau=epoch_tau,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss["total_loss"].backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    n_steps += 1
                    global_step += 1
                    for key in loss_keys:
                        value = float(loss[key].detach())
                        running[key] += value
                        recent[key] += value
                    # Log a moving window rather than only the epoch mean, so the
                    # curves are watchable while a long epoch is still running.
                    if writer is not None and global_step % log_every == 0:
                        for key in loss_keys:
                            writer.add_scalars(
                                key, {"Train": recent[key] / log_every}, global_step
                            )
                        if model.use_topic_model:
                            recent_tm_gain = (
                                tm_random_baseline - recent["tm_loss"] / log_every
                            ) * tm_gain_scale
                            writer.add_scalars(
                                "tm_gain", {"Train": recent_tm_gain}, global_step
                            )
                        writer.add_scalar("tau", epoch_tau, global_step)
                        recent = {key: 0.0 for key in loss_keys}
                pbar.set_postfix(
                    **{
                        LOSS_KEY_TO_PRINT.get(key, key): running[key] / n_steps
                        for key in loss_keys
                    }
                )
        train_metrics = {key: running[key] / max(n_steps, 1) for key in loss_keys}
        if model.use_topic_model:
            train_metrics["tm_gain"] = (
                tm_random_baseline - train_metrics["tm_loss"]
            ) * tm_gain_scale
        history["train"].append(train_metrics)

        val_metrics = evaluate(
            model=model,
            criterion=criterion,
            sequence_datasets=sequence_datasets_val,
            probabilities=probabilities_val,
            repertoire_labels=repertoire_labels,
            n_sequences_per_repertoire_in_batch=n_sequences_per_repertoire_in_batch,
            n_repertoires_in_batch=n_repertoires_in_batch,
            n_batches_per_repertoire=n_batches_per_repertoire_val,
            tau=epoch_tau,
            device=device,
            loss_keys=loss_keys,
            n_theta_sequences_per_repertoire=n_theta_sequences_per_repertoire,
        )
        if model.use_topic_model:
            val_metrics["tm_gain"] = (
                tm_random_baseline - val_metrics["tm_loss"]
            ) * tm_gain_scale
        history["val"].append(val_metrics)

        if writer is not None:
            # Same x-axis as the per-step curves above, so Train and Val overlay.
            for key in loss_keys:
                writer.add_scalars(key, {"Val": val_metrics[key]}, global_step)
            if model.use_topic_model:
                writer.add_scalars(
                    "tm_gain", {"Val": val_metrics["tm_gain"]}, global_step
                )
            writer.add_scalar("epoch", epoch, global_step)

        print(
            f"epoch {epoch}: "
            + " ".join(
                f"{LOSS_KEY_TO_PRINT.get(k, k)}={val_metrics[k]:.4f}" for k in loss_keys
            )
            + (f" tm_gain={val_metrics['tm_gain']:.4f}" if model.use_topic_model else "")
        )

        if scheduler is not None:
            scheduler.step()

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Stopping at epoch {epoch}")
                break

        if checkpoint_dir is not None and epoch % checkpoint_every == 0:
            # A state dict, not a pickled module: pickled modules break as soon as the
            # class is edited, which is how the earlier checkpoints became unloadable.
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict()},
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    if writer is not None:
        writer.close()

    if keep_best_model:
        model.load_state_dict(best_state_dict)
        print(f"Restored the model state at the best epoch ({best_epoch})")
    return history


@torch.no_grad()
def evaluate(
    *,
    model: AIRRTM_Model,
    criterion: CompositeLoss,
    sequence_datasets: list[SequenceDataset],
    probabilities: list[torch.Tensor | None],
    repertoire_labels: torch.Tensor,
    n_sequences_per_repertoire_in_batch: int,
    n_repertoires_in_batch: int,
    n_batches_per_repertoire: int,
    tau: float,
    device: torch.device,
    loss_keys: list[str],
    n_theta_sequences_per_repertoire: int = 0,
) -> dict[str, float]:
    model.eval()
    n_repertoires = len(sequence_datasets)
    n_groups = n_repertoires // n_repertoires_in_batch
    running = {key: 0.0 for key in loss_keys}
    n_steps = 0
    for _ in range(n_batches_per_repertoire):
        for group in _stratified_groups(repertoire_labels, n_repertoires_in_batch):
            batch = _make_batch(
                datasets=sequence_datasets,
                probabilities=probabilities,
                repertoire_indices=group,
                repertoire_labels=repertoire_labels,
                n_sequences=n_sequences_per_repertoire_in_batch,
                device=device,
                n_theta_sequences=n_theta_sequences_per_repertoire,
            )
            loss = _step(model=model, criterion=criterion, batch=batch, tau=tau)
            n_steps += 1
            for key in loss_keys:
                running[key] += float(loss[key].detach())
    return {key: running[key] / max(n_steps, 1) for key in loss_keys}


def _step(
    *,
    model: AIRRTM_Model,
    criterion: CompositeLoss,
    batch: Batch,
    tau: float,
) -> dict[str, torch.Tensor]:
    predictions: AIRRTM_ModelOutput = model(
        batch.repertoire_ids,
        batch.sequences,
        v_ids=batch.v_ids,
        j_ids=batch.j_ids,
        weights=batch.weights,
        theta_sequences=batch.theta_sequences,
        theta_repertoire_ids=batch.theta_repertoire_ids,
        theta_v_ids=batch.theta_v_ids,
        theta_j_ids=batch.theta_j_ids,
        theta_weights=batch.theta_weights,
    )
    targets = AIRRTM_ModelTarget(
        sequence_repertoire_indicators=torch.ones_like(batch.labels),
        sequence_repertoire_labels=batch.labels,
        sequences=batch.sequences,
        repertoire_ids=batch.repertoire_ids,
        weights=batch.weights,
    )
    loss = criterion(predictions, targets, tau=tau)
    if not torch.isfinite(loss["total_loss"]):
        raise RuntimeError(f"Non-finite loss: { {k: float(v) for k, v in loss.items()} }")
    return loss


def _make_batch(
    *,
    datasets: list[SequenceDataset],
    probabilities: list[torch.Tensor | None],
    repertoire_indices: torch.Tensor,
    repertoire_labels: torch.Tensor,
    n_sequences: int,
    device: torch.device,
    n_theta_sequences: int = 0,
) -> Batch:
    """Draw ``n_sequences`` per repertoire, plus an optional disjoint Theta sample.

    Both draws come from one pass so they can be made disjoint: with uniform
    sampling a single permutation is split in two, which guarantees no sequence is
    both scored and used to infer its own repertoire's topic proportions.
    """
    scored = _Columns()
    context = _Columns()
    for repertoire_index in repertoire_indices.tolist():
        dataset = datasets[repertoire_index]
        indices, theta_indices = _sample_indices(
            dataset.size(),
            n_sequences,
            probabilities[repertoire_index],
            n_theta_sequences,
        )
        label = int(repertoire_labels[repertoire_index])
        scored.add(dataset, indices, repertoire_index, label)
        if n_theta_sequences:
            context.add(dataset, theta_indices, repertoire_index, label)

    return Batch(
        sequences=scored.stack("data", device, torch.long),
        repertoire_ids=scored.stack("ids", device),
        labels=scored.stack("labels", device),
        v_ids=scored.stack("v_ids", device),
        j_ids=scored.stack("j_ids", device),
        weights=scored.stack("weights", device),
        theta_sequences=context.stack("data", device, torch.long),
        theta_repertoire_ids=context.stack("ids", device),
        theta_v_ids=context.stack("v_ids", device),
        theta_j_ids=context.stack("j_ids", device),
        theta_weights=context.stack("weights", device),
    )


class _Columns:
    """Accumulates the per-repertoire slices that make up one batch."""

    def __init__(self):
        self.columns = {key: [] for key in ("data", "ids", "labels", "v_ids", "j_ids", "weights")}

    def add(self, dataset: SequenceDataset, indices, repertoire_index: int, label: int):
        n = indices.shape[0]
        self.columns["data"].append(dataset.data[indices])
        self.columns["ids"].append(torch.full((n,), repertoire_index, dtype=torch.long))
        self.columns["labels"].append(torch.full((n,), label, dtype=torch.long))
        if dataset.v_ids is not None:
            self.columns["v_ids"].append(dataset.v_ids[indices])
            self.columns["j_ids"].append(dataset.j_ids[indices])
        if dataset.weights is not None:
            self.columns["weights"].append(dataset.weights[indices])

    def stack(self, key: str, device, dtype=None):
        parts = self.columns[key]
        if not parts:
            return None
        stacked = torch.concatenate(parts, dim=0)
        return (stacked.to(dtype) if dtype is not None else stacked).to(device)


def _sample_indices(
    size: int,
    n_sequences: int,
    probabilities: torch.Tensor | None,
    n_theta_sequences: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Draw sequence indices from one repertoire, plus an optional Theta sample.

    Small and large repertoires contribute equally, so sampling is effectively with
    replacement across epochs; when clonal abundances are available they set the
    sampling weights, and expanded clones are drawn in proportion to their size.

    With uniform sampling the two draws come from one permutation and are therefore
    strictly disjoint. Abundance-weighted sampling draws them independently instead
    (a shared clone may legitimately appear in both), which is enough to break the
    self-reference: Theta is no longer a deterministic function of the scored batch.
    """
    total = n_sequences + n_theta_sequences
    if probabilities is None:
        if total <= size:
            drawn = torch.randperm(size)[:total]
        else:
            drawn = torch.randint(low=0, high=size, size=(total,))
        return drawn[:n_sequences], (drawn[n_sequences:] if n_theta_sequences else None)

    drawn = torch.multinomial(probabilities, num_samples=total, replacement=True)
    return drawn[:n_sequences], (drawn[n_sequences:] if n_theta_sequences else None)


def _stratified_groups(
    repertoire_labels: torch.Tensor,
    n_repertoires_in_batch: int,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Partition the repertoires into groups holding both classes where possible.

    A uniformly random group is single-class surprisingly often -- with 44% positives
    and groups of 4, about 14% of the time (0.44^4 + 0.56^4). Such a step gives the
    label term no contrast at all: every bag carries the same target, so the gradient
    only shifts the bias.

    Each group is seeded with one repertoire of each class, then filled from the
    remaining pool. Simply interleaving the two classes is not enough -- whichever
    class is larger ends up bunched at the tail, and the final groups are single-class
    again.
    """
    labels = repertoire_labels.detach().cpu()
    positive = torch.where(labels == 1)[0]
    negative = torch.where(labels != 1)[0]
    positive = positive[torch.randperm(positive.shape[0], generator=generator)]
    negative = negative[torch.randperm(negative.shape[0], generator=generator)]

    n_groups = labels.shape[0] // n_repertoires_in_batch
    if n_groups == 0:
        return []
    # Seeding needs one of each class per group; with fewer, seed as many as we can.
    n_seeded = min(n_groups, positive.shape[0], negative.shape[0])
    groups: list[list[torch.Tensor]] = [[] for _ in range(n_groups)]
    for k in range(n_seeded):
        groups[k].append(positive[k])
        groups[k].append(negative[k])

    remaining = torch.cat([positive[n_seeded:], negative[n_seeded:]])
    remaining = remaining[torch.randperm(remaining.shape[0], generator=generator)]
    cursor = 0
    for group in groups:
        while len(group) < n_repertoires_in_batch and cursor < remaining.shape[0]:
            group.append(remaining[cursor])
            cursor += 1
    return [
        torch.stack(group)
        for group in groups
        if len(group) == n_repertoires_in_batch
    ]


def _sampling_probabilities(
    datasets: list[SequenceDataset], abundance_weighted: bool
) -> list[torch.Tensor | None]:
    if not abundance_weighted:
        return [None] * len(datasets)
    return [d.sampling_probabilities() for d in datasets]


def _tau_schedule(
    tau_start: float | None,
    tau_end: float,
    n_epochs: int,
    anneal_epochs: int | None = None,
) -> np.ndarray:
    """Geometric ramp of the MIL pooling temperature, then hold at ``tau_end``.

    ``anneal_epochs`` decouples the ramp from the run length. Spreading it over all
    ``n_epochs`` means a long run spends most of its life at a low temperature: with
    1 -> 10 over 200 epochs, tau is still 1.37 at epoch 27, which for bags of ~1e3
    pools almost exactly like the mean.

    Low tau first is still the right shape -- at initialisation the arg-max sequence
    of a bag is effectively random, so a high temperature back-propagates through one
    arbitrary sequence per bag -- but the ramp has to finish early enough to matter.
    """
    if tau_start is None or n_epochs < 2:
        return np.full(max(n_epochs, 1), tau_end)
    anneal_epochs = min(anneal_epochs or n_epochs, n_epochs)
    ramp = np.exp(np.linspace(np.log(tau_start), np.log(tau_end), max(anneal_epochs, 2)))
    if anneal_epochs >= n_epochs:
        return ramp[:n_epochs]
    return np.concatenate([ramp, np.full(n_epochs - anneal_epochs, tau_end)])


def _linear_schedule(
    start: float | None,
    end: float,
    n_epochs: int,
    anneal_epochs: int | None = None,
) -> np.ndarray:
    """Linear ramp from ``start`` to ``end``, then hold at ``end``.

    Same shape as ``_tau_schedule``, but linear rather than geometric so it can
    start at exactly 0.0 or end at exactly 1.0 -- both routine for the loss
    coefficients this schedules (``theta_entropy_coef``/``topic_usage_coef`` start
    at 0, ``tm_likelihood_coef`` can end at 1), where a log-space ramp would be
    undefined.
    """
    if start is None or n_epochs < 2:
        return np.full(max(n_epochs, 1), end)
    anneal_epochs = min(anneal_epochs or n_epochs, n_epochs)
    ramp = np.linspace(start, end, max(anneal_epochs, 2))
    if anneal_epochs >= n_epochs:
        return ramp[:n_epochs]
    return np.concatenate([ramp, np.full(n_epochs - anneal_epochs, end)])


def _loss_keys(criterion: CompositeLoss) -> list[str]:
    if not criterion.return_individual_compotents:
        return ["total_loss"]
    return [
        "total_loss",
        "reconstruction_loss",
        "reconstruction_accuracy",
        "kl_divergence",
        "tm_loss",
        "label_loss",
        "label_accuracy",
        "topic_l1",
        "theta_entropy",
        "topic_usage_entropy",
        "topic_decorrelation",
        "phi_l2",
        "predicted_witness_rate",
    ]
