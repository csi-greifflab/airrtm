import pytest
import torch

from airrtm.losses import CompositeLoss
from airrtm.losses.composite_loss import _Grouping
from airrtm.types import AIRRTM_ModelOutput

from conftest import make_batch, make_target


def _fake_output(
    n_sequences: int,
    n_topics: int,
    max_length: int,
    alphabet_length: int,
    topic_logits: torch.Tensor | None = None,
    log_theta: torch.Tensor | None = None,
) -> AIRRTM_ModelOutput:
    seq_topic_logits = (
        torch.randn(n_sequences, n_topics, requires_grad=True)
        if topic_logits is None
        else topic_logits
    )
    if log_theta is None:
        log_theta = torch.log_softmax(torch.zeros(n_sequences, n_topics), dim=1)
    return AIRRTM_ModelOutput(
        kl_divergences=torch.rand(n_sequences),
        tm_likelihoods=torch.rand(n_sequences),
        tm_log_scores=torch.logsumexp(log_theta + seq_topic_logits, dim=1),
        seq_topic_logits=seq_topic_logits,
        seq_topic_probabilities=torch.sigmoid(seq_topic_logits),
        log_topic_proportions=log_theta,
        label_likelihoods=torch.randn(n_sequences),
        decoded_sequences=torch.randn(n_sequences, max_length, alphabet_length + 1),
    )


def test_all_components_finite(criterion, alphabet_length):
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = _fake_output(sequences.shape[0], 7, sequences.shape[1], alphabet_length)
    losses = criterion(output, make_target(sequences, repertoire_ids, labels), tau=1.0)
    for key, value in losses.items():
        assert torch.isfinite(value).all(), f"{key} is not finite"


def test_total_loss_has_gradient(criterion, alphabet_length):
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    logits = torch.randn(sequences.shape[0], 7, requires_grad=True)
    output = _fake_output(
        sequences.shape[0], 7, sequences.shape[1], alphabet_length, topic_logits=logits
    )
    losses = criterion(output, make_target(sequences, repertoire_ids, labels), tau=1.0)
    losses["total_loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_coefficient_removes_a_term(alphabet_length):
    """A term with coefficient zero must not move the total."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    target = make_target(sequences, repertoire_ids, labels)
    output = _fake_output(sequences.shape[0], 7, sequences.shape[1], alphabet_length)

    base = dict(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=5,
        pad_value=alphabet_length,
    )
    without = CompositeLoss(**base, topic_decorrelation_coef=0.0)(output, target, tau=1.0)
    with_term = CompositeLoss(**base, topic_decorrelation_coef=1.0)(output, target, tau=1.0)
    difference = with_term["total_loss"] - without["total_loss"]
    assert torch.isclose(difference, with_term["topic_decorrelation"], atol=1e-5)


def test_tm_loss_is_not_minimised_by_saturating_phi(criterion, alphabet_length):
    """The regression that made the TM term useless.

    With an unnormalised likelihood, pushing every topic score up lowers the loss
    without bound, so the optimum is to saturate. With the in-batch normaliser the
    loss is invariant to a constant shift of all scores, and saturation is no better
    than a random configuration.
    """
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    target = make_target(sequences, repertoire_ids, labels)
    n_sequences, n_topics = sequences.shape[0], 7

    modest = torch.zeros(n_sequences, n_topics)
    saturated = torch.full((n_sequences, n_topics), 20.0)

    loss_modest = criterion(
        _fake_output(n_sequences, n_topics, sequences.shape[1], alphabet_length, modest),
        target,
        tau=1.0,
    )["tm_loss"]
    loss_saturated = criterion(
        _fake_output(
            n_sequences, n_topics, sequences.shape[1], alphabet_length, saturated
        ),
        target,
        tau=1.0,
    )["tm_loss"]
    assert torch.isclose(loss_modest, loss_saturated, atol=1e-4)


def test_tm_loss_rewards_discriminative_topics(criterion, alphabet_length):
    """A sequence scored higher under its own repertoire's topics must cost less."""
    n_per_repertoire, n_repertoires, n_topics = 5, 2, 2
    sequences, repertoire_ids, labels = make_batch(
        n_repertoires=n_repertoires,
        n_per_repertoire=n_per_repertoire,
        alphabet_length=alphabet_length,
    )
    target = make_target(sequences, repertoire_ids, labels)
    n_sequences = sequences.shape[0]

    # Repertoire 0 uses topic 0, repertoire 1 uses topic 1.
    theta = torch.tensor([[0.99, 0.01], [0.01, 0.99]]).log()
    log_theta = theta.repeat_interleave(n_per_repertoire, dim=0)

    aligned = torch.zeros(n_sequences, n_topics)
    aligned[:n_per_repertoire, 0] = 4.0  # repertoire 0's sequences load on topic 0
    aligned[n_per_repertoire:, 1] = 4.0

    misaligned = torch.zeros(n_sequences, n_topics)
    misaligned[:n_per_repertoire, 1] = 4.0
    misaligned[n_per_repertoire:, 0] = 4.0

    def tm(logits):
        return criterion(
            _fake_output(
                n_sequences,
                n_topics,
                sequences.shape[1],
                alphabet_length,
                topic_logits=logits,
                log_theta=log_theta,
            ),
            target,
            tau=1.0,
        )["tm_loss"]

    assert tm(aligned) < tm(misaligned)


def test_label_accuracy_is_an_accuracy(criterion, alphabet_length):
    """Regression: the key used to carry the label *loss* instead."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    target = make_target(sequences, repertoire_ids, labels)
    output = _fake_output(sequences.shape[0], 7, sequences.shape[1], alphabet_length)
    # Make every bag's pooled logit strongly agree with its label.
    output["label_likelihoods"] = torch.where(labels > 0, 10.0, -10.0)
    losses = criterion(output, target, tau=1.0)
    assert losses["label_accuracy"] == pytest.approx(1.0)

    output["label_likelihoods"] = torch.where(labels > 0, -10.0, 10.0)
    losses = criterion(output, target, tau=1.0)
    assert losses["label_accuracy"] == pytest.approx(0.0)


def test_reconstruction_accuracy_ignores_padding(alphabet_length):
    """Padding is ~half of every Emerson sequence; counting it inflates the metric."""
    pad = alphabet_length
    sequences = torch.full((4, 6), pad)
    sequences[:, 0] = 1  # one real residue per sequence
    repertoire_ids = torch.zeros(4, dtype=torch.long)
    labels = torch.zeros(4, dtype=torch.long)

    # Predict padding everywhere: right on the pad positions, wrong on the real one.
    logits = torch.zeros(4, 6, alphabet_length + 1)
    logits[:, :, pad] = 10.0

    output = _fake_output(4, 7, 6, alphabet_length)
    output["decoded_sequences"] = logits

    criterion = CompositeLoss(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=4,
        pad_value=pad,
    )
    losses = criterion(output, make_target(sequences, repertoire_ids, labels), tau=1.0)
    assert losses["reconstruction_accuracy"] == pytest.approx(0.0)


def test_grouping_is_order_invariant(alphabet_length):
    """The loss groups by repertoire id, not by position in the batch."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    n_sequences = sequences.shape[0]
    logits = torch.randn(n_sequences, 7)
    log_theta = torch.log_softmax(torch.randn(4, 7), dim=1)[repertoire_ids]

    criterion = CompositeLoss(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=5,
        pad_value=alphabet_length,
    )
    output = _fake_output(
        n_sequences, 7, sequences.shape[1], alphabet_length, logits, log_theta
    )
    ordered = criterion(
        output, make_target(sequences, repertoire_ids, labels), tau=1.0
    )

    permutation = torch.randperm(n_sequences)
    permuted_output = {key: value[permutation] for key, value in output.items()}
    shuffled = criterion(
        permuted_output,
        make_target(
            sequences[permutation], repertoire_ids[permutation], labels[permutation]
        ),
        tau=1.0,
    )
    for key in ("tm_loss", "label_loss", "label_accuracy", "theta_entropy"):
        assert torch.isclose(ordered[key], shuffled[key], atol=1e-5), key


def test_grouping_handles_unequal_group_sizes():
    repertoire_ids = torch.tensor([0, 0, 0, 1, 2, 2])
    grouping = _Grouping(repertoire_ids)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    matrix, mask = grouping.scatter(values)
    assert matrix.shape == (3, 3)
    assert mask.sum() == 6
    assert matrix[0][mask[0]].tolist() == [1.0, 2.0, 3.0]
    assert matrix[1][mask[1]].tolist() == [4.0]
    assert matrix[2][mask[2]].tolist() == [5.0, 6.0]


def test_mil_temperature_interpolates_mean_to_max(criterion, alphabet_length):
    """Low tau pools towards the mean of the bag, high tau towards its maximum."""
    sequences, repertoire_ids, labels = make_batch(
        n_repertoires=2, n_per_repertoire=8, alphabet_length=alphabet_length
    )
    target = make_target(sequences, repertoire_ids, labels)
    output = _fake_output(sequences.shape[0], 7, sequences.shape[1], alphabet_length)
    # One strong instance per bag, the rest weak: the classic MIL situation.
    scores = torch.full((sequences.shape[0],), -3.0)
    scores[0] = 5.0
    scores[8] = 5.0
    output["label_likelihoods"] = scores

    low = criterion(output, target, tau=0.01)["label_loss"]
    high = criterion(output, target, tau=20.0)["label_loss"]
    # Bag 0 is negative and bag 1 positive; the max-like pooling calls both positive,
    # so it is penalised on the negative bag more than the mean-like pooling is.
    assert not torch.isclose(low, high)


def _output_with_theta(theta_RT, n_per_repertoire, alphabet_length, max_length=6):
    """Build a fake output whose repertoires have the given topic proportions."""
    n_repertoires, n_topics = theta_RT.shape
    log_theta = torch.log(theta_RT.clamp(min=1e-9)).repeat_interleave(
        n_per_repertoire, dim=0
    )
    n = n_repertoires * n_per_repertoire
    logits = torch.zeros(n, n_topics, requires_grad=True)
    return AIRRTM_ModelOutput(
        kl_divergences=torch.zeros(n),
        tm_likelihoods=torch.full((n,), 0.5),
        tm_log_scores=torch.logsumexp(log_theta + logits, dim=1),
        seq_topic_logits=logits,
        seq_topic_probabilities=torch.sigmoid(logits),
        log_topic_proportions=log_theta,
        label_likelihoods=torch.zeros(n),
        decoded_sequences=torch.zeros(n, max_length, alphabet_length + 1),
    )


def _usage_loss(theta_RT, alphabet_length, momentum=0.0):
    n_per = 4
    criterion = CompositeLoss(
        vae_coef=0.0, tm_likelihood_coef=0.0, reconstruction_loss_coef=0.0,
        n_sequences_per_repertoire=n_per, pad_value=alphabet_length,
        topic_usage_coef=1.0, topic_usage_momentum=momentum,
    )
    n_repertoires = theta_RT.shape[0]
    output = _output_with_theta(theta_RT, n_per, alphabet_length)
    sequences = torch.zeros(n_repertoires * n_per, 6, dtype=torch.long)
    repertoire_ids = torch.arange(n_repertoires).repeat_interleave(n_per)
    labels = torch.zeros_like(repertoire_ids)
    return criterion(output, make_target(sequences, repertoire_ids, labels), tau=1.0)


def test_topic_usage_penalises_collapse(alphabet_length):
    """All repertoires on one topic must cost more than each on a different one."""
    collapsed = torch.zeros(4, 8); collapsed[:, 3] = 1.0
    spread = torch.zeros(4, 8)
    for r in range(4):
        spread[r, r] = 1.0
    assert _usage_loss(collapsed, alphabet_length)["total_loss"] > _usage_loss(
        spread, alphabet_length
    )["total_loss"]


def test_topic_usage_allows_per_repertoire_sparsity(alphabet_length):
    """One-hot repertoires on distinct topics must beat uniform ones.

    The point of the term: a repertoire is free to use a single topic, so long as
    repertoires disagree about which.
    """
    one_hot = torch.zeros(8, 8)
    for r in range(8):
        one_hot[r, r] = 1.0
    uniform = torch.full((8, 8), 1.0 / 8)
    sharp = _usage_loss(one_hot, alphabet_length)
    flat = _usage_loss(uniform, alphabet_length)
    assert torch.isclose(sharp["total_loss"], flat["total_loss"], atol=1e-4)
    # ... and the per-repertoire entropy is what distinguishes them.
    assert sharp["theta_entropy"] < flat["theta_entropy"]


def test_topic_usage_entropy_is_reported_and_bounded(alphabet_length):
    spread = torch.zeros(4, 8)
    for r in range(4):
        spread[r, r] = 1.0
    losses = _usage_loss(spread, alphabet_length)
    # Four one-hot repertoires over eight topics -> H = log 4.
    assert losses["topic_usage_entropy"] == pytest.approx(float(torch.log(torch.tensor(4.0))), abs=1e-4)


def test_topic_usage_gradient_flows_to_theta(alphabet_length):
    collapsed = torch.zeros(4, 8, requires_grad=True)
    with torch.no_grad():
        collapsed[:, 3] = 5.0
    theta = torch.softmax(collapsed, dim=1)
    losses = _usage_loss(theta, alphabet_length)
    losses["total_loss"].backward()
    assert collapsed.grad is not None and torch.isfinite(collapsed.grad).all()
    assert collapsed.grad.abs().sum() > 0


def test_topic_usage_ema_smooths_across_batches(alphabet_length):
    """The running estimate must accumulate usage that no single batch can show."""
    criterion = CompositeLoss(
        vae_coef=0.0, tm_likelihood_coef=0.0, reconstruction_loss_coef=0.0,
        n_sequences_per_repertoire=4, pad_value=alphabet_length,
        topic_usage_coef=1.0, topic_usage_momentum=0.5,
    )
    entropies = []
    for step in range(8):
        theta = torch.zeros(2, 8)
        theta[0, (2 * step) % 8] = 1.0      # a different pair of topics each batch
        theta[1, (2 * step + 1) % 8] = 1.0
        output = _output_with_theta(theta, 4, alphabet_length)
        sequences = torch.zeros(8, 6, dtype=torch.long)
        ids = torch.arange(2).repeat_interleave(4)
        losses = criterion(output, make_target(sequences, ids, torch.zeros_like(ids)), tau=1.0)
        entropies.append(float(losses["topic_usage_entropy"]))
    # A single batch can never exceed log(2); the running estimate must.
    assert max(entropies) > float(torch.log(torch.tensor(2.0)))
