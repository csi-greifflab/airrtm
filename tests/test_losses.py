import math

import pytest
import torch

from airrtm.losses import CompositeLoss
from airrtm.losses.composite_loss import _Grouping, _opposite_class_partners
from airrtm.types import AIRRTM_ModelOutput

from conftest import make_batch, make_target


def _fake_output(
    n_sequences: int,
    n_topics: int,
    max_length: int,
    alphabet_length: int,
    topic_logits: torch.Tensor | None = None,
    log_theta: torch.Tensor | None = None,
    raw_tm_score: bool = False,
) -> AIRRTM_ModelOutput:
    """``raw_tm_score=True`` mirrors ``AIRRTM_Model(theta_normalization="none")``:
    ``tm_log_scores`` is a plain ``(theta * phi).sum(1)`` dot product (and
    ``log_theta`` need not be a log-softmax row at all), instead of the default
    batch-softmax family's ``logsumexp(log_theta + phi)``."""
    seq_topic_logits = (
        torch.randn(n_sequences, n_topics, requires_grad=True)
        if topic_logits is None
        else topic_logits
    )
    if log_theta is None:
        log_theta = torch.log_softmax(torch.zeros(n_sequences, n_topics), dim=1)
    tm_log_scores = (
        (log_theta * seq_topic_logits).sum(dim=1)
        if raw_tm_score
        else torch.logsumexp(log_theta + seq_topic_logits, dim=1)
    )
    return AIRRTM_ModelOutput(
        kl_divergences=torch.rand(n_sequences),
        tm_likelihoods=torch.rand(n_sequences),
        tm_log_scores=tm_log_scores,
        seq_topic_logits=seq_topic_logits,
        seq_topic_probabilities=torch.sigmoid(seq_topic_logits),
        log_topic_proportions=log_theta,
        label_likelihoods=torch.randn(n_sequences),
        decoded_sequences=torch.randn(n_sequences, max_length, alphabet_length + 1),
    )


def _fake_output_no_topic_model(
    n_sequences: int, max_length: int, alphabet_length: int
) -> AIRRTM_ModelOutput:
    """What AIRRTM_Model.forward() returns when built with use_topic_model=False."""
    return AIRRTM_ModelOutput(
        kl_divergences=torch.rand(n_sequences),
        tm_likelihoods=None,
        tm_log_scores=None,
        seq_topic_logits=None,
        seq_topic_probabilities=None,
        log_topic_proportions=torch.log_softmax(torch.zeros(n_sequences, 7), dim=1),
        label_likelihoods=torch.randn(n_sequences),
        decoded_sequences=torch.randn(n_sequences, max_length, alphabet_length + 1),
    )


def test_no_topic_model_works_with_coefficients_zeroed(alphabet_length):
    """use_topic_model=False: phi is None, but tm/topic_l1/decorrelation are zeroed too."""
    criterion = CompositeLoss(
        vae_coef=0.5,
        tm_likelihood_coef=0.0,
        reconstruction_loss_coef=0.5,
        topic_l1_coef=0.0,
        theta_entropy_coef=0.1,
        topic_decorrelation_coef=0.0,
        n_sequences_per_repertoire=5,
        pad_value=alphabet_length,
    )
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = _fake_output_no_topic_model(
        sequences.shape[0], sequences.shape[1], alphabet_length
    )
    losses = criterion(output, make_target(sequences, repertoire_ids, labels), tau=1.0)
    assert torch.isfinite(losses["total_loss"])
    assert losses["tm_loss"] == 0.0
    assert losses["topic_l1"] == 0.0
    assert losses["topic_decorrelation"] == 0.0


def test_no_topic_model_rejects_nonzero_tm_coef(criterion, alphabet_length):
    """The default `criterion` fixture has tm_likelihood_coef=0.5 -- must not silently
    ignore phi being absent."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = _fake_output_no_topic_model(
        sequences.shape[0], sequences.shape[1], alphabet_length
    )
    with pytest.raises(ValueError, match="use_topic_model=False"):
        criterion(output, make_target(sequences, repertoire_ids, labels), tau=1.0)


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
    # Regularisers live under the (1 - vae_coef) branch alongside tm/label, so the
    # term's raw value is scaled by (1 - vae_coef) = 0.5 here, not added at full weight.
    assert torch.isclose(
        difference, (1 - base["vae_coef"]) * with_term["topic_decorrelation"], atol=1e-5
    )


def test_vae_coef_one_zeroes_every_topic_model_term(alphabet_length):
    """vae_coef=1.0 (a 'pure VAE' phase, e.g. via vae_coef_start in train_model) must
    stop *every* topic-model-reading term -- tm/label AND all four regularisers --
    not just tm/label. Otherwise a warm start meant to give the VAE branch an
    unopposed window would still leak gradient onto Theta through the regularisers."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    target = make_target(sequences, repertoire_ids, labels)
    output = _fake_output(sequences.shape[0], 7, sequences.shape[1], alphabet_length)

    pure_vae = CompositeLoss(
        vae_coef=1.0,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=5,
        pad_value=alphabet_length,
        topic_l1_coef=1.0,
        theta_entropy_coef=1.0,
        topic_usage_coef=1.0,
        topic_decorrelation_coef=1.0,
        phi_l2_coef=1.0,
    )(output, target, tau=1.0)
    vae_only = CompositeLoss(
        vae_coef=1.0,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=5,
        pad_value=alphabet_length,
    )(output, target, tau=1.0)
    assert torch.isclose(pure_vae["total_loss"], vae_only["total_loss"], atol=1e-5)


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


def test_tm_negative_mode_rejects_invalid_value(alphabet_length):
    with pytest.raises(ValueError, match="tm_negative_mode"):
        CompositeLoss(
            vae_coef=0.5,
            tm_likelihood_coef=0.5,
            reconstruction_loss_coef=0.5,
            n_sequences_per_repertoire=5,
            pad_value=alphabet_length,
            tm_negative_mode="nonsense",
        )


def test_opposite_class_partners_pairs_by_label():
    """[0, 1, 0, 1] -> repertoire 0/2 (class 0) paired with 1/3 (class 1)."""
    labels_R = torch.tensor([0, 1, 0, 1])
    partner_R = _opposite_class_partners(labels_R)
    assert labels_R[partner_R[0]] == 1
    assert labels_R[partner_R[1]] == 0
    assert labels_R[partner_R[2]] == 1
    assert labels_R[partner_R[3]] == 0


def test_opposite_class_partners_cycles_uneven_classes():
    """3 of class 0, 1 of class 1: the lone class-1 repertoire must partner all three."""
    labels_R = torch.tensor([0, 0, 0, 1])
    partner_R = _opposite_class_partners(labels_R)
    assert (partner_R[:3] == 3).all()
    assert labels_R[partner_R[3]] == 0


def test_opposite_class_partners_raises_on_single_class():
    with pytest.raises(ValueError, match="single-class"):
        _opposite_class_partners(torch.tensor([0, 0, 0, 0]))


def test_opposite_class_pair_finite_and_has_gradient(alphabet_length):
    n_per_repertoire, n_repertoires, n_topics = 5, 4, 3
    sequences, repertoire_ids, labels = make_batch(
        n_repertoires=n_repertoires,
        n_per_repertoire=n_per_repertoire,
        alphabet_length=alphabet_length,
    )
    target = make_target(sequences, repertoire_ids, labels)
    n_sequences = sequences.shape[0]
    logits = torch.randn(n_sequences, n_topics, requires_grad=True)
    criterion = CompositeLoss(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=n_per_repertoire,
        pad_value=alphabet_length,
        tm_negative_mode="opposite_class_pair",
    )
    output = _fake_output(
        n_sequences, n_topics, sequences.shape[1], alphabet_length, topic_logits=logits
    )
    losses = criterion(output, target, tau=1.0)
    assert torch.isfinite(losses["tm_loss"])
    losses["total_loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_opposite_class_pair_ignores_non_partner_repertoire(alphabet_length):
    """A repertoire's TM likelihood must depend only on itself and its assigned
    opposite-class partner, not on an unrelated third repertoire in the same batch
    -- the property that distinguishes this mode from the default "batch" one,
    where every repertoire's normaliser reads every sequence in the batch."""
    n_per_repertoire, n_repertoires, n_topics = 5, 4, 2
    sequences, repertoire_ids, labels = make_batch(
        n_repertoires=n_repertoires,
        n_per_repertoire=n_per_repertoire,
        alphabet_length=alphabet_length,
    )
    target = make_target(sequences, repertoire_ids, labels)
    n_sequences = sequences.shape[0]
    # labels alternate [0, 1, 0, 1] (make_batch) -> pairs (0, 1) and (2, 3).
    grouping = _Grouping(repertoire_ids)
    partner_R = _opposite_class_partners(labels[grouping.first_index_R])

    log_theta = torch.log_softmax(torch.randn(n_repertoires, n_topics), dim=1)
    log_theta_ST = log_theta.repeat_interleave(n_per_repertoire, dim=0)

    criterion = CompositeLoss(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=n_per_repertoire,
        pad_value=alphabet_length,
        tm_negative_mode="opposite_class_pair",
    )

    def own_repertoire_0_likelihood(rep23_logits: torch.Tensor) -> torch.Tensor:
        logits = torch.cat(
            [torch.zeros(2 * n_per_repertoire, n_topics), rep23_logits], dim=0
        )
        output = _fake_output(
            n_sequences,
            n_topics,
            sequences.shape[1],
            alphabet_length,
            topic_logits=logits,
            log_theta=log_theta_ST,
        )
        log_likelihood_S = criterion._tm_log_likelihood_per_sequence(
            output, grouping, partner_R
        )
        return log_likelihood_S[:n_per_repertoire]  # repertoire 0's own sequences

    baseline = own_repertoire_0_likelihood(torch.zeros(2 * n_per_repertoire, n_topics))
    perturbed = own_repertoire_0_likelihood(torch.full((2 * n_per_repertoire, n_topics), 20.0))
    assert torch.allclose(baseline, perturbed, atol=1e-6)


def test_tm_likelihood_family_rejects_invalid_value(alphabet_length):
    with pytest.raises(ValueError, match="tm_likelihood_family"):
        CompositeLoss(
            vae_coef=0.5,
            tm_likelihood_coef=0.5,
            reconstruction_loss_coef=0.5,
            n_sequences_per_repertoire=5,
            pad_value=alphabet_length,
            tm_likelihood_family="nonsense",
        )


def test_theta_is_normalized_false_rejects_nonzero_entropy_coefs(alphabet_length):
    """theta_entropy_coef/topic_usage_coef assume a probability simplex, which a
    raw (theta_is_normalized=False) Theta does not have."""
    with pytest.raises(ValueError, match="theta_is_normalized"):
        CompositeLoss(
            vae_coef=0.5,
            tm_likelihood_coef=0.5,
            reconstruction_loss_coef=0.5,
            n_sequences_per_repertoire=5,
            pad_value=alphabet_length,
            theta_entropy_coef=0.5,
            theta_is_normalized=False,
        )


def _raw_bce_criterion(alphabet_length, **overrides) -> CompositeLoss:
    kwargs = dict(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        n_sequences_per_repertoire=5,
        pad_value=alphabet_length,
        tm_likelihood_family="raw_bce",
        theta_is_normalized=False,
    )
    kwargs.update(overrides)
    return CompositeLoss(**kwargs)


def test_raw_bce_finite_and_has_gradient(alphabet_length):
    n_per_repertoire, n_repertoires, n_topics = 5, 4, 3
    sequences, repertoire_ids, labels = make_batch(
        n_repertoires=n_repertoires,
        n_per_repertoire=n_per_repertoire,
        alphabet_length=alphabet_length,
    )
    target = make_target(sequences, repertoire_ids, labels)
    n_sequences = sequences.shape[0]
    logits = torch.randn(n_sequences, n_topics, requires_grad=True)
    # Raw, unconstrained "theta" -- any real values, not a log-softmax row.
    log_theta = torch.randn(n_sequences, n_topics)

    criterion = _raw_bce_criterion(alphabet_length)
    output = _fake_output(
        n_sequences,
        n_topics,
        sequences.shape[1],
        alphabet_length,
        topic_logits=logits,
        log_theta=log_theta,
        raw_tm_score=True,
    )
    losses = criterion(output, target, tau=1.0)
    assert torch.isfinite(losses["tm_loss"])
    losses["total_loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_raw_bce_rewards_alignment_with_own_repertoire(alphabet_length):
    """Aligned: phi matches own theta and mismatches the opposite-class partner's
    -- both BCE targets are already satisfied, so the loss should be near its
    floor. Misaligned: the reverse, so the loss should be much higher."""
    n_per_repertoire, n_topics = 5, 2
    sequences, repertoire_ids, labels = make_batch(
        n_repertoires=2, n_per_repertoire=n_per_repertoire, alphabet_length=alphabet_length
    )
    target = make_target(sequences, repertoire_ids, labels)
    n_sequences = sequences.shape[0]

    # Repertoire 0 (label 0) uses raw theta [1, 0]; repertoire 1 (label 1) uses [0, 1]
    # -- each repertoire's own theta is the other's opposite-class partner's theta.
    theta_R = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    log_theta = theta_R.repeat_interleave(n_per_repertoire, dim=0)

    aligned = torch.zeros(n_sequences, n_topics)
    aligned[:n_per_repertoire] = torch.tensor([1.0, 0.0])  # matches repertoire 0's theta
    aligned[n_per_repertoire:] = torch.tensor([0.0, 1.0])  # matches repertoire 1's theta

    misaligned = torch.zeros(n_sequences, n_topics)
    misaligned[:n_per_repertoire] = torch.tensor([0.0, 1.0])  # matches the partner instead
    misaligned[n_per_repertoire:] = torch.tensor([1.0, 0.0])

    criterion = _raw_bce_criterion(alphabet_length)

    def tm(logits):
        output = _fake_output(
            n_sequences,
            n_topics,
            sequences.shape[1],
            alphabet_length,
            topic_logits=logits,
            log_theta=log_theta,
            raw_tm_score=True,
        )
        return criterion(output, target, tau=1.0)["tm_loss"]

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


# ------------------------------------------- abundance-weighted losses (experiment 1)


def _weighted_criterion(alphabet_length, **overrides):
    base = dict(
        vae_coef=0.5,
        tm_likelihood_coef=0.5,
        reconstruction_loss_coef=0.5,
        pad_value=alphabet_length,
    )
    base.update(overrides)
    return CompositeLoss(**base)


def test_normalised_abundance_has_global_mean_one_and_per_repertoire_scale():
    from airrtm.losses.composite_loss import _Grouping, _normalised_abundance

    repertoire_ids = torch.tensor([0, 0, 0, 1, 1])
    grouping = _Grouping(repertoire_ids)
    # Repertoire 0 holds one huge clone; repertoire 1 is flat.
    weights = torch.tensor([1.0, 1.0, 100.0, 3.0, 3.0])
    normalised = _normalised_abundance(weights, grouping)

    # Exactly mean 1 overall, for any group sizes -- what lets callers pass it
    # straight into a 'mean'-reduction loss.
    assert torch.allclose(normalised.mean(), torch.tensor(1.0), atol=1e-6)
    # And each repertoire's weights sum to its own sequence count, so the huge
    # clone cannot drown the other repertoire.
    assert torch.allclose(normalised[:3].sum(), torch.tensor(3.0), atol=1e-5)
    assert torch.allclose(normalised[3:].sum(), torch.tensor(2.0), atol=1e-5)
    # A flat repertoire is left untouched.
    assert torch.allclose(normalised[3:], torch.ones(2), atol=1e-6)


def test_normalised_abundance_is_none_without_weights():
    from airrtm.losses.composite_loss import _Grouping, _normalised_abundance

    grouping = _Grouping(torch.tensor([0, 0, 1, 1]))
    assert _normalised_abundance(None, grouping) is None


def test_abundance_weighted_losses_reduce_to_unweighted_when_weights_are_equal(
    model, alphabet_length
):
    """The weighted path is a drop-in: equal abundances must reproduce the plain mean."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    plain_target = make_target(sequences, repertoire_ids, labels)
    weighted_target = dict(plain_target)
    weighted_target["weights"] = torch.full((sequences.shape[0],), 7.0)

    unweighted = _weighted_criterion(alphabet_length)(output, plain_target, tau=1.0)
    weighted = _weighted_criterion(alphabet_length, abundance_weighted_losses=True)(
        output, weighted_target, tau=1.0
    )

    for key in ("tm_loss", "reconstruction_loss", "total_loss"):
        assert torch.allclose(unweighted[key], weighted[key], atol=1e-5), key


def test_abundance_weighted_losses_change_the_loss_when_weights_differ(
    model, alphabet_length
):
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    target = dict(make_target(sequences, repertoire_ids, labels))
    target["weights"] = torch.arange(1, sequences.shape[0] + 1).float()

    criterion = _weighted_criterion(alphabet_length, abundance_weighted_losses=True)
    weighted = criterion(output, target, tau=1.0)
    unweighted = _weighted_criterion(alphabet_length)(output, target, tau=1.0)

    assert not torch.allclose(weighted["tm_loss"], unweighted["tm_loss"], atol=1e-6)
    assert not torch.allclose(
        weighted["reconstruction_loss"], unweighted["reconstruction_loss"], atol=1e-6
    )


def test_abundance_weighted_losses_falls_back_when_the_target_has_no_weights(
    model, alphabet_length
):
    """A dataset without duplicate_count must not crash the weighted path."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    target = make_target(sequences, repertoire_ids, labels)

    weighted = _weighted_criterion(alphabet_length, abundance_weighted_losses=True)(
        output, target, tau=1.0
    )
    unweighted = _weighted_criterion(alphabet_length)(output, target, tau=1.0)
    assert torch.allclose(weighted["total_loss"], unweighted["total_loss"], atol=1e-6)


# --------------------------------------------- normalize_loss_scales (loss rescaling)


def test_normalize_loss_scales_is_off_by_default_and_changes_nothing(model, alphabet_length):
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    target = make_target(sequences, repertoire_ids, labels)
    base = dict(
        vae_coef=0.5, tm_likelihood_coef=0.5, reconstruction_loss_coef=0.5,
        pad_value=alphabet_length,
    )
    default = CompositeLoss(**base)(output, target, tau=1.0)
    explicit_off = CompositeLoss(**base, normalize_loss_scales=False)(output, target, tau=1.0)
    assert torch.allclose(default["total_loss"], explicit_off["total_loss"], atol=1e-7)


def test_normalize_loss_scales_leaves_reported_components_unnormalised(
    model, alphabet_length
):
    """Only total_loss changes, so `tm`/`rec`/`label` stay comparable to old runs."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    target = make_target(sequences, repertoire_ids, labels)
    base = dict(
        vae_coef=0.5, tm_likelihood_coef=0.5, reconstruction_loss_coef=0.5,
        pad_value=alphabet_length,
    )
    plain = CompositeLoss(**base)(output, target, tau=1.0)
    scaled = CompositeLoss(**base, normalize_loss_scales=True)(output, target, tau=1.0)

    for key in ("tm_loss", "label_loss", "reconstruction_loss", "kl_divergence"):
        assert torch.allclose(plain[key], scaled[key], atol=1e-6), key
    assert not torch.allclose(plain["total_loss"], scaled["total_loss"], atol=1e-6)


def test_normalize_loss_scales_divides_each_term_by_its_chance_value(
    model, alphabet_length
):
    """Reconstruct total_loss by hand from the documented chance values."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    target = make_target(sequences, repertoire_ids, labels)
    criterion = CompositeLoss(
        vae_coef=0.25, tm_likelihood_coef=0.6, reconstruction_loss_coef=0.8,
        pad_value=alphabet_length, normalize_loss_scales=True,
    )
    out = criterion(output, target, tau=1.0)

    n_sequences = sequences.shape[0]
    # Chance: reconstruction is uniform over the alphabet (pad excluded), tm is a
    # softmax over the whole batch, label is a balanced BCE.
    rec_scale = math.log(alphabet_length)
    tm_scale = math.log(n_sequences)
    label_scale = 0.5 * math.log(2.0)

    expected_vae = 0.8 * out["reconstruction_loss"] / rec_scale + 0.2 * out["kl_divergence"]
    expected_non_vae = (
        0.6 * out["tm_loss"] / tm_scale + 0.4 * out["label_loss"] / label_scale
    )
    expected = 0.25 * expected_vae + 0.75 * expected_non_vae
    assert torch.allclose(out["total_loss"], expected, atol=1e-5)


def test_normalize_loss_scales_puts_the_terms_on_a_comparable_footing(
    model, alphabet_length
):
    """At chance, every normalised likelihood term should sit near 1.0.

    This is the whole point: unnormalised, an untrained model reports tm ~= 10.4,
    reconstruction ~= 3.0 and label ~= 0.35, so a coefficient does not mean what it
    looks like.
    """
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    target = make_target(sequences, repertoire_ids, labels)
    criterion = CompositeLoss(
        vae_coef=0.5, tm_likelihood_coef=0.5, reconstruction_loss_coef=0.5,
        pad_value=alphabet_length, normalize_loss_scales=True,
    )
    scales = criterion._loss_scales(output, target, _Grouping(repertoire_ids))
    assert scales["reconstruction"] == pytest.approx(math.log(alphabet_length))
    assert scales["tm"] == pytest.approx(math.log(sequences.shape[0]))
    assert scales["label"] == pytest.approx(0.5 * math.log(2.0))
    assert scales["entropy"] == pytest.approx(
        math.log(output["log_topic_proportions"].shape[1])
    )


def test_loss_scale_mode_range_uses_the_reachable_drop_for_tm(model, alphabet_length):
    """tm cannot reach 0: its floor is log(pool / n_repertoires), not 0.

    So its reachable range is log(n_repertoires_in_batch), not log(pool) -- a factor
    of 7.5 at the default 4x8192 batch. Reconstruction and the label can reach 0, so
    their scales are identical in both modes.
    """
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    target = make_target(sequences, repertoire_ids, labels)
    base = dict(
        vae_coef=0.5, tm_likelihood_coef=0.5, reconstruction_loss_coef=0.5,
        pad_value=alphabet_length, normalize_loss_scales=True,
    )
    grouping = _Grouping(repertoire_ids)
    chance = CompositeLoss(**base)._loss_scales(output, target, grouping)
    rng = CompositeLoss(**base, loss_scale_mode="range")._loss_scales(
        output, target, grouping
    )

    n_repertoires = int(repertoire_ids.unique().numel())
    assert rng["tm"] == pytest.approx(math.log(n_repertoires))
    assert chance["tm"] == pytest.approx(math.log(sequences.shape[0]))
    # Only tm differs.
    for key in ("reconstruction", "label", "entropy"):
        assert rng[key] == pytest.approx(chance[key]), key
    # Range mode therefore gives tm a larger gradient, by exactly the pool/group ratio.
    assert chance["tm"] / rng["tm"] == pytest.approx(
        math.log(sequences.shape[0]) / math.log(n_repertoires)
    )


def test_loss_scale_mode_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="loss_scale_mode"):
        CompositeLoss(
            vae_coef=0.5, tm_likelihood_coef=0.5, reconstruction_loss_coef=0.5,
            loss_scale_mode="nonsense",
        )
