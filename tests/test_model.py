import pytest
import torch

from airrtm.models import model_factory

from conftest import MAX_LENGTH, N_TOPICS_SIGNAL, make_batch, make_model_config


def test_forward_shapes(model, alphabet_length):
    sequences, repertoire_ids, _ = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    n_sequences = sequences.shape[0]
    assert output["kl_divergences"].shape == (n_sequences,)
    assert output["tm_log_scores"].shape == (n_sequences,)
    assert output["label_likelihoods"].shape == (n_sequences,)
    assert output["seq_topic_logits"].shape == (n_sequences, model.n_topics)
    assert output["log_topic_proportions"].shape == (n_sequences, model.n_topics)
    assert output["decoded_sequences"].shape == (
        n_sequences,
        MAX_LENGTH,
        alphabet_length + 1,
    )


def test_topic_proportions_are_a_distribution(model, alphabet_length):
    sequences, repertoire_ids, _ = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    theta = torch.exp(output["log_topic_proportions"])
    assert torch.allclose(theta.sum(dim=1), torch.ones(sequences.shape[0]), atol=1e-5)


def test_theta_is_constant_within_a_repertoire(model, alphabet_length):
    sequences, repertoire_ids, _ = make_batch(
        n_repertoires=3, n_per_repertoire=6, alphabet_length=alphabet_length
    )
    output = model(repertoire_ids, sequences)
    theta = output["log_topic_proportions"]
    for repertoire in repertoire_ids.unique():
        rows = theta[repertoire_ids == repertoire]
        assert torch.allclose(rows, rows[0].expand_as(rows), atol=1e-6)


def test_decoder_returns_logits_not_probabilities(model, alphabet_length):
    """Regression: the decoder used to softmax its output, and the reconstruction
    loss then applied log_softmax on top of that, flattening the gradients."""
    sequences, repertoire_ids, _ = make_batch(alphabet_length=alphabet_length)
    decoded = model(repertoire_ids, sequences)["decoded_sequences"]
    sums = decoded.exp().sum(dim=2)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
    assert (decoded < 0).any(), "logits should not all be non-negative"


def test_scoring_matches_the_trained_quantity(model, alphabet_length):
    """Regression: predict_* used raw logits while forward used sigmoid, so the score
    used at evaluation time was not the quantity the model was trained on."""
    sequences, repertoire_ids, _ = make_batch(alphabet_length=alphabet_length)
    model.eval()
    with torch.no_grad():
        probabilities = model.predict_topic_probabilities(sequences)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        label_logits = model.predict_label_logits(sequences)
        expected = model.repertoire_label_prediction_layer(
            probabilities[:, :N_TOPICS_SIGNAL]
        ).flatten()
        assert torch.allclose(label_logits, expected, atol=1e-6)

        intensity = model.predict_signal_intensity(sequences)
        assert torch.allclose(intensity, torch.sigmoid(label_logits), atol=1e-6)


def test_amortized_theta_generalises_to_unseen_repertoires(alphabet_length):
    """The point of theta_mode='amortized': a repertoire index the model never saw."""
    model = model_factory(**make_model_config(2, alphabet_length, theta_mode="amortized"))
    model.eval()
    unseen = torch.randint(0, alphabet_length + 1, (32, MAX_LENGTH))
    with torch.no_grad():
        theta = model.infer_repertoire_topic_proportions(unseen)
    assert theta.shape == (model.n_topics,)
    assert theta.sum() == pytest.approx(1.0, abs=1e-5)


def test_free_theta_refuses_unseen_repertoires(alphabet_length):
    model = model_factory(**make_model_config(2, alphabet_length, theta_mode="free"))
    with pytest.raises(ValueError, match="unseen repertoire"):
        model.infer_repertoire_topic_proportions(
            torch.randint(0, alphabet_length + 1, (8, MAX_LENGTH))
        )


def test_amortized_theta_is_permutation_invariant(alphabet_length):
    model = model_factory(**make_model_config(2, alphabet_length))
    model.eval()
    sequences = torch.randint(0, alphabet_length + 1, (24, MAX_LENGTH))
    with torch.no_grad():
        a = model.infer_repertoire_topic_proportions(sequences)
        b = model.infer_repertoire_topic_proportions(sequences[torch.randperm(24)])
    assert torch.allclose(a, b, atol=1e-5)


def test_vj_genes_are_used(alphabet_length):
    model = model_factory(
        **make_model_config(2, alphabet_length, n_v_genes=5, n_j_genes=3)
    )
    model.eval()
    sequences = torch.randint(0, alphabet_length + 1, (16, MAX_LENGTH))
    v_ids = torch.zeros(16, dtype=torch.long)
    j_ids = torch.zeros(16, dtype=torch.long)
    with torch.no_grad():
        first = model.predict_topic_probabilities(sequences, v_ids, j_ids)
        second = model.predict_topic_probabilities(
            sequences, torch.full_like(v_ids, 4), torch.full_like(j_ids, 2)
        )
    assert not torch.allclose(first, second)

    with pytest.raises(ValueError, match="V/J genes"):
        model.predict_topic_probabilities(sequences)


def test_per_sequence_posterior_width(model, alphabet_length):
    """Regression: sigma used to be one global scalar, so the KL term reduced to a
    norm penalty on the mean and generation was isotropic."""
    sequences = torch.randint(0, alphabet_length + 1, (16, MAX_LENGTH))
    _, log_sigma = model.sequence_to_latent_distribution(sequences)
    assert log_sigma.shape == (16, model.latent_dim)
    assert log_sigma.std() > 0


def test_no_dead_parameters(model, alphabet_length):
    """Every parameter must receive a gradient; z_batch_norm used to be built and
    never called."""
    sequences, repertoire_ids, labels = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    total = (
        output["tm_log_scores"].sum()
        + output["label_likelihoods"].sum()
        + output["decoded_sequences"].sum()
        + output["kl_divergences"].sum()
    )
    total.backward()
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not parameter.grad.any()
    ]
    assert unused == [], f"parameters received no gradient: {unused}"


def test_signal_topic_weights_are_l1_normalised(alphabet_length):
    model = model_factory(**make_model_config(6, alphabet_length, theta_mode="free"))
    labels = torch.tensor([1, 1, 1, 0, 0, 0])
    weights = model._compute_signal_topic_weights(labels, label_of_interest=1)
    assert weights.shape == (N_TOPICS_SIGNAL,)
    assert weights.abs().sum().item() == pytest.approx(1.0, abs=1e-5)


def test_factory_rejects_unknown_config_keys(alphabet_length):
    config = make_model_config(2, alphabet_length)
    config["airrtm_params"]["not_a_real_parameter"] = 1
    with pytest.raises(ValueError, match="Unknown keys"):
        model_factory(**config)


def test_disjoint_theta_context(alphabet_length):
    """Theta must come from the context sample, not from the scored sequences."""
    model = model_factory(**make_model_config(4, alphabet_length))
    model.eval()
    scored, repertoire_ids, _ = make_batch(
        n_repertoires=2, n_per_repertoire=6, alphabet_length=alphabet_length
    )
    context = torch.randint(0, alphabet_length + 1, (2 * 8, MAX_LENGTH))
    context_ids = torch.arange(2).repeat_interleave(8)

    with torch.no_grad():
        without = model(repertoire_ids, scored)["log_topic_proportions"]
        with_context = model(
            repertoire_ids,
            scored,
            theta_sequences=context,
            theta_repertoire_ids=context_ids,
        )["log_topic_proportions"]

    assert not torch.allclose(without, with_context, atol=1e-4)
    # Still one distribution per repertoire, constant within it.
    for repertoire in repertoire_ids.unique():
        rows = with_context[repertoire_ids == repertoire]
        assert torch.allclose(rows, rows[0].expand_as(rows), atol=1e-6)
    assert torch.allclose(
        with_context.exp().sum(dim=1), torch.ones(scored.shape[0]), atol=1e-5
    )


def test_theta_context_independent_of_scored_sequences(alphabet_length):
    """The self-reference this fixes: changing the scored batch must not move Theta."""
    model = model_factory(**make_model_config(4, alphabet_length))
    model.eval()
    repertoire_ids = torch.arange(2).repeat_interleave(6)
    context = torch.randint(0, alphabet_length + 1, (2 * 8, MAX_LENGTH))
    context_ids = torch.arange(2).repeat_interleave(8)

    def theta(scored):
        with torch.no_grad():
            return model(
                repertoire_ids,
                scored,
                theta_sequences=context,
                theta_repertoire_ids=context_ids,
            )["log_topic_proportions"]

    a = theta(torch.randint(0, alphabet_length + 1, (12, MAX_LENGTH)))
    b = theta(torch.randint(0, alphabet_length + 1, (12, MAX_LENGTH)))
    assert torch.allclose(a, b, atol=1e-6)


def test_theta_context_must_cover_every_scored_repertoire(alphabet_length):
    model = model_factory(**make_model_config(4, alphabet_length))
    scored = torch.randint(0, alphabet_length + 1, (12, MAX_LENGTH))
    with pytest.raises(ValueError, match="Theta context"):
        model(
            torch.arange(2).repeat_interleave(6),
            scored,
            theta_sequences=torch.randint(0, alphabet_length + 1, (8, MAX_LENGTH)),
            theta_repertoire_ids=torch.zeros(8, dtype=torch.long),
        )


def _attention_config(alphabet_length, n_repertoires=4):
    config = make_model_config(n_repertoires, alphabet_length)
    config["airrtm_params"]["theta_pooling"] = "attention"
    config["airrtm_params"]["attention_hidden_dim"] = 16
    return config


def test_attention_pooling_builds_and_produces_a_distribution(alphabet_length):
    model = model_factory(**_attention_config(alphabet_length))
    sequences, repertoire_ids, _ = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    theta = output["log_topic_proportions"].exp()
    assert torch.allclose(theta.sum(dim=1), torch.ones(sequences.shape[0]), atol=1e-5)
    for repertoire in repertoire_ids.unique():
        rows = theta[repertoire_ids == repertoire]
        assert torch.allclose(rows, rows[0].expand_as(rows), atol=1e-6)


def test_attention_pooling_is_permutation_invariant(alphabet_length):
    model = model_factory(**_attention_config(alphabet_length, 2))
    model.eval()
    sequences = torch.randint(0, alphabet_length + 1, (24, MAX_LENGTH))
    with torch.no_grad():
        a = model.infer_repertoire_topic_proportions(sequences)
        b = model.infer_repertoire_topic_proportions(sequences[torch.randperm(24)])
    assert torch.allclose(a, b, atol=1e-5)


def test_attention_can_represent_a_rare_subset(alphabet_length):
    """The property mean pooling lacks: sensitivity to a few sequences among many.

    Mean pooling dilutes one distinctive sequence by 1/n. Attention can put most of
    its mass on it, so the repertoire representation shifts appreciably.
    """
    torch.manual_seed(0)
    from airrtm.models.airrtm_model import TopicAttentionPooling, _grouped_mean

    n, dim, topics = 512, 8, 4
    pooling = TopicAttentionPooling(input_dim=dim, n_topics=topics, hidden_dim=16)
    # Make the attention head strongly prefer large first coordinates.
    with torch.no_grad():
        pooling.attend.weight.zero_(); pooling.attend.weight[:, 0] = 3.0
        pooling.attend.bias.zero_()
        pooling.gate.weight.zero_(); pooling.gate.bias.fill_(3.0)
        pooling.score.weight.fill_(3.0); pooling.score.bias.zero_()

    background = torch.zeros(n, dim)
    inverse = torch.zeros(n, dtype=torch.long)
    rare = background.clone()
    rare[0, 0] = 5.0  # one distinctive sequence out of 512

    attention_shift = (
        pooling(rare, inverse, 1) - pooling(background, inverse, 1)
    ).abs().max()
    mean_shift = (
        _grouped_mean(pooling.value(rare), inverse, 1)
        - _grouped_mean(pooling.value(background), inverse, 1)
    ).abs().max()
    assert attention_shift > 10 * mean_shift


def test_grouped_softmax_normalises_within_groups():
    from airrtm.models.airrtm_model import _grouped_softmax

    scores = torch.randn(9, 3)
    inverse = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    attention = _grouped_softmax(scores, inverse, 3)
    for group in range(3):
        assert torch.allclose(
            attention[inverse == group].sum(dim=0), torch.ones(3), atol=1e-5
        )


def test_attention_uses_clonal_abundance(alphabet_length):
    """An expanded clone should draw attention mass in proportion to its size."""
    from airrtm.models.airrtm_model import TopicAttentionPooling

    torch.manual_seed(0)
    pooling = TopicAttentionPooling(input_dim=6, n_topics=3, hidden_dim=8)
    h = torch.randn(64, 6)
    inverse = torch.zeros(64, dtype=torch.long)
    weights = torch.ones(64)
    weights[0] = 1000.0
    flat = pooling(h, inverse, 1)
    weighted = pooling(h, inverse, 1, torch.log(weights))
    assert not torch.allclose(flat, weighted, atol=1e-4)


def test_factory_rejects_bad_theta_pooling(alphabet_length):
    config = make_model_config(2, alphabet_length)
    config["airrtm_params"]["theta_pooling"] = "nonsense"
    with pytest.raises(ValueError, match="theta_pooling"):
        model_factory(**config)


def test_label_input_repertoire_gives_one_prediction_per_repertoire(alphabet_length):
    config = _attention_config(alphabet_length)
    config["airrtm_params"]["label_input"] = "repertoire"
    model = model_factory(**config)
    sequences, repertoire_ids, _ = make_batch(
        n_repertoires=3, n_per_repertoire=6, alphabet_length=alphabet_length
    )
    logits = model(repertoire_ids, sequences)["label_likelihoods"]
    for repertoire in repertoire_ids.unique():
        rows = logits[repertoire_ids == repertoire]
        assert torch.allclose(rows, rows[0].expand_as(rows), atol=1e-6)
    # Different repertoires must not collapse to the same prediction.
    distinct = torch.stack([logits[repertoire_ids == r][0] for r in repertoire_ids.unique()])
    assert distinct.std() > 0


def test_label_input_repertoire_rejects_free_theta(alphabet_length):
    """That combination is the v1 memorisation shortcut."""
    config = make_model_config(2, alphabet_length)
    config["airrtm_params"]["theta_mode"] = "free"
    config["airrtm_params"]["label_input"] = "repertoire"
    with pytest.raises(ValueError, match="v1 shortcut"):
        model_factory(**config)
