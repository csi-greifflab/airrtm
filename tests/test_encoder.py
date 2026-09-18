import pytest
import torch

from airrtm.models import model_factory
from airrtm.models.encoder import TransformerCNNEncoder, TransformerEncoder, _masked_mean

from conftest import MAX_LENGTH, make_batch, make_model_config


def test_mean_pooling_output_is_half_of_mean_max(alphabet_length):
    """"mean" is meant as a cheaper alternative to "mean_max" -- same encoder,
    half the output width, and exactly the mean half of what mean_max returns."""
    kwargs = dict(
        max_sequence_length=MAX_LENGTH,
        alphabet_length=alphabet_length,
        attention_dim=32,
        attention_dim_head=32,
        attention_heads=2,
        depth=2,
    )
    sequences, _, _ = make_batch(alphabet_length=alphabet_length)

    torch.manual_seed(0)
    mean_encoder = TransformerEncoder(pooling="mean", **kwargs).eval()
    torch.manual_seed(0)
    mean_max_encoder = TransformerEncoder(pooling="mean_max", **kwargs).eval()

    # .eval() turns off dropout so the two forward passes are deterministic and
    # directly comparable despite running sequentially against one shared RNG.
    with torch.no_grad():
        mean_output = mean_encoder(sequences)
        mean_max_output = mean_max_encoder(sequences)

    assert mean_output.shape == (sequences.shape[0], mean_encoder.get_output_dim())
    assert mean_encoder.get_output_dim() == mean_max_encoder.get_output_dim() // 2
    assert torch.allclose(mean_output, mean_max_output[:, : mean_encoder.get_output_dim()])


def test_attention_pooling_output_shape_and_gradient(alphabet_length):
    kwargs = dict(
        max_sequence_length=MAX_LENGTH,
        alphabet_length=alphabet_length,
        attention_dim=32,
        attention_dim_head=32,
        attention_heads=2,
        depth=2,
    )
    sequences, _, _ = make_batch(alphabet_length=alphabet_length)

    encoder = TransformerEncoder(pooling="attention", **kwargs)
    output = encoder(sequences)

    assert output.shape == (sequences.shape[0], encoder.get_output_dim())
    assert encoder.get_output_dim() == 32

    output.sum().backward()
    assert encoder.attention_pool.score[0].weight.grad is not None
    assert torch.isfinite(encoder.attention_pool.score[0].weight.grad).all()


def test_attention_pooling_differs_from_mean_pooling(alphabet_length):
    """A degenerate check that attention pooling actually weights positions
    unevenly rather than collapsing to a disguised mean: on the same encoder
    output, a random (untrained) attention query should almost surely not
    reproduce the uniform-mean result."""
    kwargs = dict(
        max_sequence_length=MAX_LENGTH,
        alphabet_length=alphabet_length,
        attention_dim=32,
        attention_dim_head=32,
        attention_heads=2,
        depth=2,
    )
    sequences, _, _ = make_batch(alphabet_length=alphabet_length)

    torch.manual_seed(0)
    encoder = TransformerEncoder(pooling="attention", **kwargs).eval()
    with torch.no_grad():
        mask = encoder.padding_mask(sequences)
        x_SPE = encoder.transformer(sequences, mask=mask)
        attention_output = encoder.attention_pool(x_SPE, mask)
        mean_output = _masked_mean(x_SPE, mask)

    assert not torch.allclose(attention_output, mean_output)


def test_mean_pooling_rejected_by_unsupported_value(alphabet_length):
    with pytest.raises(ValueError, match="pooling"):
        TransformerEncoder(
            max_sequence_length=MAX_LENGTH,
            alphabet_length=alphabet_length,
            attention_dim=32,
            attention_dim_head=32,
            attention_heads=2,
            depth=2,
            pooling="nonsense",
        )


def _make_encoder(alphabet_length: int, **cnn_kwargs) -> TransformerCNNEncoder:
    return TransformerCNNEncoder(
        max_sequence_length=MAX_LENGTH,
        alphabet_length=alphabet_length,
        attention_dim=32,
        attention_dim_head=32,
        attention_heads=2,
        depth=2,
        pooling="mean_max",
        **cnn_kwargs,
    )


def test_output_shape_matches_get_output_dim(alphabet_length):
    encoder = _make_encoder(alphabet_length, cnn_channels=16)
    sequences, _, _ = make_batch(alphabet_length=alphabet_length)
    output = encoder(sequences)
    assert output.shape == (sequences.shape[0], encoder.get_output_dim())


def test_output_dim_is_transformer_plus_cnn_branch(alphabet_length):
    encoder = _make_encoder(alphabet_length, cnn_channels=16)
    assert (
        encoder.get_output_dim()
        == encoder.transformer_encoder.get_output_dim() + 16 * 2
    )


def test_gradient_flows_through_both_branches(alphabet_length):
    encoder = _make_encoder(alphabet_length, cnn_channels=16)
    sequences, _, _ = make_batch(alphabet_length=alphabet_length)
    output = encoder(sequences)
    output.sum().backward()
    assert encoder.cnn.weight.grad is not None
    assert torch.isfinite(encoder.cnn.weight.grad).all()
    transformer_param = next(encoder.transformer_encoder.parameters())
    assert transformer_param.grad is not None
    assert torch.isfinite(transformer_param.grad).all()


def test_registered_in_factory_and_wires_into_a_full_model(alphabet_length):
    """encoder_type="transformer_cnn" must be usable end-to-end, same as any other
    encoder -- AIRRTM_Model sizes its latent layers from encoder.get_output_dim()
    generically, so nothing else should need to change."""
    config = make_model_config(4, alphabet_length)
    config["encoder_type"] = "transformer_cnn"
    config["encoder_params"] = {**config["encoder_params"], "cnn_channels": 16}
    model = model_factory(**config)

    sequences, repertoire_ids, _ = make_batch(alphabet_length=alphabet_length)
    output = model(repertoire_ids, sequences)
    n_sequences = sequences.shape[0]
    assert output["kl_divergences"].shape == (n_sequences,)
    assert output["decoded_sequences"].shape == (
        n_sequences,
        MAX_LENGTH,
        alphabet_length + 1,
    )
