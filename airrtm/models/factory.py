import inspect

from airrtm.models.airrtm_model import AIRRTM_Model
from airrtm.models.decoder import LSTMDecoder, TransformerDecoder
from airrtm.models.encoder import LSTMEncoder, TransformerCNNEncoder, TransformerEncoder


ENCODERS = {
    "transformer": TransformerEncoder,
    "lstm": LSTMEncoder,
    "transformer_cnn": TransformerCNNEncoder,
}
DECODERS = {"transformer": TransformerDecoder, "lstm": LSTMDecoder}


def model_factory(
    encoder_type: str,
    decoder_type: str,
    encoder_params: dict,
    decoder_params: dict,
    airrtm_params: dict,
) -> AIRRTM_Model:
    if encoder_type not in ENCODERS:
        raise ValueError(
            f"Unsupported encoder type {encoder_type}, expected one of {sorted(ENCODERS)}"
        )
    if decoder_type not in DECODERS:
        raise ValueError(
            f"Unsupported decoder type {decoder_type}, expected one of {sorted(DECODERS)}"
        )

    encoder_params = dict(encoder_params)
    decoder_params = dict(decoder_params)
    airrtm_params = dict(airrtm_params)
    # The decoder starts from the VAE latent, so its input width is not free.
    decoder_params["input_dim"] = airrtm_params["latent_dim"]

    encoder_class, decoder_class = ENCODERS[encoder_type], DECODERS[decoder_type]
    _validate_config(encoder_params, encoder_class, "encoder_params")
    _validate_config(decoder_params, decoder_class, "decoder_params")
    _validate_config(airrtm_params, AIRRTM_Model, "airrtm_params", skip={"encoder", "decoder"})

    return AIRRTM_Model(
        encoder=encoder_class(**encoder_params),
        decoder=decoder_class(**decoder_params),
        **airrtm_params,
    )


def _validate_config(
    config: dict, target: type, name: str, skip: set[str] | None = None
) -> None:
    """Reject unknown or missing keys before a half-built model fails obscurely."""
    skip = skip or set()
    signature = inspect.signature(target.__init__)
    parameters = {
        key: parameter
        for key, parameter in signature.parameters.items()
        if key not in {"self", *skip}
    }
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        accepted = None
    else:
        accepted = set(parameters)

    if accepted is not None:
        unknown = set(config) - accepted
        if unknown:
            raise ValueError(
                f"Unknown keys in {name}: {sorted(unknown)}. "
                f"Accepted: {sorted(accepted)}"
            )
    required = {
        key
        for key, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing keys in {name}: {sorted(missing)}")
