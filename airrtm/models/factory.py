from airrtm.models.airrtm_model import AIRRTM_Model
from airrtm.models.decoder import LSTMDecoder, TransformerDecoder
from airrtm.models.encoder import LSTMEncoder, TransformerEncoder


def model_factory(
    encoder_type: str,
    decoder_type: str,
    encoder_params: dict[str, str | int | float],
    decoder_params: dict[str, str | int | float],
    airrtm_params: dict[str, str | int | float],
) -> AIRRTM_Model:
    _validate_config(encoder_params)
    _validate_config(decoder_params)
    _validate_config(airrtm_params)
    # encoder_params["output_dim"] = airrtm_params["latent_dim"]
    decoder_params["input_dim"] = airrtm_params["latent_dim"]
    if encoder_type == "transformer":
        encoder = TransformerEncoder(**encoder_params)
    else:
        raise ValueError(f"Unsupported encoder type {encoder_type}")
    if decoder_type == "transformer":
        decoder = TransformerDecoder(**decoder_params)
    else:
        raise ValueError(f"Unsupported decoder type {decoder_type}")
    model = AIRRTM_Model(
        encoder=encoder,
        decoder=decoder,
        **airrtm_params,
    )
    return model


def _validate_config(config: dict) -> None:  # TODO
    return
