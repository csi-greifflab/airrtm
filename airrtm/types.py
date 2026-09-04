from typing import TypedDict

import torch


class AIRRTM_ModelOutput(TypedDict):
    # l1_topic_layer_reg: torch.Tensor
    kl_divergences: torch.Tensor
    tm_likelihoods: torch.Tensor
    label_likelihoods: torch.Tensor
    decoded_sequences: torch.Tensor


class AIRRTM_ModelTarget(TypedDict):
    sequence_repertoire_indicators: torch.Tensor
    sequence_repertoire_labels: torch.Tensor
    sequences: torch.Tensor
