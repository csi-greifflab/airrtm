from typing import NotRequired, TypedDict

import torch


class AIRRTM_ModelOutput(TypedDict):
    #: Per-sequence KL divergence between the encoded posterior and N(0, I).
    kl_divergences: torch.Tensor
    #: Bounded per-sequence score sum_t theta_rt * sigmoid(phi_ts), in (0, 1).
    #: Reported for continuity with earlier runs; not used by the TM loss.
    #: None when the model was built with use_topic_model=False (phi dropped).
    tm_likelihoods: torch.Tensor | None
    #: Unnormalised per-sequence log score log sum_t theta_rt * exp(phi_ts).
    #: The TM loss normalises this over the sequences in the batch.
    #: None when the model was built with use_topic_model=False (phi dropped).
    tm_log_scores: torch.Tensor | None
    #: (S, T) raw sequence-topic logits.
    #: None when the model was built with use_topic_model=False (phi dropped).
    seq_topic_logits: torch.Tensor | None
    #: (S, T) sigmoid of the above; what the scoring path also uses.
    #: None when the model was built with use_topic_model=False (phi dropped).
    seq_topic_probabilities: torch.Tensor | None
    #: (S, T) log topic proportions of each sequence's own repertoire.
    log_topic_proportions: torch.Tensor
    #: Per-sequence label logits (the MIL instance scores).
    label_likelihoods: torch.Tensor
    #: (S, P, A) per-position logits over the alphabet.
    decoded_sequences: torch.Tensor


class AIRRTM_ModelTarget(TypedDict):
    sequence_repertoire_indicators: torch.Tensor
    sequence_repertoire_labels: torch.Tensor
    sequences: torch.Tensor
    #: (S,) repertoire index of each sequence, used to group the batch for the
    #: per-repertoire terms. Grouping is derived from these ids rather than from
    #: the batch layout, so the loss is invariant to how the batch is ordered.
    repertoire_ids: torch.Tensor
    #: (S,) clonal abundance of each sequence, or None. Only read when
    #: CompositeLoss(abundance_weighted_losses=True); see that flag for why
    #: abundance belongs here rather than in the batch sampler or Theta pooling.
    weights: NotRequired[torch.Tensor | None]
