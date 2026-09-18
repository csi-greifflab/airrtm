# AIRRTM v2 on Emerson: restore true topic modeling + fix blocking defects

## 2026-09-07 — new machine, next 24-48h

Fresh GPU box (8x H100 80GB, no prior venv). Everything below "Implementation status"
is now historical — all listed defects are fixed and merged (commit `143ab60` / `d713a2c`,
55 tests passing) and `airrtm_model.py`/`composite_loss.py` already expose
`theta_pooling`, `label_input`, `theta_entropy_coef`, `topic_usage_coef` as config knobs.
`FINDINGS.md` (from the previous A100 machine) is the up-to-date empirical log — read that
first, this section only says what changes on *this* machine and what's running now.

**Environment**: `poetry` is not installed here; used `python3 -m venv .venv && pip install -e .`
instead (torch 2.14+cu130, sees all 8 GPUs). `/Warehouse/Andrei/emerson/` has
`processed_data_vj/` (748 metadata rows, matches FINDINGS.md) but no `raw_data/` and no
prior current-code checkpoints — only one old pickled-`nn.Module` run
(`model/per_rep_logsumexp_..._rep_4_seq_2048/`, 800+ epoch checkpoints) that predates this
code and per `PLAN.md` original guidance should be **retrained, not converted**.
`emerson_run/.env` was missing (gitignored) and `CONFIG_PATH` in `.env.example` pointed at
a `/home/as/...` path from the old machine — recreated `.env` with this machine's paths.

**Decisions for this run, and why:**
1. Skip re-deriving whether the TM objective is sound on Emerson before checking it on
   synthetic data — FINDINGS.md's own "what to do next" says this control is missing and is
   "highest value, cheap". Cloned `csi-greifflab/airrtm_data`, unzipped `S1` (single-signal)
   at witness rates 0.005 and 0.001, and `S2` (poly-signal) at 0.001, into
   `/Warehouse/Andrei/synthetic/`. Preprocessed with `--max_len 20 --min_len 6` (paper's
   settings for this data, not Emerson's 26). Training queued once preprocessing lands
   (~2.5-3h/dataset on this CPU, ran in parallel).
2. Launched three Emerson runs immediately (GPUs 0-2) rather than waiting on synthetic
   validation first, since each takes many hours and the synthetic check cannot fail fast
   enough to gate them without wasting a day of idle H100s:
   - `emerson_run/config_v7.yaml` (GPU 0, `model/v7_attn_dualentropy/`) — FINDINGS.md's
     recommended next run: `theta_pooling: attention`, `label_input: repertoire`, bag size
     raised from the shipped default (1024) to 8192/repertoire (finding: 1024 leaves most
     CMV+ bags with no signal clonotype), both entropy terms together
     (`theta_entropy_coef: 0.5`, `topic_usage_coef: 1.0`), `tm_likelihood_coef: 0.7`.
   - `emerson_run/config_v7_wide_topics.yaml` (GPU 1) — same, but 60 topics (20 signal + 40
     non-signal) instead of 30, testing finding 5's "bottleneck may be too narrow" note.
   - `emerson_run/config_ablation_meanpool.yaml` (GPU 2) — same bag size and entropy coefs
     as v7, but `theta_pooling: mean` / `label_input: sequence`. Control: isolates whether
     attention pooling itself is still load-bearing in this codebase (as opposed to the
     larger bag or the entropy terms), since nothing on *this* machine has confirmed that
     yet — the old machine's finding used a different commit.
   All three smoke-tested clean on a 12-repertoire slice before the full launch (no crash,
   `rec_acc` rising off the pad floor). Full runs confirmed alive at ~70-72GB/GPU, 100%
   util, `n_epochs: 200, patience: 25` — expect low tens of hours each, not all 200 epochs.

**Deferred, not started this session** (next priorities per `FINDINGS.md`, in order):
restrict the TM likelihood to public clonotypes (needs a preprocessing/data-loader change,
not just a config flag — do this once the synthetic control and v7 report back, not blind);
evaluate whichever of the three above finishes first, or is killed by early stopping, against
the 151-repertoire test split and the 0.781 burden-score baseline (no run on this machine has
been evaluated yet, unlike the old machine where v4/v5 checkpoints existed unevaluated);
generation, still blocked on signal topics meaning something.

**Update, same session — a real bug found and fixed, GPU allocation changed to 7 runs + 1 eval.**
All six runs above collapsed identically (`H_theta` and `H_usage` both to ~0 by epoch 0-4,
Emerson and synthetic alike) because `composite_loss.py::_topic_usage`'s EMA only passes
gradient through its `(1-momentum)` branch, making the shipped `topic_usage_coef: 1.0` an
effective 0.1 — five times weaker than `theta_entropy_coef`, not two times stronger as the
config's own comment intended. Full account, the failed naive fix, and the chosen value
(`topic_usage_coef: 5.0`, unconfirmed past a small smoke test) are in `FINDINGS.md`'s
"same day" update — read that before touching this coefficient again. All six runs were killed
and relaunched with the fix, plus a seventh (`v7_tmheavy`, GPU 6, `tm_likelihood_coef: 0.85`) to
use all 8 GPUs minus one reserved for `evaluate-model`. Launching 4 Emerson jobs at once also
hit a separate CPU-thread-thrashing stall (each spawns ~168 OMP threads); fixed with
`OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` on relaunch, worth doing by default for >2-3 concurrent
jobs on this box.

**To pick this back up**: check `model/*.log` (nohup, no tmux) and `nvidia-smi`. All seven
training processes were launched with `nohup ... & disown`, independent of any interactive
session. Kill specific PIDs (`ps aux | grep train-model`) if these should be superseded rather
than left to finish — avoid a blanket `pkill -f train-model`, which needs explicit confirmation
here.

---

## Implementation status

Everything in Parts 1–3 below is **implemented and unit-tested**; no training run has been
started (that happens on the GPU machine). What remains is Execution-order steps 2 and 5–7:
re-preprocessing with counts/V/J, running the baselines, and the coefficient sweep.

| Done | Where |
|---|---|
| CLI blockers, repertoire mini-batching, `zero_grad` | `airrtm/training/train.py`, `airrtm/cli/train.py` |
| Decoder returns logits (double-softmax removed) | `airrtm/models/decoder.py` |
| In-batch normalised TM likelihood | `airrtm/losses/composite_loss.py`, `AIRRTM_Model.forward` |
| Amortized `Theta` (`theta_mode`) | `airrtm/models/airrtm_model.py` |
| Per-sequence `log sigma`; `z_batch_norm` removed | `airrtm/models/airrtm_model.py` |
| Padding mask in encoder + reconstruction loss | `airrtm/models/encoder.py`, `composite_loss.py` |
| Real `label_accuracy`; regularisers named correctly | `airrtm/losses/composite_loss.py` |
| Counts, V/J, clonotype dedup, exact truncation | `airrtm/cli/preprocess_sequences.py`, `airrtm/utils/sequence_dataset.py` |
| AUC / precision@k / topics / generation / burden baseline | `airrtm/evaluation/`, `airrtm/cli/evaluate.py` |
| 34 unit tests, incl. regression tests per defect | `tests/` |

Verified end to end on synthetic data: `preprocess-from-csv` → `train-model` → `evaluate-model`
all complete, reconstruction accuracy rises off the padding floor, and `tm_loss` starts at
exactly `log(batch_size)` — the correct uniform baseline for a normalised likelihood.

Two things to know before running on the GPU box:

1. `poetry install` (or `pip install scikit-learn scipy`) — both are new dependencies.
2. The old checkpoints under `model/` are pickled `nn.Module`s and will **not** load against
   the edited classes. New checkpoints are `{state_dict, model_config}` dicts instead. Re-train
   rather than trying to convert; the old runs used the degenerate objective anyway.

## Context

AIRRTM (Slabodkin et al., bioRxiv 2023.09.24.558823) learns which AIR sequences drive a
repertoire-level binary label, using only repertoire labels. The architecture couples a
sequence VAE to a topic model: repertoire→topic proportions Θ (`n_repertoires × n_topics`),
sequence→topic probabilities Φ derived from the encoder, and three losses — TM likelihood
`p(s|R_k) = Σ_t φ_ts · θ_tk`, repertoire-label BCE on signal-topic proportions, and the VAE
(reconstruction + KL).

v2 is the torch reimplementation (`airrtm/`, 2483 lines, commit `143ab60`). On Emerson CMV
data it currently reaches ~0.61–0.64 repertoire accuracy against a 0.558 majority baseline,
with the TM term switched off (`tm_likelihood_coef: 0.00`) and reconstruction switched off
(`reconstruction_loss_coef: 0.0`). With both off, `label_likelihood_coef = 1 - 0 = 1` and the
objective reduces to **MIL noisy-label learning on sequences** — the topic model is inert.

Goal: make the topic-modeling half actually carry signal, and evaluate properly on Emerson.

### Confirmed environment

| | |
|---|---|
| Data | `/Warehouse/Andrei/emerson/` — `raw_data/` 760 TSVs (AIRR format), `processed_data/` 760 `.pt` + `metadata.csv` (748 rows: 597 train / 151 test; label mean 0.442 / 0.464) |
| Sequences | 35k–592k per repertoire, `max_length=26`, alphabet 21 aa (`RHKDESTNQCGPAVILMFYW*`) + pad token 21; pad is ~54% of all positions; ~4% exact duplicate rows per repertoire |
| Raw columns unused | `duplicate_count`, `v_call`, `j_call`, `v_resolved`, `j_resolved` |
| Checkpoints | 39 model dirs under `model/`; whole pickled `nn.Module`s saved as `checkpoint_epoch_N.py` |
| Hardware | 4× A100 80GB; poetry venv `airrtm-eW4W92NM-py3.12` (torch 2.6.0+cu124) — note the venv does **not** have `airrtm` installed, and the notebooks ran on py3.11 |
| Missing | `emerson_run/.env` (gitignored); needs `PROJECT_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODEL_DIR`, `MAX_SEQUENCE_LENGTH`, `MIN_SEQUENCE_LENGTH`, `MIN_N_SEQUENCES`, `CONFIG_PATH` |

---

## Part 1 — Blocking defects (must fix before any tuning)

These are wrong-code, not hyperparameters. Ranked by impact.

### 1.1 `train-model` CLI cannot start

- `airrtm/cli/train.py:73` — `CompositeLoss(**training_config["loss_config"])` passes only the
  4 YAML keys; `CompositeLoss.__init__` also requires `n_sequences_per_repertoire` and
  `positive_class_weight`, neither with a default → `TypeError`.
- `airrtm/training/train.py:140,150,222,232` — `criterion(predictions, targets)` omits the
  required `tau` argument → `TypeError`.
- `airrtm/training/train.py:48` — `total_batch_size = n_repertoires * n_sequences_per_repertoire_in_batch`
  = 597 × 2048 ≈ 1.2M sequences in one forward pass. No repertoire mini-batching.
- `airrtm/training/train.py:155-156` — `backward()` then `step()`, **`optimizer.zero_grad()` is
  never called**. Gradients accumulate across the entire epoch.

This is why all real runs live in `emerson_run/2a_train_interactive.ipynb`. The notebook's loop
(repertoire mini-batching, `MIN_REPERTOIRE_SIZE` resampling, per-group step) is the working
version and should be **ported back into `airrtm/training/train.py`**, with its own bug fixed:
`zero_grad()` is called once per outer batch while `step()` runs once per inner repertoire group
(`2a_train_interactive.ipynb` cell 4), so gradients leak across groups.

### 1.2 Double softmax kills reconstruction

`airrtm/models/decoder.py:126-129` returns `softmax(transformer_output, dim=2)`, and
`airrtm/losses/composite_loss.py:152` feeds it to `torch.nn.CrossEntropyLoss`, which applies
`log_softmax` again. Gradients are flattened; reconstruction accuracy in the live run is 0.005.

Fix: return logits from the decoder; apply softmax only where probabilities are wanted
(`AIRRTM_Model.latent_to_sequence`, generation).

### 1.3 The TM likelihood is unnormalized → degenerate

`airrtm/models/airrtm_model.py:87-90`:
```
seq_topic_probabilities_ST = sigmoid(seq_topic_logits_ST)      # independent per topic
seq_total_probability_S    = (topic_proportions_ST * seq_topic_probabilities_ST).sum(1)
```
Θ is softmax-normalized over topics (sums to 1), but Φ is an **independent sigmoid per topic**,
never normalized over sequences. With `sequence_repertoire_indicators` always 1 in the live loop
(the negative branch is dead — see 1.4), the TM loss is monotone decreasing in every σ(φ), so the
optimum is σ(φ_ts) → 1 for all t, s. Observed `tm_likelihoods` are 0.9991 for self and 0.9993 for
"other" repertoires — the term carries no information, which is exactly why `tm_likelihood_coef`
had to be set to 0. **This single defect is what reduced AIRRTM to noisy-label learning.**

Fix (chosen route): in-batch normalized likelihood — sampled softmax / InfoNCE over the batch.
```
log p(s | r) = log Σ_t θ_rt φ_ts  −  logsumexp_{s' ∈ batch} log Σ_t θ_rt φ_ts'
```
The batch already contains `n_repertoires_in_batch × n_sequences_per_repertoire` sequences, giving
a natural negative pool. Topics now compete: raising φ for one sequence lowers the normalized
likelihood of the others, so Φ must specialize. Keep φ as an unnormalized non-negative score
(`exp` of the logit, or `softplus`) rather than a sigmoid, since the normalizer handles scale.

### 1.4 Dead negative-sampling branch

`airrtm/training/train.py:275-294` `_sample_other_repertoire_ids` is intact in the package but
its call site in the notebook is commented out, and the notebook's copy opens with
`raise Exception("FIXME: use global list of rep ids to choose from")`. The package version also
samples only from opposite-label repertoires, which conflates "different repertoire" with
"different label". With 1.3 in place this term becomes redundant — delete it rather than repair it.

### 1.5 `label_accuracy` is not an accuracy

`airrtm/losses/composite_loss.py:181` — `"label_accuracy": label_loss`. The genuinely computed
`label_accuracy` (lines 86, 107-109, 131-134) is discarded. Every "accuracy ≈ 0.67" in the
notebooks and in `config_acc_0.6_loss_-0.22.yaml` is a **loss value**, not an accuracy.

### 1.6 Scoring path ≠ training path

`airrtm/models/airrtm_model.py:154` and `:166` apply `latent_space_to_topic_proportions_layer`
and use the **raw logits**; `forward` at `:87` applies `sigmoid`. So `predict_topic_probabilities`
and `predict_signal_intensity` score a different quantity than the one trained. (The notebooks
noticed and hand-rolled `predict_signal_intensity_2` with the sigmoid restored — the package
method stayed wrong.)

### 1.7 Padding is never masked

- `airrtm/models/encoder.py:71-72` — `self.transformer(x_SP)` with no `mask=`. Attention sees
  ~54% pad tokens as real content.
- `composite_loss.py:152` — reconstruction CE includes pad positions, so ~54% of "accuracy" is free.
- `airrtm_model.py:141` — `x_SPE.reshape(batch, -1)` flattens 26×22=572 into `z_mean_layer`,
  baking in absolute position.

Fix: pass `mask = x_SP != pad_value` to the encoder, `ignore_index=pad_value` in the CE, and
consider masked mean/attention pooling instead of the flatten.

### 1.8 Dead `z_batch_norm`

`airrtm/models/airrtm_model.py:50` creates `BatchNorm1d(latent_dim)`; `forward` never calls it.
80 unused parameters, present in every checkpoint.

### 1.9 `topic_l1_reg` regularizes the label head, not topics

`composite_loss.py:137-140` — `|sigmoid(label_likelihoods)|.sum() / n_sequences_per_repertoire`.
That is the mean predicted label probability (a witness-rate prior), not an L1 on Θ or Φ. Rename
it `predicted_witness_rate`, and add a real topic regularizer separately (see 2.3).

### 1.10 Not-a-VAE posterior

`airrtm_model.py:46-48,122-124` — `z_log_sigma_param` is a **single global scalar** shared by
every sequence and every latent dimension. KL collapses to ≈ ‖μ‖²/(2·latent_dim) + const, so the
latent has no input-dependent uncertainty and sampling for generation is isotropic noise of a
learned global width. Fix: have the encoder emit per-sequence `log σ` (a second `Linear` head).

---

## Part 2 — Restoring repertoire label-assisted topic modeling

### 2.1 Amortize Θ (required by the chosen metric)

`airrtm_model.py:35-37` — `repertoire_topic_proportions` is an `Embedding(n_repertoires, n_topics)`
indexed by row. There is **no row for the 151 test repertoires**, which is why both notebooks
filter to `split == "train"` and use a sequence-level holdout instead. Repertoire AUC on the test
split is impossible without this change.

Replace with an inference head:
```
θ_r = softmax( pool_{s ∈ sample(R_r)} [ h(φ_s) ] )
```
- `pool` = masked mean over a sampled subset of the repertoire's sequences (DeepSets-style;
  permutation invariant, matches the bag-of-sequences assumption the paper argues for).
- Optionally weight the pool by `duplicate_count` (see 2.4).
- Keep the free `Embedding` behind a config flag (`theta_mode: free | amortized`) so the old
  behaviour is reproducible for ablation, and so amortized-vs-free can be compared directly.

This is the single change that turns the model from "597 fitted rows" into something that
generalizes — and it makes `_compute_signal_topic_weights` (`airrtm_model.py:178-204`, a faithful
port of the v1 formula) meaningful on unseen repertoires.

### 2.2 Rebalance the loss once the TM term works

`config.yaml` currently: `vae_coef: 0.06`, `reconstruction_loss_coef: 0.0`, `tm_likelihood_coef: 0.0`,
`topic_l1_coef: 0.0`. Note the nesting in `composite_loss.py:164-173`:
`total = vae_coef·(rec_coef·rec + (1−rec_coef)·kl) + (1−vae_coef)·(tm_coef·tm + (1−tm_coef)·label + l1_coef·l1)`
— so `reconstruction_loss_coef: 0.0` means the VAE branch is **pure KL**, and `tm_likelihood_coef: 0.0`
means the non-VAE branch is **pure label loss**. Start from the paper's regime
(`vae_coef ≈ 0.05`, `reconstruction_loss_coef ≈ 0.95`, `tm_likelihood_coef ≈ 0.95`) once 1.2 and
1.3 are fixed, and sweep from there.

### 2.3 Real topic regularization

Add, as separate configurable terms:
- L1 or entropy penalty on Θ rows (sparse topic usage per repertoire).
- Topic decorrelation on Φ (v1 had `decorrelation_regularizer_coef`, unimplemented in v2).
- Keep the existing witness-rate prior (1.9) under its correct name; the notebook's `q_l1` quantile
  trick already depends on it being interpretable as an expected signal fraction.

### 2.4 Add counts and V/J to preprocessing

`airrtm/cli/preprocess_sequences.py:79-82` reads only `cdr3_aa`. Extend to carry
`duplicate_count`, `v_call`, `j_call`:
- `SequenceDataset` (`airrtm/utils/sequence_dataset.py`) gains optional `weights`, `v_ids`, `j_ids`
  tensors alongside `data`/`lengths`, saved and loaded by `save`/`load` (:128-146). Keep the
  existing 2-tuple path working so old `.pt` files still load.
- Batch sampler draws sequences ∝ `duplicate_count` (clonal expansion is real CMV signal).
- V/J one-hot concatenated to `z` before `latent_space_to_topic_proportions_layer`, matching v1's
  `--use_vj` (`archive/model.py:224-231`).
- Deduplicate identical (cdr3, v, j) rows into a single weighted row — removes the ~4% exact
  duplicates and shrinks each repertoire.
- Re-run `emerson_run/1_preprocess_repertoires.py` over all 760 repertoires into a **new**
  output dir; do not overwrite `processed_data/`.

Guard while in there: `SequenceDataset.pad` (:113-126) silently *ignores* `max_length` when the
data is longer, so repertoires can end up with different `max_length` and `torch.concatenate` in
the training loop would fail. All 760 currently land on 26, but assert it explicitly.

---

## Part 3 — Evaluation

All four metrics were requested. Build them as a reusable module (`airrtm/evaluation/`), not
notebook cells, so runs are comparable.

1. **Repertoire AUC on the held-out 151.** Requires 2.1. Score every sequence, build the
   quantile-vector representation the notebooks already use
   (`3_evaluate_interactive.ipynb` cell 17: upper quantiles of the per-sequence signal-intensity
   distribution + mean), classify. Report ROC-AUC and PR-AUC, not just accuracy. Baseline to beat:
   Emerson-2017 Fisher-exact burden score on the same split.
2. **Sequence-level recovery of known CMV TCRs.** Rank all sequences by signal intensity; measure
   enrichment of the published Emerson-2017 CMV-associated CDR3 list. Report precision@k for
   `k ∈ 1..k_max` as in the paper (Fig. 2A), plus the score-distribution comparison (Fig. S1).
3. **Topic interpretability.** Per signal topic: top-scoring CDR3 motifs, V-gene usage bias, length
   distribution, and Θ separation between CMV+ and CMV− repertoires (the paper's Fig. S1/S2 middle
   and right columns).
4. **Generation precision.** Paper's Methods 5: take the top-k identified sequences, compute mean μ
   and std σ of their latents, sample from `N(μ, diag((σ·t)²))` for a range of temperatures `t`,
   decode, and measure novelty/uniqueness plus CMV-likeness. Blocked on 1.2 (double softmax) and
   1.10 (global σ) — do it last.

---

## Execution order

1. **Fix the blockers** (1.1–1.2, 1.5–1.8). Get `train-model` runnable end-to-end on a 32-repertoire
   subset; confirm reconstruction accuracy climbs above the 54% pad floor. *~1 day.*
2. **Baselines.** Emerson Fisher-burden classifier on the test split, plus the current model as-is,
   both under the Part 3 metrics. Nothing after this is interpretable without these numbers. *~half a day.*
3. **In-batch normalized TM likelihood** (1.3) + delete the dead negative branch (1.4). Verify
   `tm_likelihoods` no longer saturate and topics differentiate. *~1 day.*
4. **Amortized Θ** (2.1). Unlocks the test split. *~1–2 days.*
5. **Re-preprocess with counts + V/J** (2.4), in parallel with 3–4 since it is compute-bound. *~1 day.*
6. **Loss rebalance + topic regularizers** (2.2, 2.3), sweep on the A100s. *~2–3 days.*
7. **Generation** (1.10 + Part 3 item 4). *~1–2 days.*

## Files touched

| File | Change |
|---|---|
| `airrtm/training/train.py` | Port the notebook's repertoire mini-batching; fix `zero_grad`/`step`; pass `tau`; drop `_sample_other_repertoire_ids` |
| `airrtm/cli/train.py` | Pass the missing `CompositeLoss` args; add test-split evaluation hook |
| `airrtm/losses/composite_loss.py` | In-batch normalized TM term; fix `label_accuracy`; rename `topic_l1_reg`; add real topic regularizers; `ignore_index` on CE |
| `airrtm/models/airrtm_model.py` | Amortized Θ head; per-sequence `log σ`; remove `z_batch_norm`; sigmoid consistency in `predict_*`; optional V/J concat |
| `airrtm/models/decoder.py` | Return logits, not softmax |
| `airrtm/models/encoder.py` | Accept and apply a padding mask; masked pooling option |
| `airrtm/utils/sequence_dataset.py` | Optional `weights` / `v_ids` / `j_ids`; assert consistent `max_length` |
| `airrtm/cli/preprocess_sequences.py` | Read `duplicate_count`, `v_call`, `j_call`; dedup into weighted rows |
| `airrtm/evaluation/` (new) | AUC, precision@k, topic inspection, generation metrics |
| `emerson_run/config.yaml`, `.env` | New coefficient regime; document the missing `.env` keys |

## Verification

- `airrtm/tests/` (new): shape and gradient tests for `CompositeLoss` (each term non-NaN, finite,
  and zero when its coefficient is zero); a test that the TM likelihood is *not* maximized by
  saturating φ; a test that `AIRRTM_Model.forward` and `predict_signal_intensity` return the same
  topic probabilities for the same input.
- Smoke run: `train-model --input_dir <processed> --output_dir /tmp/smoke --config emerson_run/config.yaml`
  on a 32-repertoire subset for 3 epochs — must complete without a `TypeError` (it currently
  raises two) and produce a decreasing total loss.
- Regression: reload `model/kld_0.06_tau_1.0_256-512_rep_4_seq_2048/checkpoints/checkpoint_epoch_99.py`
  and confirm the refactor still loads old checkpoints (or provide a converter).
- Headline: repertoire ROC-AUC on the 151-repertoire test split, reported against the
  Emerson-2017 baseline, before and after each of steps 3–6.
