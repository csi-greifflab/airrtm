# AIRRTM

This repository contains the code of the AIRRTM model from the article [Weakly supervised identification and generation of adaptive immune receptor sequences associated with immune disease status](https://www.biorxiv.org/content/10.1101/2023.09.24.558823v1).

AIRRTM learns which immune receptor sequences drive a **repertoire-level** binary label,
using only repertoire labels — no per-sequence annotation. It couples a sequence VAE to a
topic model: repertoire→topic proportions `Theta`, sequence→topic probabilities `Phi`, and a
loss combining the topic-model likelihood, a repertoire-label term, and the VAE objective.

`v2` (this branch) is the PyTorch reimplementation. See [`PLAN.md`](PLAN.md) for the current
work plan and the rationale behind the recent changes.

## Installation

```shell
poetry install
```

Python >= 3.11. The evaluation module additionally needs `scikit-learn` and `scipy`, both
declared in `pyproject.toml`.

## Data

The two synthetic datasets used in the article are in a [separate repository](https://github.com/csi-greifflab/airrtm_data).

Expected input layout:

```
INPUT_DATA_DIR
│   metadata.csv
│   repertoire_1.tsv
│   repertoire_2.tsv
│   ...
```

`metadata.csv` must have columns `label` (repertoire label), `filename`, and `split`
(`train`/`test`):

```
label,filename,split
1,P00492.tsv,train
0,P00413.tsv,train
1,P00875.tsv,test
```

Each sequence file must have a column `cdr3_aa`. These optional columns are used when present:

| column | effect |
|---|---|
| `duplicate_count` | identical clonotypes are collapsed into one weighted row, and sequences are sampled in proportion to clonal abundance |
| `v_call`, `j_call` | V/J gene embeddings are concatenated to the sequence latent before the topic layer |

Pass `--no-counts`, `--no-vj` or `--no-dedup` to preprocessing to ignore them.

## Usage

### 1. Preprocess

```shell
preprocess-from-csv \
    --input_dir INPUT_DATA_DIR \
    --output_dir PROCESSED_DATA_DIR \
    --max_len 26 --min_len 4 --min_n_sequences 10000
```

Every repertoire is padded **and truncated** to exactly `--max_len`, so all repertoires can be
concatenated into a single batch. Writes one `.pt` per repertoire, a filtered `metadata.csv`,
and (when V/J columns exist) a shared `gene_vocabulary.pt`.

### 2. Train

```shell
train-model \
    --input_dir PROCESSED_DATA_DIR \
    --output_dir MODEL_DIR \
    --config emerson_run/config.yaml \
    [--repertoire_slice 0:32]
```

Training uses **repertoire mini-batching**: each optimizer step sees
`n_repertoires_in_batch` repertoires contributing `n_sequences_per_repertoire_in_batch`
sequences each. Both the MIL label term and the batch-normalised topic-model likelihood need
several repertoires per step. `--repertoire_slice` restricts the run to a few repertoires for
a quick shakedown.

Writes `model.pt` (weights plus the config needed to rebuild the module), `config.yaml`,
`history.json`, TensorBoard logs, and per-epoch checkpoints.

### 3. Evaluate

```shell
evaluate-model \
    --input_dir PROCESSED_DATA_DIR \
    --model MODEL_DIR/model.pt \
    --output_dir MODEL_DIR/evaluation
```

Reports, on the held-out `split == "test"` repertoires:

- ROC-AUC / PR-AUC / F1 for repertoire classification, from the quantile representation of
  per-sequence signal intensities and (for amortized models) directly from topic proportions;
- the same metrics for the Emerson-2017 Fisher-exact burden-score baseline;
- per-topic separation between positive and negative repertoires (`topic_separation.csv`);
- the top-ranked candidate sequences (`top_candidate_sequences.csv`) and the sequences that
  load most strongly on each topic (`top_sequences_per_topic.csv`).

For the synthetic datasets, where ground-truth sequence labels exist,
`airrtm.evaluation.precision_at_k` and `signal_enrichment` reproduce the paper's Fig. 2A
precision curves.

### 4. Generate

`airrtm.evaluation.generation` implements Methods 5: fit a diagonal Gaussian to the latents of
the top-scoring sequences, sample at a range of temperatures, and decode.

## Configuration

Key options in `config.yaml`:

| option | meaning |
|---|---|
| `airrtm_params.theta_mode` | `amortized` infers repertoire topic proportions from a sample of the repertoire's own sequences, so **unseen repertoires can be scored**. `free` is the original per-repertoire embedding (training repertoires only). `both` uses the free row as a residual. |
| `loss_config.tm_likelihood_coef` | Weight of the topic-model likelihood. At `0.0` the model degenerates into plain noisy-label MIL with no topic structure. |
| `loss_config.vae_coef`, `reconstruction_loss_coef` | The VAE branch. `reconstruction_loss_coef: 0.0` makes it pure KL. |
| `training_config.tau`, `tau_start` | MIL pooling temperature. `tau -> 0` pools per-sequence label logits towards their mean, `tau -> inf` towards their max; `tau_start` anneals between them. |
| `encoder_params.pooling` | `mean_max` pools over positions with the padding mask applied; `flatten` concatenates all positions (position-dependent, the original behaviour). |
| `data_config.use_vj`, `abundance_weighted_sampling` | Use V/J genes and clonal counts when the data carries them. |

## Emerson run

`emerson_run/` holds the driver scripts for the Emerson CMV dataset. Copy
`emerson_run/.env.example` to `emerson_run/.env`, adjust the paths, then:

```shell
python 1_preprocess_repertoires.py
./2_train_model.sh my_model
./3_evaluate_model.sh my_model
```

The two notebooks in that directory predate the CLIs and call an older API; the shell scripts
above are the supported path.

## Tests

```shell
PYTHONPATH=.:tests pytest tests/
```

## Legacy

`archive/` holds the original TensorFlow implementation from the paper.
