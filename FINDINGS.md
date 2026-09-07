# AIRRTM v2 on Emerson — findings

Working log for the Emerson CMV work. Companion to [`PLAN.md`](PLAN.md), which holds the
original defect list and rationale. Everything here is measured on
`/Warehouse/Andrei/emerson/`; run directories are under `model/`, TensorBoard at
`tb_v2/`.

---

## The one-paragraph summary

The Emerson CMV signal lives in **set membership of rare, public clonotypes** — roughly 482
discriminative CDR3s out of 3.5M, present in ~2.2e-4 of a CMV+ repertoire's clonotypes. Mean
pooling a sequence embedding destroys that information before any loss can see it, which is
why the label branch sat exactly at chance through five architectural fixes. Attention pooling
was the first change that made it move. Separately, the TM branch has never left its uniform
baseline; that is **not** because the data lacks structure (repertoires differ enormously in
V/J usage) but because topic proportions collapse onto 2 of 30 topics early in training, and a
2-dimensional Theta cannot express a ~100-dimensional systematic difference.

---

## Reference numbers

Anything claiming progress should be read against these.

| quantity | value | source |
|---|---|---|
| **Burden-score baseline, test ROC-AUC** | **0.781** (PR-AUC 0.753, acc 0.715) | Fisher exact, p<1e-3, 482 of 3,539,036 clonotypes |
| Majority-class accuracy, test split | 0.536 | 151 test repertoires, 46.4% positive |
| Chance label loss (weighted BCE) | 0.3466 | `0.5 * ln 2` |
| Signal-clonotype rate, CMV+ | 2.18e-4 (31.3 of 143,756) | measured against the 482 selected |
| Signal-clonotype rate, CMV− | 4.61e-5 (6.6 of 143,313) | ~4.7x enrichment is the whole basis |
| `I(s; r)` ceiling, 4 repertoires | **≥ 0.052 nats** | V+J+length classifier, lower bound |
| Best `tm` reduction achieved | 0.0091 nats | ~17% of that lower bound |

**Dataset.** 748 metadata rows (597 train / 151 test, 44.2% / 46.4% positive); 760 `.pt` files
on disk, so **12 orphan files are not in either split** — always read `metadata.csv`, never glob
the directory. Repertoire sizes: train min 50,335, median 202,848, max 591,738. The sharp floor
near 50,000 is a `min_n_sequences` filter from the original preprocessing, not a property of the
data.

**GPU budget.** Memory is linear at **2.1 GB per 1024 sequences** (fp32), so ~32,768 sequences
per step is the ceiling on an 80 GB A100. bf16 saves only 33%, still OOMs at 65k, and shifts
`tm` by 0.02 — the same magnitude as the entire effect under study. Do not use it for this.

---

## What each run tested

All use `theta_mode: amortized`; none has yet been evaluated on the held-out 151.

| run | change under test | result |
|---|---|---|
| `v2_..._tau0.1_dead` | tau annealed 0.1→1.0 over 200 epochs | label flat at 0.3430 for 12 epochs. Schedule kept tau at ~1.1; useless. |
| `v2_..._tau10` | tau 1.0→10.0 over 200 epochs | label 0.3426 at epoch 31. tau only reaches 1.37 by epoch 27 — anneal far too slow. |
| `v3a_disjoint_theta` | Theta from a disjoint sample | **No effect.** 200 epochs; `tm` 8.9918 vs 8.9918 for the mean-pooled control. |
| `v3b_counts_vj` | counts + V/J + abundance sampling + bag 8192 + tau→20 | **No effect on label.** Early stop epoch 89; `tm` 10.3891, label 0.3429. |
| `v4_attention` | attention pooling + `label_input: repertoire` | **First movement.** label 0.3445→0.3183, `label_acc` 0.554→0.657. Theta collapsed to 2 topics by epoch 18. |
| `v5_tm_heavy_strat` | `tm_likelihood_coef` 0.475→0.855, stratified groups | `tm` unmoved (10.3875). Collapse **delayed** (epoch ~80 vs ~18) but not prevented. |
| `v6_load_balance` | + `topic_usage_coef: 1.0` | Collapse prevented, nothing gained. `tm` 10.3883, label back to baseline. Theta went **uniform** instead — see finding 5. |

---

## Findings

### 1. The label branch was at chance for a structural reason, not a tuning one

Mean pooling asks *what is the average topic composition of this repertoire*. A repertoire with
31 discriminative clonotypes out of 143,756 has an essentially identical average to one with 6.
The information is gone at the pooling step, and no bag size, temperature, or loss weight
downstream recovers it.

Confirmed by a linear probe on the frozen trained representation (597 repertoires, `rec_acc`
0.999): **no pooled statistic beat a shuffled control.**

```
mean of first 10 topics (the label head's view)   0.581 +/- 0.050
mean of all 30 topics                             0.556 +/- 0.044
mean+std of all 30 topics                         0.537 +/- 0.025
5 quantiles of all 30 topics                      0.510 +/- 0.028
softmax(mean logit) = amortized theta             0.567 +/- 0.038
SHUFFLED labels (control)                         0.533 +/- 0.045
```

Attention pooling (`TopicAttentionPooling`, gated attention with one map per topic) makes the
rare-subset information representable — a single distinctive sequence among 512 shifts the
attention-pooled representation >10x more than the mean-pooled one. That is what unblocked v4.

### 2. Theta collapses, and that is what gates the TM branch

v4 reached its label score by collapsing Theta onto **2 of 30 topics** (entropy 0.187 vs `log 30
= 3.401`, max probability 0.956; 44 repertoires peaked on topic 5, 16 on topic 4). A collapsed
Theta makes the TM likelihood uninformative by construction, so `tm` cannot move regardless of
its coefficient.

The causal link is measured: raising `tm_likelihood_coef` from 0.475 to 0.855 both slowed the
collapse and lowered `tm`.

```
             tm        label     label_acc    H_theta      (epoch 18)
v4        10.3920     0.3183      0.6573      0.2836
v5        10.3880     0.3321      0.5778      2.0840
```

But v5 collapsed anyway by epoch ~80 (`H_theta` 0.3123). The coefficient delays collapse; it
does not prevent it. And it is nearly exhausted — `tm_likelihood_coef` is bounded in [0,1] by
construction and sits at 0.9.

### 3. The TM branch has real headroom — roughly 6x

Emerson repertoires differ **systematically and substantially**:

```
V-gene usage   across-repertoire std 0.4312 | max per-bin range 0.3464
J-gene usage   across-repertoire std 0.2579 | max per-bin range 0.3884
CDR3 length    across-repertoire std 0.1948 | max per-bin range 0.3783
```

A single V gene's frequency varies by 35 percentage points between individuals. The maximum
reduction of `-log p(s|r)` below `log(S)` is exactly `I(s; r)`, bounded below by a supervised
classifier:

```
K= 4:  chance CE 1.3863   achieved 1.3341   ->  I(s;r) >= 0.0522 nats
K= 8:  chance CE 2.0794   achieved 2.0277   ->  I(s;r) >= 0.0517 nats
K=16:  chance CE 2.7726   achieved 2.6767   ->  I(s;r) >= 0.0959 nats

observed tm reduction:  0.0091 nats
```

That bound uses only V gene, J gene and length, ignoring the CDR3 sequence — so the true ceiling
is higher. **`tm` at the uniform baseline is not the information-theoretic optimum.**

Caveat on what a working TM branch buys: these systematic differences are **not** CMV-related
(V usage predicts CMV at 0.487 ROC-AUC). A well-fit TM branch would model individual and HLA
variation — the paper's signal-agnostic topics, whose role is to absorb nuisance so the signal
topics have a clean background. It will not move the label on its own.

### 4. The Emerson TSVs mislabel their gene columns

`v_call` and `j_call` hold bare allele suffixes (3 and 2 distinct values). The real genes are in
**`v_resolved` / `j_resolved`** (75–99 V, 14–15 J). Preprocessing now raises if a gene vocabulary
has fewer than 6 entries, so this cannot pass silently again.

### 5. Theta has *two* degenerate solutions, and each branch needs it to avoid both

v6 enabled `topic_usage_coef` alone. It did exactly what it promised and gained nothing:

```
              tm        label    label_acc   H_theta   H_usage      (max H = log 30 = 3.4012)
ep  0     10.3960     0.3448     0.5563     3.3665    3.3963
ep 24     10.3879     0.3411     0.5563     3.2170    3.3917
```

Collapse was prevented (`H_usage` held at 3.39), but `H_theta` stayed at 3.22 — every repertoire
using all 30 topics near-uniformly. `tm` matched v5 exactly (10.3879 vs 10.3881) and the label
fell back to the majority baseline.

**Uniform Theta is as useless as collapsed Theta**, for the same underlying reason: if every
repertoire has the same Theta, then `p(s|r)` is identical for all `r` (so `tm = log S`), and a
`label_input: repertoire` head reads a constant. Both branches need Theta to *differ across
repertoires*; collapse and uniformity are two different ways of destroying that.

Crucially, `topic_usage_coef` cannot distinguish them. Maximising `H(theta_bar)` is satisfied
equally by "each repertoire sharp on a different topic" and by "every repertoire uniform" —
`tests/test_losses.py::test_topic_usage_allows_per_repertoire_sparsity` asserts precisely that
equivalence. The term *permits* sparsity; it does not *ask* for it, and from a uniform
initialisation uniform is where training stays.

**The two entropy terms are a pair and must be used together:**

| term | quantity | want | alone gives |
|---|---|---|---|
| `theta_entropy_coef` | mean of the entropies, `(1/R) sum_r H(theta_r)` | low | collapse (v4, v5) |
| `topic_usage_coef` | entropy of the mean, `H(theta_bar)` | high | uniformity (v6) |

Target signature: `H_theta` falling toward ~1 while `H_usage` holds near 3.4.

### 6. Ruled out by measurement

- **VAE coefficient.** Per-term encoder gradient norms: `rec` 1.56e-2, `kl` 9.67e-3, `tm`
  4.95e-2, `label` 1.25e-2. The VAE is not crowding anything out; `tm` is the largest gradient
  in the model.
- **Theta self-reference.** Inferring Theta from a disjoint sample changed nothing over 200
  epochs.
- **Witness rate / bag size.** At bag 8192, ~83% of CMV+ bags contain signal (vs ~20% at 1024).
  No effect on the label.
- **MIL temperature.** tau from 0.1 to 20, annealed and fixed. No effect. Note tau=1 beat tau=5
  in a controlled test — with `label_input: repertoire` the pooling is inert anyway.

---

## Code changes

On top of everything in `PLAN.md` (all merged and tested):

| change | file |
|---|---|
| `TopicAttentionPooling` — gated attention, one map per topic, abundance as log-weights | `models/airrtm_model.py` |
| `label_input: sequence \| repertoire`, with a guard rejecting `repertoire` + `theta_mode: free` (the v1 memorisation shortcut) | `models/airrtm_model.py` |
| `topic_usage_coef` — load balancing on the entropy of the mean Theta, EMA-smoothed across steps | `losses/composite_loss.py` |
| Stratified repertoire groups — single-class batches drop from ~14% to <2% | `training/train.py` |
| `tau_anneal_epochs` — decouples the ramp from run length | `training/train.py` |
| `--v-colname` / `--j-colname` plus a guard on implausible gene vocabularies | `cli/preprocess_sequences.py` |

**55 tests** in `tests/`, including regression tests for each defect and behavioural tests for
attention pooling and load balancing.

**Watch the two entropy terms.** `theta_entropy` is the *mean of the entropies* (per-repertoire
spread, want low). `topic_usage_entropy` is the *entropy of the mean* (corpus-level usage, want
high). They differ only in where the average sits, and minimising the first alone drives
collapse.

---

## What to do next

### First, and before more regularisers

1. **Validate on the synthetic datasets** (`S_single`, `S_poly` from the
   [airrtm_data](https://github.com/csi-greifflab/airrtm_data) repo). They have ground-truth
   sequence labels, and the TM branch is known to work there because repertoires differ *only*
   by the planted signal. This separates "the rewritten TM term is wrong" from "Emerson is
   hostile to this formulation" — and **nothing so far has established which**.
   `airrtm.evaluation.precision_at_k` and `signal_enrichment` reproduce the paper's Fig. 2A
   curves directly. Highest value, cheap, and the control this work is missing.

   Three interventions have now moved `tm` by ~0.009 nats against a ≥0.052 ceiling. That is the
   signal to stop tuning the objective and check the objective is right.

2. **Run `evaluate-model` on the held-out 151.** No run has been evaluated yet; v4 and v5 have
   usable checkpoints. Until this number exists, nothing here is comparable to the 0.781
   baseline. v4 is the interesting one — the only run whose label term learned anything.

### The obvious next run (v7)

3. **Both entropy terms together**, per finding 5 — neither works alone:

   ```yaml
   theta_entropy_coef: 0.5    # per-repertoire sparsity; alone -> collapse
   topic_usage_coef:   1.0    # corpus-level spread;     alone -> uniformity
   topic_usage_momentum: 0.9
   ```

   Both live on the same [0, `log n_topics`] scale, so the ratio is meaningful; usage is
   weighted higher because collapse is the failure with a real attractor. Watch for `H_theta`
   and `H_usage` **separating**. Otherwise identical to v6, so it stays a clean A/B.

### Then

4. **Restrict the TM likelihood to public clonotypes.** 93.3% of clonotypes are private to one
   individual, and origin is near-unpredictable from sequence (0.177 vs 0.125 chance). The 6.7%
   shared fraction is where cross-individual structure and the CMV signal both live. Private
   sequences would contribute to the VAE only.
5. **Reconsider the topic bottleneck.** 30 topics to express variation across 99 V genes and 597
   individuals may simply be too few, independent of how Theta is regularised.
6. **Generation** (`airrtm.evaluation.generation`, paper Methods 5) — blocked until the signal
   topics mean something.

---

## Continuing on another machine

**The code is not committed.** Everything above lives in the working tree on branch `v2`
(~2,100 lines changed across 17 files, plus 8 new paths: `PLAN.md`, `FINDINGS.md`,
`airrtm/evaluation/`, `airrtm/cli/evaluate.py`, `airrtm/utils/loading.py`, `tests/`,
`emerson_run/.env.example`, `emerson_run/3_evaluate_model.sh`). Commit and push before moving,
or it is lost.

**Environment.** `poetry install`. `scikit-learn` and `scipy` are new dependencies added to
`pyproject.toml` for `airrtm/evaluation/`; the lock file has not been regenerated, so
`poetry lock` may be needed. Tests: `PYTHONPATH=.:tests pytest tests/` — 55 should pass.

**Data.** Under `/Warehouse/Andrei/emerson/`:

| path | what | note |
|---|---|---|
| `raw_data/` | 760 AIRR TSVs + `metadata.csv` | source of truth |
| `processed_data/` | CDR3 only, no counts, no V/J | what runs v2–v3a used |
| `processed_data_vj/` | **counts + V/J + clonotype dedup** | what runs v3b–v6 used; prefer this |
| `model/v*` | run outputs, checkpoints, TensorBoard logs | ~300 MB total |
| `tb_v2/` | symlinks for a single TensorBoard view | `tensorboard --logdir tb_v2` |

Regenerating `processed_data_vj` from scratch takes ~2h of CPU:

```shell
preprocess-from-csv --input_dir raw_data --output_dir processed_data_vj \
    --max_len 26 --min_len 4 --min_n_sequences 10000 \
    --v-colname v_resolved --j-colname j_resolved
```

The `--v-colname` / `--j-colname` flags are **not optional** — see finding 4.

**Hardware note.** All runs used bag 4x8192 = 32,768 sequences per step, ~68 GB. On a smaller
card, reduce `n_sequences_per_repertoire_in_batch` before `n_repertoires_in_batch`: bags below
~8192 leave most CMV+ bags free of any signal clonotype (finding: 83% at 8192 vs 20% at 1024).
Do not reach for bf16 — it shifts `tm` by 0.02, the size of the effect under study.

### Method note

Four hypotheses were tested and discarded before the frozen-representation probe was run; that
probe cost two minutes and pointed straight at the answer. **When a loss term looks inert,
probe the representation before tuning the objective.** Related: two "controls" in this work
were not controls (a `free`-Theta test where the free parameters received no gradient, and an
origin probe that omitted V/J). Check that a control can actually exhibit the effect it is
supposed to bound.
