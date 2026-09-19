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
| **Burden-score baseline, test ROC-AUC** | **0.874** (PR-AUC 0.827, acc 0.829) | Fisher exact, p<1e-3, use_vj=True, 528 of 3,539,036 clonotypes. Supersedes the 0.781 CDR3-only number below (2026-09-11, see "Burden-score baseline reconciled") |
| Burden-score baseline, CDR3-only (legacy) | 0.781 (PR-AUC 0.753, acc 0.715) | Fisher exact, p<1e-3, use_vj=False, 482 of 3,539,036 clonotypes — the original Emerson et al. 2017 methodology; kept for continuity, no longer the primary reference |
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

## 2026-09-07 — new machine (8x H100), six runs launched

Moved to a fresh box: 8x H100 80GB, no prior venv (`python3 -m venv .venv && pip install -e .`,
torch 2.14+cu130). `/Warehouse/Andrei/emerson/processed_data_vj/` already exists (748 rows,
matches this doc's reference numbers) and 55/55 tests pass unmodified — the defect list above
is confirmed still fixed on this machine's copy of the code. No current-code Emerson checkpoint
existed yet (only the old pickled-`nn.Module` run predating this code), and the recommended
"validate on synthetic data first" step had never been done, so this session did both:

**Synthetic validation (the missing control).** Cloned `csi-greifflab/airrtm_data`, ran
`S1` (single-signal) at witness rates 0.005 and 0.001, and `S2` (poly-signal) at 0.001, through
`preprocess-from-csv --max_len 20 --min_len 6` into `/Warehouse/Andrei/synthetic/`. Training
config: `theta_pooling: attention`, `label_input: repertoire`, bag 4x8192, both entropy terms
(`theta_entropy_coef: 0.5`, `topic_usage_coef: 1.0`), 15 topics (5 signal + 10 non-signal, small
on purpose — this data has one planted signal and no V/J nuisance structure). Running on
GPUs 3/4/5 as `model/S1_w0.005`, `model/S1_w0.001`, `model/S2_w0.001` under
`/Warehouse/Andrei/synthetic/`. **No results yet** — this entry records what was launched and
why, not an outcome. The question this answers: does `tm` leave its uniform baseline and does
`label_acc` clear chance when the signal is clean and V/J nuisance is absent? If not, the
objective itself (not Emerson's difficulty) is the problem.

**Emerson, three parallel configs** (GPUs 0/1/2, `emerson_run/config_v7*.yaml`,
`config_ablation_meanpool.yaml`), all raising the shipped default bag size (1024/repertoire) to
8192 per the hardware note below — 1024 was never enough to guarantee a CMV+ bag contains a
signal clonotype:
- `v7_attn_dualentropy`: attention pooling + `label_input: repertoire` + both entropy terms
  together, `tm_likelihood_coef: 0.7` — the flagship run this document's "what to do next"
  pointed at.
- `v7_wide_topics`: identical, but 60 topics (20+40) instead of 30 — tests finding 5's aside
  that 30 may be too narrow to express Emerson's ~100-dimensional V/J variation.
- `ablation_meanpool`: identical entropy/bag-size settings but `theta_pooling: mean` /
  `label_input: sequence` — isolates whether attention pooling specifically (not just the
  larger bag) is what's load-bearing, since this codebase/commit had not been checked on this
  machine.

All three smoke-tested on a 12-repertoire slice first (clean, `rec_acc` rising off the pad
floor) before the full launch. Confirmed alive post-launch at ~70-73GB/GPU, ~100% util.
`n_epochs: 200, patience: 25`; expect early stopping well before 200 given the epoch length
here (604 steps/epoch, `n_batches_per_repertoire: 4` over ~149 repertoire groups).

**Gap this leaves**: nothing on this machine has been evaluated against the 151-repertoire test
split yet (`evaluate-model` not run this session) — that is the immediate next step once any of
the six above finishes or is killed by early stopping. Public-clonotype TM restriction (this
doc's "Then" item 4) was not started — it needs a data-loader change, not a config flag, and
should wait for the synthetic control's answer rather than be built blind.

### Update, same day: `topic_usage_coef` was silently ~10x weaker than intended

All six runs above collapsed `H_theta` **and** `H_usage` to ~0 by epoch 0-4 — Emerson and
synthetic alike, attention pooling and mean pooling alike. That both entropies died together,
identically, on every config regardless of `theta_pooling`, is what made this a code/config bug
rather than an Emerson-specific finding: finding 5's two failure modes (collapse: both low;
uniform: both high) are supposed to need different causes, but every run hit the *same* one
instantly, on data as different as a 100-repertoire synthetic set and 748-repertoire Emerson.

Root cause, in `composite_loss.py::_topic_usage`:

```python
smoothed_T = momentum * self._theta_bar + (1 - momentum) * batch_mean_T   # theta_bar is .detach()ed
```

Only the `(1 - momentum)` branch carries a gradient, so a configured `topic_usage_coef: 1.0` at
`momentum: 0.9` has an *effective* weight of `1.0 * 0.1 = 0.1` -- five times **weaker** than
`theta_entropy_coef: 0.5`, not the intended two times **stronger**. `theta_entropy_coef` pulls
every repertoire towards a one-hot Theta; with nothing effectively opposing it, all repertoires
picked the same one-hot topic, which trivially minimizes both terms at once. This is a distinct,
more basic problem than finding 5's — that finding assumed the two terms were actually
adversarial, which they never were with the coefficients as shipped.

The naive fix (undo the attenuation: `topic_usage_coef: 10.0`, giving effective weight 1.0)
**overshot into the other failure mode** — both entropies pinned near `log(n_topics)`, i.e.
uniform Theta everywhere, in the same handful of epochs. `5.0` (effective 0.5, 1:1 against
`theta_entropy_coef`) was the first value where a 64-repertoire smoke test showed `H_theta`
falling while `H_usage` stayed contested in a middle range rather than saturating either way —
checked for only ~6 epochs on a small slice, not confirmed on the full 748-repertoire run yet.
All seven configs below use `topic_usage_coef: 5.0`. **If `H_usage` in any live run drifts
toward its floor or `log(n_topics)` ceiling over the first real epochs, that value needs
revisiting** — this was picked from a fast, small-scale check, not a converged sweep.

**All six prior runs were killed and relaunched** with the corrected coefficient, plus a
seventh (`v7_tmheavy`, GPU 6): identical to `v7_attn_dualentropy` but `tm_likelihood_coef: 0.85`
instead of 0.7, extending finding 2's "raising `tm_likelihood_coef` delays collapse" along the
same axis. GPU 7 reserved for `evaluate-model`, per request. A second snag on relaunch, unrelated
to the loss bug: launching 4 Emerson jobs at once left each spawning ~168 OMP/MKL threads (the
core count), so 4×168 threads thrashed over 208 cores and none reached the GPU for several
minutes despite plenty of free RAM. Fixed by capping `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` per
process on relaunch — worth doing by default whenever launching more than 2-3 concurrent jobs on
this box, synthetic runs (lighter model, apparently under the same default) didn't show it.

Current allocation, all seven `nohup ... & disown`, `topic_usage_coef: 5.0` everywhere:

| GPU | run |
|---|---|
| 0 | `emerson/model/v7_attn_dualentropy` |
| 1 | `emerson/model/v7_wide_topics` |
| 2 | `emerson/model/ablation_meanpool` |
| 6 | `emerson/model/v7_tmheavy` |
| 3 | `synthetic/model/S1_w0.005` |
| 4 | `synthetic/model/S1_w0.001` |
| 5 | `synthetic/model/S2_w0.001` |
| 7 | *(reserved for evaluate-model)* |

### Update, next day (2026-09-08): the fix worked on Emerson, not (yet) on synthetic

All four Emerson runs finished by early stopping overnight (patience 25), and for the first
time in this document's history the label branch moved **and** the entropy pair hit the
intended signature simultaneously:

| run | best epoch | `label_acc` (train) | `H_theta` | `H_usage` | `tm` |
|---|---|---|---|---|---|
| `v7_attn_dualentropy` | 84 | 0.606 | 0.152 | 3.069 | 10.398 (unmoved) |
| `v7_wide_topics` (60 topics) | 41 | 0.594 | 0.265 | 3.296 | 10.398 |
| `ablation_meanpool` | 96 | 0.580 | 0.244 | 2.947 | 10.399 |
| `v7_tmheavy` (tm_coef 0.85) | 84 | 0.568 | 0.137 | 3.056 | 10.396 |

`H_theta` low, `H_usage` near `log(30)=3.401` -- exactly finding 5's target, achieved on the
first try with the corrected coefficient. `label_acc` above the 0.536-0.558 majority baseline on
**all four**, including the mean-pooling control, which is a surprise worth flagging rather than
burying: attention pooling was finding 1's answer for why the label branch was stuck, but with
the entropy bug fixed, mean pooling also moved. Read this as "the entropy bug was blocking
everything, and pooling mode may matter less than finding 1 concluded" until the held-out
`evaluate-model` numbers confirm it's real generalization and not train-set overfitting -- these
are training-loop metrics, not the test-split ROC-AUC, and every prior "accuracy" mistake in
this document came from exactly that conflation. `tm` still has not left its uniform baseline in
any of the four; the label win looks independent of the TM branch so far.

**`evaluate-model` landed -- here is the real, held-out comparison** (152 test repertoires,
majority baseline accuracy 0.539, burden-score baseline ROC-AUC **0.781**):

| run | `theta_features` test ROC-AUC | test PR-AUC | test acc | train ROC-AUC | `quantile_features` test ROC-AUC | train ROC-AUC |
|---|---|---|---|---|---|---|
| `v7_tmheavy` (tm_coef 0.85) | **0.686** | 0.709 | 0.645 | 0.755 | 0.565 | 0.808 |
| `v7_attn_dualentropy` (flagship) | 0.678 | 0.660 | 0.625 | 0.759 | 0.522 | 0.795 |
| `v7_wide_topics` (60 topics) | 0.643 | 0.633 | 0.586 | 0.759 | 0.440 | 0.820 |
| `ablation_meanpool` | 0.627 | 0.592 | 0.586 | 0.743 | 0.442 | 0.870 |

This is a real result, not train-set noise: this is the **first time in this document's history
that any current-code Emerson run has scored above both chance and the majority baseline on the
held-out 152**, via `theta_features` (classifying repertoires directly from their inferred Θ).
All four land 0.10-0.15 AUC below the 0.781 burden-score baseline -- a real gap, not a win --
but a categorical change from "every prior run was indistinguishable from a shuffled control"
(finding 1) to "generalizes, just not as well as Fisher-exact clonotype counting yet."

Two things to flag, not celebrate past:

1. **`quantile_features` (the per-sequence signal-intensity summary) does not generalize at
   all** -- train ROC-AUC 0.75-0.87, test ROC-AUC 0.44-0.57, i.e. at or *below* chance on two of
   four runs despite training AUC in the 0.80s. `theta_features`' train/test gap is much smaller
   (0.74-0.76 -> 0.63-0.69). The discriminative signal these models learned lives in the
   amortized Θ inference path, not in the per-sequence label head -- expected given all four use
   `label_input: repertoire` (the label loss is computed on Θ directly, never on the per-sequence
   quantiles), but worth stating plainly: **the sequence-ranking use case (finding the actual
   CMV-associated CDR3s, Part 3 item 2) is not yet supported by any of these checkpoints.**
   Ranking candidate sequences would need either `label_input: sequence` (which this session's
   `ablation_meanpool`-style controls, not yet re-tested with the corrected coefficient, existed
   to check) or a different read-out of `theta_features`-trained models.
2. **`v7_tmheavy` (tm_likelihood_coef 0.85) is the best of the four**, not the flagship's 0.7 --
   consistent with finding 2 ("raising tm_likelihood_coef delays collapse") now showing a direct
   downstream benefit in held-out AUC rather than just a training-loop entropy trace. Worth a
   coefficient sweep past 0.85 as a next step, alongside the usage/seed/topic-count sweeps
   already running.

### Update, later same day: the seed/usage/topic-count sweep, and a tm_likelihood_coef sweep

Of the four follow-ups launched after the eval landed, three finished (GPUs 2-5 freed):

- **`v7_usage70`** (`topic_usage_coef: 7.0`): best epoch **9** (vs. `usage=5.0`'s 84),
  `label_acc` 0.551 (vs. 0.606). Converges faster to a *weaker* solution -- 7.0 looks like too
  much anti-collapse pressure, cutting off the slower climb that 5.0 got to ride. Not yet
  evaluated against the test split; `v7_seed1` (reproducibility) and `v7_usage30` /
  `v7_narrow_topics` are still training.
- **All three synthetic `*_v2` reruns** (raised to `patience: 60, n_epochs: 300`) **still
  collapsed** -- best epoch 10-64, `label_acc` pinned at exactly 0.5, `H_theta`/`H_usage` both
  near 0 again (0.001-0.009). Raising patience did not fix it: whatever keeps this small,
  simple dataset collapsing is a different mechanism than what raising `topic_usage_coef` fixed
  on Emerson, and needs its own look rather than more patience. Not chased further this round --
  the tm_likelihood_coef sweep below took priority per direct request.

**tm_likelihood_coef sweep launched, GPUs 2-5 and 7** (all `v7_attn_dualentropy`'s base config,
i.e. `topic_usage_coef: 5.0`, `theta_pooling: attention`, `label_input: repertoire`, varying only
`tm_likelihood_coef`, filling the gap between the flagship's 0.7 and `v7_tmheavy`'s 0.85 that
won, and pushing past it): `config_v7_tm075.yaml` (0.75), `config_v7_tm080.yaml` (0.80),
`config_v7_tm090.yaml` (0.90), `config_v7_tm095.yaml` (0.95), `config_v7_tm099.yaml` (0.99, right
at the edge where `label_likelihood_coef = 1 - tm_coef` -> 0.01 and the label term nearly
vanishes -- included deliberately to see where the win reverses). No results yet.

**Current allocation**: `v7_seed1` (0), `v7_usage30` (1), `v7_tm075` (2), `v7_tm080` (3),
`v7_tm090` (4), `v7_tm095` (5), `v7_narrow_topics` (6), `v7_tm099` (7). All 8 GPUs training;
nothing reserved for eval right now -- next `evaluate-model` batch waits for a free slot or the
next finisher.

### Update, 2026-09-09: a re-seed alone moved test AUC as much as any coefficient did

Leaderboard after evaluating everything finished so far (`theta_features` test ROC-AUC, 152
held-out repertoires, burden-score baseline 0.781):

| run | config | test ROC-AUC |
|---|---|---|
| `v7_seed1` | flagship config, `seed: 1` (vs. the original `239`) | **0.7145** |
| `v7_usage30` | `topic_usage_coef: 3.0` | 0.7096 |
| `v7_tm095` | `tm_likelihood_coef: 0.95` | 0.7008 |
| `v7_tmheavy` | `tm_likelihood_coef: 0.85` | 0.6858 |
| `v7_usage70` | `topic_usage_coef: 7.0` | 0.6864 |
| `v7_narrow_topics` | 15 topics | 0.6670 |
| `v7_attn_dualentropy` | flagship, `seed: 239` | 0.6780 |
| `v7_wide_topics` | 60 topics | 0.6433 |
| `ablation_meanpool` | mean pooling, `label_input: sequence` | 0.6274 |

**The methodological problem this exposes**: `v7_seed1` is the *identical config* to
`v7_attn_dualentropy`, differing only in random seed, and moved test AUC by **+0.036** --
larger than the gap between most adjacent points in the coefficient sweep. Every coefficient
value above has been tried with exactly one seed. We cannot yet tell how much of this
leaderboard is the coefficient and how much is seed noise of similar magnitude. Do not read the
ranking above as settled.

**Response**: launched `seed: 2` replicates of the four strongest configs (flagship, `usage30`,
`tm095`, `tm090` -- the last picked from its in-training-loop numbers, being the best of the
still-unevaluated tm-sweep runs, before its own eval had landed) on GPUs 4-7, alongside
evaluating the four remaining tm-sweep checkpoints (`tm075/080/090/099`) on GPUs 0-3. Once these
land, each of the four leading configs will have two seeds -- still not a real distribution, but
enough to say whether a config's edge exceeds single-seed noise or not.

**Multi-seed results so far**: flagship now has 3 seeds -- 239->0.678, 1->0.7145, 2->0.6474
(mean 0.680, std 0.034, range 0.067). `usage30` has 2 confirmed -- 239->0.710, 3->0.6685 (a
3rd, seed 2, collapsed the same way as the other two and is evaluating now). **The two
configs' seed ranges now visibly overlap** (flagship 0.647-0.715, `usage30` 0.669-0.710) --
there is no longer a clean separation between them once seed spread is accounted for, and the
apparent "usage30 is the best config" conclusion from the single-seed leaderboard does not
survive this. `v7_usage30_seed3` collapsed to the same near-zero `H_theta`/`H_usage` as the
original `usage30` run, at a best epoch (33 vs. 32) with *nearly identical* training-loop stats
to the original's best epoch -- yet the two checkpoints' test AUCs differ by 0.04. Training-loop
numbers at the best epoch do not reliably predict test AUC even when they look the same across
runs; there is no shortcut around actually running `evaluate-model`.

### Update, 2026-09-09: can the TM likelihood be fit at all? Yes -- the flat `tm` was a step-budget artifact

Prompted by a direct question: the label/theta-features result generalizes, but `tm` never
leaves its uniform baseline anywhere -- so is the topic-model half of this architecture doing
anything, or is Theta's generalization coming entirely from the entropy regularizers and (where
applicable) direct label supervision? Designed a diagnostic to isolate the TM likelihood from
every confound at once: `theta_mode="free"` (removes amortized-inference difficulty),
`theta_entropy_coef`/`topic_usage_coef`/`topic_decorrelation_coef` all zeroed and
`tm_likelihood_coef: 1.0` (removes competition from every other loss term), small
`--repertoire_slice` (removes cross-repertoire-competition scale), and (a second variant)
`vae_coef: 0.0` (removes even the VAE branch's competition for the shared encoder).

**First attempt was confounded by a mistake in the diagnostic itself, not the objective**:
`steps/epoch = n_batches_per_repertoire * n_repertoire_groups`, and with only 4-16 repertoires
and the default `n_batches_per_repertoire: 4`, epoch count and step count decoupled badly --
`overfit_free_4` ran 393 "epochs" but only ~1,572 actual optimizer steps, nowhere near enough to
reach an extreme corner of parameter space (near-one-hot Theta, large-magnitude phi) under
`grad_clip: 1.0`. All variants showed `tm` stuck within 0.001-0.03 nats of baseline, mirroring
every full-scale result in this document -- but for the wrong reason.

**Refitted with `n_batches_per_repertoire` raised to hold steps/epoch at ~200 regardless of
repertoire count** (`n_epochs: 100, patience: 40`, ~20,000-step budget). Results, at each run's
actual best epoch (not the last logged line, which reflects post-optimum drift once early
stopping's patience window has passed):

| run | val `tm` reduction | train `tm` reduction | best epoch |
|---|---|---|---|
| `free_16` (mean pooling would-be-amortized w/ theta_mode=free) | **+0.159** | +0.337 | 46 |
| `free_vae0_16` (+ vae_coef=0) | +0.149 | +0.295 | 32 |
| `amortized_vae0_16` | +0.087 | +0.098 | 2 |
| `amortized_16` | +0.071 | +0.224 | 10 |
| `free_4` | +0.060 | +0.181 | 5 |
| `amortized_4` | -0.011 (flat) | +0.260 | 9 |

Five of six show genuine reductions -- train reductions of 0.18-0.34 nats, **an order of
magnitude more than any full-scale run this session has shown** (0.001-0.03 nats), and well past
the >=0.052-nat ceiling FINDINGS.md previously estimated from a supervised V+J+length classifier
on 4 repertoires. **The TM likelihood is not fundamentally broken or unfittable -- the flat `tm`
seen everywhere else this session is substantially a step-budget / competition artifact, not
proof the objective can't work.** `amortized_4` is the one holdout, still flat even at its best
checkpoint; with only one repertoire-group and a maximally repetitive batch each step, I'd
chalk this up to that specific tiny setup rather than a generalizable finding.

**But scale still matters, separately from step count.** The two full-scale variants
(`theta_mode` free/amortized, all 597 train repertoires, same zeroed regularizers and
`tm_coef=1.0`) had, after ~27,400 steps each -- more steps than any 16-repertoire run needed to
show real movement -- `tm` still sitting at baseline (10.39-10.40). Discriminating among 597
repertoires is intrinsically a harder task than among 16; matching this session's small-scale
reductions at full scale likely needs substantially more than 27,000 steps, not just "enough."
Both still running; will report when they move or hit patience.

**Implication for the earlier open question** (does Theta's generalization come from the
entropy regularizers/label supervision alone, or does the TM likelihood contribute too): this
doesn't settle that -- it establishes the TM objective *can* be fit given enough dedicated
optimization pressure, not that it *is* being fit in the configs that actually generalize (where
it's competing against the label loss and entropy terms, and operating at full 597-repertoire
scale). The natural next check, not yet run: take a config that generalizes well (e.g.
`v7_seed1`) and see whether a much larger step budget or a schedule that gives TM an early,
unopposed head start changes its `tm` trajectory or the resulting `theta_features` test AUC.

### Update, 2026-09-10: the warm-start schedule resolves the open question -- competition is real, and scale is a separate, harder blocker

Added a coefficient-annealing mechanism (`airrtm/training/train.py::_linear_schedule`, threaded
through `cli/train.py` as `tm_likelihood_coef_start` / `theta_entropy_coef_start` /
`topic_usage_coef_start` / `coef_anneal_epochs`, mirroring the existing `tau_start` pattern) to
give the TM likelihood an unopposed warm-up window before ramping in the label loss and entropy
regularisers -- testing whether competition, not just insufficient step count, is what keeps
`tm` flat in configs that actually generalise. Six runs: 16-repertoire and full-597-repertoire
scale, x short/medium/long warm-up (~1,000-9,000 steps). Full per-epoch traces for the two
"long" variants settle the question cleanly:

**`warmstart_16_long`** (30-epoch warm-up, 16 repertoires): `tm` dips modestly during warm-up
(10.345 at epoch 5, vs. 10.397 uniform), but once the label loss and entropy terms ramp in, `tm`
does not merely flatten -- it climbs **above** baseline (10.46 -> 10.62 by epoch 20-90), while
`label_acc` climbs to a strong 0.625-0.75. The model actively trades TM fit *away* in exchange
for label accuracy once both are available: real, direct competition for the same shared
parameters, not just a bandwidth-starvation story.

**`warmstart_full_long`** (15-epoch warm-up, all 597 repertoires): `tm` stays almost exactly
flat the entire run (10.39-10.41, never reversing), while `label_acc` reaches a healthy
0.58-0.65 and the entropy signature looks exactly like the good real runs (`H_theta` down to
~0.17-0.38, `H_usage` up to ~2.8-3.0). At full scale the interference is neutral rather than
adversarial -- but TM still never improves.

**Combined with `overfit_amortized_full`** (the permanently-unopposed full-scale diagnostic
from the day before, left running): after 114 epochs / ~68,000 steps of zero competition ever,
`tm` still never moved (best epoch 13, delta +0.007; last epoch flat). So at real scale, this
objective yields to neither a larger step budget nor a competition-free schedule.

**This settles the open question from finding "can TM be fit at all"**: yes, in a small,
controlled setting (confirmed 2026-09-09), but that result does not transfer to the scale and
configuration that actually matters. Two independent things are true simultaneously: (1)
competition for the shared encoder/Theta between the TM likelihood and the label+entropy terms
is real -- at small scale it actively reverses TM's gains, not just dilutes them; (2) scale
itself is a separate, harder blocker that neither fixing the competition nor throwing more steps
at it resolves. **Best-supported conclusion across both days of this investigation: Theta's
generalization to held-out repertoires (0.62-0.71 test AUC across this session's runs) comes
from the entropy regularisers and direct label supervision, not from a working topic model in
the generative sense this architecture is named for.** Evaluating the full-scale warm-start
checkpoints against the held-out split now, to check whether the schedule changed downstream
test AUC even though it didn't change `tm` -- results pending.

**Results landed**: `warmstart_full_medium` (~4,800-step warm-up) scored **0.740 test AUC** --
a new session best, beating `v7_seed1`'s 0.7145. But `warmstart_full_short` (~1,800 steps) and
`warmstart_full_long` (~8,900 steps) both landed at ~0.634, and `warmstart_16_long` at 0.641 --
a non-monotonic "sweet spot at medium" pattern. Given this session has already shown seed-alone
variance of 0.647-0.715 on the *identical* flagship config, and a `tm_likelihood_coef` sweep
that bounced non-monotonically between 0.58 and 0.71, **one run per warm-up duration is not
enough to credit 8 epochs of warm-up specifically for the 0.740** -- this could easily be
favorable seed noise rather than the schedule being causally better at that duration. Recorded
as a new best-observed checkpoint, not as evidence the warm-up duration itself is optimal at
8 epochs; would need multiple seeds per duration to tell those apart.

### Update, 2026-09-10: `warmstart_full_medium` may have real sequence-ranking signal -- unconfirmed, seed replicates running

Ran the same Fisher-list check (both phi and the attention*value pathway, all 70 positive test
repertoires, `pr_auc`-corrected) against `warmstart_full_medium` (test AUC 0.740, the current
best). Result, contrasted with every prior checkpoint checked this way (`v7_seed1`,
`v7_tmheavy`, `v7_usage30`, all ~0.86-1.15x chance on both pathways):

| pathway | `pr_auc_over_chance` | enrichment @ 0.001% / 0.01% / 0.1% / 1% / 10% |
|---|---|---|
| phi | 0.86x (still noise, as always) | 0 / 0 / 0 / 0.99x / 0.91x |
| **attention*value** | **16.9x** | **166x / 66x / 43x / 6.1x / 1.4x** |

This is qualitatively different from anything seen before -- a smooth decay from 166x at the
extreme top down to 1.4x at 10%, the shape of genuine ranking signal rather than a lucky spike
in one slice (contrast the earlier `v7_seed1` check: 0.94-1.15x everywhere, indistinguishable
from noise). If real, this checkpoint may have learned actual sequence-level attribution, not
just repertoire classification -- the thing this whole architecture is supposed to do and which
nothing had shown any sign of until now.

**But this checkpoint's own `tm` trajectory looks unremarkable** -- flat at 10.39-10.42
throughout, the same "neutral, never improves" pattern as `warmstart_full_long`, and
`label_acc` (~0.55-0.57) is not obviously higher than other checkpoints either. Nothing in the
aggregate loss curves predicts this result, which raises the obvious question: is this a real
property of the medium-length warm-up, or did this one seed happen to concentrate attention on
a handful of genuinely discriminative sequences by chance -- exactly the kind of one-off this
document has repeatedly found reason to distrust (seed spread of 0.647-0.715 on the identical
flagship config, a non-monotonic tm-coefficient sweep, etc.)? `warmstart_full_medium_seed1` and
`_seed2` are already running (launched for the classification-AUC question) -- running this same
check against them, once they finish, will directly settle whether the signal reproduces.

**Per-repertoire breakdown (not pooled) clarifies what the 16.9x actually means.** For each of
the 70 positive test repertoires, ranked its own sequences and checked its own top fraction
against its own selected clonotypes (not the global pool):

| fraction | candidate-list size | mean enrichment | reps with >=1 hit | expected by chance |
|---|---|---|---|---|
| 0.001% | ~2 | 270x | 1/70 | 0.01 |
| 0.01% | ~20 | 55x | 3/70 | 0.07 |
| 0.1% | ~196 | 33x | **17/70** | **0.72** |
| 1% | ~1,962 | 6.3x | **30/70** | **6.9** |
| 10% | ~19,622 | 1.1x | 48/70 | 45.2 |

Median precision is 0 at every fraction up to 1% -- most repertoires get zero hits in their own
ranked list. The pooled 166x/66x enrichment at the finest fractions was driven by a handful of
exceptional repertoires, not a general phenomenon. But it is not pure noise either: at 0.1% and
1%, the number of repertoires getting >=1 hit (17, 30) is far beyond what chance alone would
produce (0.72, 6.9) -- real signal in a meaningful minority of patients (24-43%), fading to
nothing by 10% (48 observed vs. 45.2 expected). **Best-supported read: for roughly a quarter to
two-fifths of CMV+ test repertoires, this checkpoint's top ~200-2,000 ranked sequences contain a
real, non-chance enrichment of the known discriminative clonotypes; for the rest, nothing.**
Patchy, not universal -- but still categorically more than any other checkpoint has shown.
Still contingent on the seed-replication check above.

**`overfit_bigenc_full` finished**: a 2x-deeper, 2x-wider transformer (depth 8, attention_dim
128, 16 heads -- had to cut the batch to 1024 seqs/repertoire to fit memory) run through the
same full-scale unopposed-TM recipe. Best epoch 7: `tm` reduction +0.009 nats against its own
(batch-size-adjusted) uniform baseline -- essentially flat, the same magnitude as every default-
capacity full-scale run. **Encoder/transformer capacity is not the bottleneck**; the blocker is
specifically about the number of repertoires being discriminated among, not the sequence
encoder's expressiveness. `overfit_moretopics_full` (90 topics instead of 30, same recipe)
finished the same day: best epoch 11, `tm` delta +0.0094 -- essentially identical to
`overfit_bigenc_full`'s +0.009. **Both capacity axes tested (transformer depth/width, topic
count) come back negative.** Repertoire-count scale, not model capacity in either dimension, is
the blocker for fitting the TM likelihood at full scale.

**Operational note**: this session's background job-completion watchers (a `until ps ...; do
sleep N; done` loop under `run_in_background`) started getting killed by the harness's own
background-task memory limit, repeatedly, regardless of poll interval (30s and 120s both got
killed) or watch scope (a broad multi-PID loop and a single-PID wait both got killed). Host RAM
was never actually under pressure (1.7 of 1.8 TiB free throughout). Stopped relying on
autonomous watchers for the rest of this session; status now checked on request instead.

### Update, 2026-09-10 (evening): user caveat on the Fisher-list ground truth, seed replicates landed

**Methodological caveat, worth keeping in mind for every check above and below**: the
Fisher-exact selected clonotype list is a p-value threshold on exact clonotype identity, not
biological ground truth. It has real false negatives (true signal clonotypes too rare to reach
significance, or near-neighbors from convergent recombination -- different individuals arriving
at biochemically similar but non-identical CDR3s for the same epitope, well documented in TCR
immunology) and real false positives (multiple-testing noise). A "miss" against this list is not
proof the model found nothing real; a "hit" is not proof it found something biologically
meaningful. Decided *not* to build a near-neighbor/edit-distance expansion of the list to
compensate (discussed and declined) -- just hold this caveat against every precision/enrichment
number in this document rather than over-read small differences as definitive.

**The two `warmstart_full_medium` seed replicates (seed 1, seed 2) and the duration-sweep
refinement (6, 12 epochs) all finished.** `evaluate-model` and the per-repertoire ranking check
are running now on all four -- results pending. This is the check that actually matters: does
the 0.740 test AUC and the 16.9x/patchy-but-real sequence-ranking signal reproduce across seeds,
or was `warmstart_full_medium` (seed 239) a lucky draw? Given this session's already-documented
seed variance (flagship: 0.647-0.715), a single additional data point either way will not be
conclusive, but a large drop back to the ~0.63-0.68 range seen in `warmstart_full_short`/`_long`
would be the more likely outcome if this is mostly noise, matching the pattern seen everywhere
else this session.

**Reproducibility result landed**: ran the per-repertoire ranking check against
`warmstart_full_medium_seed1` and `_seed2`.

| seed | enrichment @ 0.001%/0.01%/0.1%/1%/10% | reps with >=1 hit |
|---|---|---|
| 239 (original) | 270x / 55x / 33x / 6.3x / 1.1x | 1/3/17/30/48 |
| 1 | 406x / 39x / 13x / 6.5x / 1.6x | 2/2/9/26/46 |
| 2 | 0x / 0x / 1.7x / 0.66x / 0.66x | 0/0/1/4/23 (~chance) |

**Seed 1 reproduces the pattern closely; seed 2 shows nothing at all.** This is real -- not a
one-off fluke of a single training run -- but seed-dependent: 2 of 3 seeds show it, 1 does not.
The warm-start schedule does not *guarantee* this sequence-ranking signal, it makes it possible
some of the time. `evaluate-model` on all four (seed1/seed2/dur6/dur12) still running for the
matching classification-AUC question -- caveat from the user: the Fisher-exact list itself is
not ground truth (real false negatives from convergent recombination, real false positives from
multiple-testing noise), so read every number in this section as measured against an imperfect
proxy, not biological certainty.

### Correction, same day: the earlier "sequence ranking is chance" claim was about the wrong layer

The Fisher-list check two entries up scored `phi` (`seq_topic_logits_ST`, from
`latent_space_to_topic_proportions_layer`, read via `predict_signal_intensity` /
`quantile_features`) -- a layer whose only real training signal is the inert TM likelihood.
It is a *different*, independently-parameterized layer from what actually builds Theta:
`TopicAttentionPooling.value` / `.attend` / `.gate` / `.score`, trained by every term that
shapes Theta (entropy, usage, decorrelation, and the label loss directly in
`label_input="repertoire"` mode). Re-ran the same Fisher-list check on `v7_seed1` (best
checkpoint, CPU, restricted to the 70 positive test repertoires holding >=1 selected
clonotype), scoring `attention_ST[:, :n_topics_signal] * values_ST[:, :n_topics_signal]`
(summed over signal topics) instead of phi:

| | phi (`quantile_features`) | `attention * value` (this check) |
|---|---|---|
| overall per-sequence ROC-AUC | 0.46-0.60 | **0.341** (below chance) |
| enrichment @ top 0.01% (pooled) | not tested that fine | **49.7x** |
| enrichment @ top 0.1% | 0.5-1.9x at every fraction tried | **5.0x** |
| enrichment @ top 1% / top 10% | 0.5-1.6x / 0.5-1.4x | 1.16x / 0.71x |
| per-repertoire top-k precision, all 70 reps | 0 | 0 |

Real, if narrow: a small subset of true signal clonotypes gets sharply concentrated at the
extreme top of the pooled score distribution (top 0.01-0.1%) -- something phi never showed at
any fraction. But overall AUC *below* chance means most of the 603 true positives in this
subset score no better than background, and per-repertoire top-k precision is still exactly
zero everywhere (k is small here, 1-20, capped at each bag's own true-positive count -- a
stricter test than a fixed top-100 list, so this doesn't necessarily mean the top ~100
candidates are worthless, just that they don't align with the literal true-positive count).
**Net: this pathway carries real per-sequence signal that phi does not, but it is concentrated
in a handful of cases and does not add up to a usable candidate-discovery tool.** The earlier
"sequence ranking is chance across the board" conclusion was too strong -- it was accurate for
phi, not for the pathway that actually explains Theta's generalization.

**Further correction, same day: ROC-AUC was the wrong metric for this, and the "real signal"
conclusion above does not survive PR-AUC.** At a witness rate this extreme (1-in-20,000 to
1-in-38,000), ROC-AUC is close to uninformative -- it aggregates over mostly signal-vs-noise
pairs far from the decision-relevant region and barely moves either direction. `pr_auc`
(average precision, chance-calibrated to the witness rate itself) is the metric that matters
here, and `evaluation/metrics.py::signal_enrichment` never computed it despite importing
`average_precision_score` -- fixed now (`pr_auc`, `pr_auc_over_chance` added; 55 tests still
pass). Re-ran both pathways on the identical 70-repertoire subset for a clean paired comparison:

| | attention*value | phi |
|---|---|---|
| roc_auc | 0.341 | 0.497 |
| **pr_auc_over_chance** | **0.945x** | **1.11x** |
| enrichment @ top 0.01% / 0.1% | 49.7x / 5.0x | 0 / 0 |
| enrichment @ top 1% / 10% | 1.16x / 0.71x | 1.66x / 1.28x |

Both pathways sit within noise of random guessing by the metric that actually applies here. The
attention pathway's 49.7x top-fraction enrichment corresponds to only ~3 sequences landing in a
~1,370-candidate slice (out of 603 true positives pooled across 13.7M candidates) -- a Poisson
null puts p~7e-5 on getting >=3 there by chance alone (so not pure noise at that one operating
point), but the absolute yield is too small to move the overall precision-recall curve, which is
exactly why `pr_auc` correctly reports ~chance overall. By this metric phi is *very slightly*
ahead (1.11x vs. 0.94x) -- the opposite of what the raw enrichment numbers suggested.
**Corrected conclusion: neither per-sequence pathway shows real ranking signal by the
appropriate metric.** The repertoire-level `theta_features` numbers (0.62-0.71 test ROC-AUC)
are unaffected by this -- that is a roughly-balanced (44-56%) classification problem where
ROC-AUC is the right call; this issue is specific to sequence-level ranking.

**Narrowed further, same day: restricting to correctly-classified positive repertoires barely
moves either pathway.** Of the 70 true-positive test repertoires, 44 were correctly classified
by `v7_seed1`'s `theta_features` classifier (reproduced by reloading the saved
`theta_train.npy`/`theta_test.npy` and refitting the same `RandomForestClassifier`). Re-scored
both pathways on just those 44 (405 selected-clonotype sequences, witness rate 4.70e-5):

| | all 70 positive reps | just the 44 correctly-classified |
|---|---|---|
| attention*value `pr_auc_over_chance` | 0.945x | 1.15x |
| phi `pr_auc_over_chance` | 1.11x | 1.10x |
| attention*value roc_auc | 0.341 | 0.336 (still below chance) |
| precision@100, either pathway | 0 | 0 |

A small bump for the attention pathway, no movement at all for phi, and precision@100 still
exactly zero for both. **This is itself informative**: if the 44 correct classifications came
from the model genuinely detecting those repertoires' individual signal clonotypes, restricting
to them should have produced a much larger jump in per-sequence ranking quality. It didn't.
Best-supported read now: correct repertoire classification is driven by something more diffuse
in Theta's overall shape (entropy regularizers, and direct label supervision in
`label_input="repertoire"` mode) rather than genuine sequence-level attribution -- consistent
with, and now more directly evidenced than, the mechanistic argument made earlier the same day.

**Also confirmed dead end**: combining `topic_usage_coef: 3.0` with a higher `tm_likelihood_coef`
(`v7_usage30_tm085`, `v7_usage30_tm090` -- hypothesis: tm-heavy delays collapse the way it did at
`topic_usage_coef: 5.0`, so it might rescue `usage30` from its early collapse) **did not work**:
both collapsed (`H_theta`/`H_usage` ~0.0005-0.0009) even faster than `usage30` alone (best epoch
14 and 47 vs. 32). `topic_usage_coef` and `tm_likelihood_coef` do not substitute for each other;
don't retry this combination without a new reason to expect a different result.

**Operational note**: launching 4 training jobs at once without `OMP_NUM_THREADS`/
`MKL_NUM_THREADS` capped hit the same CPU-thread-thrashing stall from 2026-09-08 again (168
threads/process x 4, none reached the GPU for minutes) -- this keeps recurring because it is easy
to forget on a quick relaunch. Always cap threads (16 was used throughout) whenever launching
more than 2-3 concurrent jobs on this box, no exceptions.

### Update, later same day: the tm_likelihood_coef sweep is not monotonic -- variance, not signal

The remaining four sweep checkpoints evaluated. **All seven points use the same seed (239)** --
this is not the seed-noise issue above, it is separate, additional noise from the training
dynamics themselves at fixed seed:

| `tm_likelihood_coef` | test ROC-AUC | run |
|---|---|---|
| 0.70 | 0.678 | `v7_attn_dualentropy` (flagship) |
| 0.75 | **0.710** | `v7_tm075` |
| 0.80 | 0.642 | `v7_tm080` |
| 0.85 | 0.686 | `v7_tmheavy` |
| 0.90 | **0.581** | `v7_tm090` (worst of all nine runs this session, despite the *best* in-training-loop `label_acc` of the four un-evaluated sweep points at the time it was picked for a seed-2 replicate -- that choice looks bad in hindsight) |
| 0.95 | 0.701 | `v7_tm095` |
| 0.99 | 0.618 | `v7_tm099` |

No trend survives this: 0.75 and 0.95 are the two best points in the entire sweep, 0.80/0.90/0.99
in between are mediocre-to-worst, 0.90 is the single worst result of the session. The
`r~0.75` correlation computed earlier in this document (tm-loss reduction vs. test AUC, n=6,
before this batch existed) should be read as provisional at best -- it no longer looks like it
would survive adding these seven points, though it was computed on a different, overlapping set
of runs and hasn't been recomputed against this table.

**Read this as**: training-run-to-training-run variance at a *fixed* seed and a small coefficient
perturbation is large enough (0.58-0.71 AUC swing) to swamp whatever the coefficient's real
effect is. Combined with the seed-to-seed gap above (+0.036 at fixed config), there are now two
independent, similarly-sized sources of noise in every single-run comparison made so far in this
document. **Nothing in this session's leaderboard should be treated as a settled ranking** until
multiple seeds per config exist. The `seed: 2` replicates of the four strongest configs
(flagship, `usage30`, `tm095`, `tm090`) are running now for exactly this reason -- `tm090`'s
seed-2 run in particular will be informative given how badly its seed-239 run did.

**Synthetic runs did not benefit the same way** -- all three still early-stopped almost
immediately (best epoch 0-1, `label_acc` pinned at 0.5), though `H_theta`/`H_usage` were no
longer fully collapsed to ~0 (e.g. S1_w0.001 ended at 0.15/0.13, not 1e-5 as before the fix) --
partial, not zero, improvement. Most likely explanation: `patience: 15` / `n_epochs: 100` was
tuned for the broken coefficient and cuts off far too early on a noisy 80-repertoire dataset
that only gets ~80 steps/epoch (vs. Emerson's 604) -- relaunched as `*_v2` with
`patience: 60, n_epochs: 300`, unconfirmed yet.

### Update, same day: direct check -- ranking sequences does not recover the Fisher clonotypes

Asked and checked directly rather than argued: does the model's per-sequence ranking (the
`quantile_features` pathway, i.e. `predict_signal_intensity`) recover the Fisher-exact selected
clonotypes the burden-score baseline is built on -- including for `v7_usage30`, the best
repertoire classifier found so far (test AUC 0.710)? Script:
`fisher_overlap_check.py` (scratchpad, not committed -- rerun by refitting
`BurdenScoreClassifier` on the training split and hashing test sequences the same way).

**No.** All three models checked (`v7_attn_dualentropy`, `v7_tmheavy`, `v7_usage30`):

| model | repertoire test AUC (Θ) | per-sequence AUC (is this sequence Fisher-selected?) | precision@100/1000/5000, pooled across 28.9M test sequences |
|---|---|---|---|
| `v7_attn_dualentropy` | 0.678 | 0.465 | 0 / 0 / 0 |
| `v7_tmheavy` | 0.686 | 0.485 | 0 / 0 / 0 |
| `v7_usage30` | 0.710 (best) | 0.604 | 0 / 0 / 0 |

Coarser `signal_enrichment` (top 1%/10% of all test sequences) is 0.5-1.9x over the 2.6e-5
background rate -- noise, and *below* 1x (worse than random) for `v7_tmheavy` at the top 1%.
Zero of the top-100 ranked sequences within any of the 70 positive test repertoires that
actually contain a selected clonotype were hits, for any of the three models.

One methodology note: used the code's default `p_threshold=1e-4` (126 selected clonotypes),
not the `1e-3` this document's reference table used (482) -- a real difference, but the margin
here is wide enough (exact zero at every k, near-chance AUC) that it does not change the
conclusion. Confirms the architectural read from earlier the same day: `usage30`'s better
repertoire classification comes with a *slightly* less-chance per-sequence AUC (0.604 vs.
0.465-0.485) -- consistent with whatever helped Θ generalize leaking a little into φ -- but
nowhere near enough to produce a usable candidate list on a 1-in-38,500 needle-in-haystack
problem. **None of these checkpoints support sequence-level CMV-TCR discovery**; that remains
blocked on `label_input="sequence"` actually working (untested with the corrected coefficient)
or a redesigned per-sequence objective.

**Relaunched 2026-09-08, all seven GPUs (0-6) plus GPU 7 running `evaluate-model`:**

| GPU | run | tests |
|---|---|---|
| 0 | `emerson/model/v7_seed1` | same as flagship, `seed: 1` -- is the label_acc win reproducible or one lucky init? |
| 1 | `emerson/model/v7_usage30` | `topic_usage_coef: 3.0` (below the working 5.0) |
| 2 | `emerson/model/v7_usage70` | `topic_usage_coef: 7.0` (above the working 5.0) |
| 6 | `emerson/model/v7_narrow_topics` | 15 topics (5+10) instead of 30 -- complements `v7_wide_topics`' 60 |
| 3 | `synthetic/model/S1_w0.005_v2` | same data, `patience: 60, n_epochs: 300` |
| 4 | `synthetic/model/S1_w0.001_v2` | same |
| 5 | `synthetic/model/S2_w0.001_v2` | same |
| 7 | `evaluate-model` on all four finished Emerson checkpoints | the number that actually matters: test-split ROC-AUC vs. the 0.781 burden-score baseline |

---

## Continuing on another machine (historical — describes the old A100 box)

This section predates the 2026-09-07 entry above and the code has since been committed
(`143ab60`/`d713a2c`, this machine's `.venv` + `pip install -e .` confirms it installs and all
55 tests pass); the "not committed" warning below no longer applies. Left as-is for the
hardware note and the gene-column gotcha, both still true. `processed_data/` (no V/J) does not
exist on the current 8x H100 machine — only `processed_data_vj` was carried over.

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

---

## Update, 2026-09-11/12: `topic_usage_coef` collapse floor, `theta_features_cv`, and a paired re-look at whether TM/phi actually helps

**`topic_usage_coef` below 3.0 collapses topics, with or without the warm-start schedule, and
collapse doesn't predict AUC.** Nine runs at `topic_usage_coef` in {1.0, 2.0, 3.0} (some with
the warm-start schedule, some without) landed:

| run | `topic_usage_coef` | warm-start | H_theta/H_usage (final) | test theta ROC-AUC |
|---|---|---|---|---|
| `warmstart_usage3_seed239` | 3.0 | yes | 0.03/1.13 (healthy) | 0.684 |
| `warmstart_usage3_seed1` | 3.0 | yes | 0.06/1.23 (healthy) | 0.609 |
| `warmstart_usage3_seed2` | 3.0 | yes | 0.001/0.001 (**collapsed**) | 0.708 |
| `warmstart_usage2` | 2.0 | yes | 0.002/0.009 (**collapsed**) | 0.555 |
| `usage_lower2` | 2.0 | no | 0.0004/0.0004 (**collapsed**) | 0.722 |
| `usage_lower1` | 1.0 | no | 0.0001/0.0001 (**collapsed**) | 0.704 |
| `warmstart_medium_seed3` (control, 5.0) | 5.0 | yes | 0.19/3.02 (healthy) | 0.572 |
| `warmstart_medium_seed4` (control, 5.0) | 5.0 | yes | 0.19/2.95 (healthy) | 0.738 |

Two things worth internalizing: (1) 3.0 is not a safe value even with warm-start -- 1 of 3
seeds collapsed anyway, so the earlier `v7_usage30` result (0.710 test AUC, healthy) was not
guaranteed to reproduce, it happened not to collapse; (2) **collapse does not predict AUC** --
collapsed runs (0.56-0.72) score all across the same range as healthy ones (0.57-0.74), overlapping
in the middle. `topic_usage_coef=5.0` (the flagship default) remains the only value confirmed
never to collapse across 5 seeds, but that same 5-seed set spans test AUC 0.572-0.740 (std
0.072) on an *unchanged* config -- **seed variance alone is larger than every effect this
sweep, the tm-coefficient sweep, or the warm-start-duration sweep has produced.** Conclusion:
stopped tuning `topic_usage_coef` further; it's not worth the collapse risk for an effect this
size relative to seed noise.

**Added `theta_features_cv`** (`airrtm/cli/evaluate.py`, mirroring the existing
`quantile_features_cv`): 5-fold `cross_validated_report` over Θ features, using the same
`StratifiedKFold` machinery already in `airrtm/evaluation/metrics.py`. Recomputed for the 5
`usage=5.0`/warm-start-medium seeds directly from already-saved `theta_train.npy`/
`theta_test.npy` (no retraining needed):

| seed | CV ROC-AUC (mean +/- std, 5-fold) | single-holdout ROC-AUC |
|---|---|---|
| 239 (orig) | 0.717 +/- 0.032 | 0.740 |
| 1 | 0.704 +/- 0.010 | 0.734 |
| 2 | 0.635 +/- 0.013 | 0.700 |
| 3 | 0.628 +/- 0.026 | 0.572 |
| 4 | 0.738 +/- 0.037 | 0.738 |

**Caveat, not yet resolved**: this CV pools the 606 training repertoires' Theta (computed by a
model whose weights were fit *using* those same repertoires) with the 152 genuinely held-out
ones, so folds are a mix of optimistic (in-sample) and clean (held-out) repertoires -- inherited
from `quantile_features_cv`'s existing design, not new here. Still informative directionally:
seed3's CV mean (0.628) is *higher* than its single-holdout score (0.572), seed2's is *lower*
(0.635 vs 0.700) -- the single 152-repertoire holdout is noisy in both directions relative to
what CV suggests is more typical for a given trained model. A clean fix (CV restricted to just
the 152 held-out repertoires, or full nested retraining-per-fold) is still open; see the CV item
under "next steps" below.

### Does dropping phi/TM actually hurt, once compared fairly?

The original no-TM-vs-TM comparison (`use_topic_model=False`, 3 seeds, vs a warm-start-schedule
comparison set) mixed two different recipes (no-TM had no warm-start; the comparison set did).
Re-checked with a clean, paired comparison instead -- same base recipe (plain flagship, no
warm-start), matched by seed, with vs. without phi/TM:

| seed | TM-present (`theta_features` ROC-AUC) | no-TM (same) | delta |
|---|---|---|---|
| 239 | `v7_attn_dualentropy` 0.678 | `notm_seed239` 0.630 | +0.048 |
| 1 | `v7_seed1` 0.7145 | `notm_seed1` 0.671 | +0.044 |
| 2 | `v7_seed2` 0.647 | `notm_seed2` 0.633 | +0.014 |

**All 3/3 pairs favor TM, by 0.01-0.05 AUC** -- a small but consistent, directionally
unambiguous effect, and a much tighter read than the earlier mismatched-recipe comparison (which
was itself confounded by the ~0.07-0.17 seed variance documented above). Phi/TM is doing
something real, even though `tm_loss` itself never leaves its batch-chance ceiling in any of
these runs.

### How TM could affect AUC despite its own loss staying flat -- the architecture, precisely

In the winning config (`label_input: "repertoire"`), the label loss reads Theta directly
(`torch.exp(log_topic_proportions_ST)` in `airrtm/models/airrtm_model.py`) and **never reads
phi at all** -- confirmed by tracing the forward pass. But phi's layer
(`latent_space_to_topic_proportions_layer`) and Theta's pooling head (`repertoire_topic_head` or
`theta_attention`) both read the *same* shared embedding (`topic_input_ST`, built from the
encoder's reparameterized sample `z_SL`), and there is no `.detach()`/stop-gradient anywhere
separating them -- `tm_loss`'s gradient reaches the identical shared encoder + `z_mean_layer`/
`z_log_sigma_layer` weights that the label loss and reconstruction/KL also update every step.

**The only architecturally-plausible channel, then: TM's gradient reshapes the shared encoder
representation even while `tm_loss`'s own scalar value sits flat, and Theta's pooling head
(especially the attention variant) reads that reshaped representation.** This is consistent
with this document's own earlier finding that the attention*value pathway carries real
per-repertoire ranking signal that phi's own diagnostic (`quantile_features`) does not show.

### Batch/negative-pool construction, precisely (informs the sampling-strategy experiments below)

- `_stratified_groups` (`airrtm/training/train.py`) groups repertoires into
  `n_repertoires_in_batch` (=4 in every current config) per step, stratified **only by CMV
  label** (guarantees each group has both classes; a naive random group is single-class ~14% of
  the time, which zeroes the label loss's contrast), reshuffled fresh every pass.
- `_make_batch`/`_sample_indices` draws `n_sequences_per_repertoire_in_batch` (=8192) sequences
  per repertoire, abundance-weighted (multinomial proportional to `duplicate_count`, with
  replacement) in every real Emerson config -- on, not a dead knob.
- `n_theta_sequences_per_repertoire` (a disjoint sample specifically for inferring Theta,
  separate from the sequences TM/label/reconstruction score) **defaults to 0 and is unset in
  every current config** -- Theta is currently pooled from the exact same sequences the TM loss
  scores against it. This was tested once before (`v3a_disjoint_theta`, no effect) but that
  predates attention pooling, stratified groups, and the entropy-pair fix -- i.e. it predates
  every piece of the architecture that currently works, so the null result may be stale.
- The TM loss's negative/candidate pool is **the entire batch** -- all repertoires in the group
  times their sampled sequences (4x8192=32,768 by default) -- set by `n_repertoires_in_batch`
  and `n_sequences_per_repertoire_in_batch`, *not* by total corpus size. The earlier "16 vs 597
  total repertoires" experiments (this document, 2026-09-10) varied the total corpus available
  to sample from, not this per-step group composition -- `n_repertoires_in_batch` itself has
  never been swept.

**Launched, same day**: 8 runs testing the two untested levers above, all built on the
`warmstart_full_medium` recipe (attention pooling, `label_input: repertoire`,
`coef_anneal_epochs: 8`, `topic_usage_coef: 5.0`), on seeds 239/1/2 already used for the
existing baseline so results are directly paired:

| run | change from baseline | seeds |
|---|---|---|
| `disjoint_theta_seed{239,1,2}` | `n_theta_sequences_per_repertoire: 2048`, `n_sequences_per_repertoire_in_batch: 8192->6144` (per-repertoire draw total unchanged at 8192) | 239, 1, 2 |
| `reps8_seed{239,1,2}` | `n_repertoires_in_batch: 4->8`, `n_sequences_per_repertoire_in_batch: 8192->4096` (total batch pool unchanged at 32768) | 239, 1, 2 |
| `reps16_seed{239,1}` | `n_repertoires_in_batch: 4->16`, `n_sequences_per_repertoire_in_batch: 8192->2048` (total batch pool unchanged at 32768) | 239, 1 |

**Results landed, 2026-09-13** (`theta_features` test ROC-AUC, single 152-repertoire holdout and
5-fold CV mean; flagship 5-seed baseline range 0.572-0.740):

| run | holdout ROC-AUC | CV mean ROC-AUC (5-fold) |
|---|---|---|
| `disjoint_theta_seed239` | 0.680 | 0.675 |
| `disjoint_theta_seed1` | 0.678 | 0.699 |
| `disjoint_theta_seed2` | 0.627 | 0.621 |
| `reps8_seed239` | 0.680 | 0.693 |
| `reps8_seed1` | 0.696 | 0.651 |
| `reps8_seed2` | 0.634 | 0.645 |
| `reps16_seed239` | 0.671 | 0.658 |
| `reps16_seed1` | 0.686 | 0.655 |

All eight land inside the existing 5-seed flagship spread (0.572-0.740) -- neither lever
(`disjoint_theta`'s separate Theta-inference sample, `reps8`/`reps16`'s repertoire-group size)
produces a result distinguishable from seed noise. Same conclusion as the `topic_usage_coef`
sweep: no config here clears the bar the seed variance itself sets.

`reps8_seed1` is worth flagging on its own: during training it looked collapsed (best epoch only
5, `H_usage` 0.76-1.25 vs. the healthy ~2.7-3.1 the other seven runs held). Despite that, it
scored the *highest* holdout AUC of all eight (0.696) and a mid-pack CV mean (0.651) -- another
direct confirmation of this document's 2026-09-11/12 finding that **collapse does not predict
AUC**. Don't read `reps8_seed1`'s training-loop entropy trace as disqualifying without checking
the held-out number first.

**`tm_gain` now has an independent ceiling to compare against**: this document's own
`I(s;r) >= 0.052 nats` (a V+J+length classifier lower bound, unrelated to this model) is the
right reference frame for judging TM engagement, not "did the raw loss move." Our best `tm_gain`
so far (`v7_usage30`, ~1.4 on the x100-scaled metric = 0.014 nats) is roughly 25% of that
ceiling -- real engagement, nowhere near saturating the available signal.

### Next planned step (not started yet): a representation probe

If TM's benefit really is "reshapes the shared encoder, not its own converged loss value," that
should be visible directly: extract the shared encoder's frozen embedding (`z_mean_SL` or
`topic_input_ST`, before the phi/Theta heads) from matched TM-present vs. no-TM checkpoints (the
seed-239/1/2 pairs above), and fit a simple linear probe (logistic regression, `sklearn`, same
style as `airrtm/evaluation/metrics.py`) predicting repertoire identity or V/J-usage cluster
from it. If TM-present embeddings probe measurably better than no-TM embeddings at this task,
that's direct evidence for the shared-encoder-reshaping hypothesis, independent of downstream
classification AUC. Deferred until the 8 runs above land, in case they change which config is
worth probing.

---

## Update, 2026-09-11: burden-score baseline reconciled (0.781 was CDR3-only, not the current default)

An `evaluate-model` run (with `--skip_baseline` omitted, unlike most prior runs) reported burden
ROC-AUC **0.931** for the `notm_seed*` checkpoints -- far above the 0.781 reference. Root cause:
`BurdenScoreClassifier`'s current defaults (`p_threshold=1e-4`, `use_vj=True`) never match the
0.781/482-clonotype reference's methodology (`p<1e-3`, CDR3 only). This document's own
2026-09-08 note (above, "one methodology note") already flagged the threshold half of this; the
`use_vj` half had not been checked. Re-ran `BurdenScoreClassifier` directly on the current split
(606 train / 152 test -- 10 more repertoires than the original 597/151) at all four
threshold x use_vj combinations:

| p_threshold | use_vj | n_selected | test ROC-AUC | PR-AUC | acc |
|---|---|---|---|---|---|
| 1e-4 | True (code default) | 126 | 0.931 | 0.904 | 0.882 |
| 1e-3 | True | **528** | **0.874** | 0.827 | 0.829 |
| 1e-3 | False | 479 | 0.753 | 0.686 | 0.645 |
| 1e-3 | False (original reference) | 482 | 0.781 | 0.753 | 0.715 |

`use_vj=False, p<1e-3` reproduces the original reference almost exactly (479 vs 482 selected,
0.753 vs 0.781 AUC -- the residual gap is just the 10 extra repertoires in the current split).
This confirms both the reference number's methodology (CDR3-only, not V/J) and that the gap was
methodology drift, not a bug.

**Going forward, the primary reference is `p_threshold=1e-3, use_vj=True` (0.874 ROC-AUC)** --
see the Reference numbers table above. V/J genes measurably help the baseline too (0.753 ->
0.874 from adding them at the same threshold), and every model evaluated here is trained on
V/J-annotated sequences, so the V/J-augmented baseline is the fairer bar to clear. The legacy
CDR3-only 0.781 number is kept in the reference table for continuity but is no longer primary.
Practical note: `evaluate-model`'s own default (`p_threshold=1e-4`) matches neither reference --
pass `--burden_p_threshold 1e-3` (or equivalent) if the CLI is ever extended to expose it, or
recompute directly via `BurdenScoreClassifier(p_threshold=1e-3)` as done here.

**Same-session no-TM architecture eval** (`use_topic_model=False`, three seeds, evaluated
against the code's *default* baseline settings since the eval landed before this reconciliation):
`theta_features` test ROC-AUC 0.671 (seed 1), 0.630 (seed 239), 0.633 (seed 2) -- below every
TM/phi-containing flagship variant (0.70-0.74 across seeds). Dropping phi/TM entirely is worse,
not neutral: whatever `tm`'s inert-looking loss term is doing, it is not dead weight at full
scale. `quantile_features` is structurally `None` for these checkpoints (no phi, no per-sequence
score to build repertoire quantile features from).

---

## Update, 2026-09-13: repertoire-count scaling sweep and a new TM negative-sampling mode, both launched

**Repertoire-count scaling sweep.** Extends the 2026-09-09 "can the TM likelihood be fit at
all?" diagnostic, which only ever compared 16 repertoires (fits well, `tm` delta up to
+0.34 nats) against the full 597/606 (flat, delta ~0.007-0.009 even with 114 unopposed
epochs). Launched the identical unopposed-TM recipe
(`theta_mode: free`, all competing coefficients zeroed, `tm_likelihood_coef: 1.0`) at three
intermediate scales via the existing `--repertoire_slice` flag: `config_overfit_free_64.yaml`,
`_128.yaml`, `_256.yaml` (GPUs 0/1/2), each with `n_batches_per_repertoire` re-derived
(13/6/3) to hold steps/epoch at ~200, matching the 16-repertoire run's cadence rather than
reusing its raw value.

**Results landed.** All three finished by early stopping (patience 40): `overfit_free_64`
best epoch 97, `_128` best epoch 88, `_256` best epoch 99. `tm` reduction at best epoch
(nats, converting the logged `tm_gain` back from its x100 scale):

| repertoires | val `tm` reduction | train `tm` reduction | best epoch |
|---|---|---|---|
| 16 (`free_16`, 2026-09-09) | +0.159 | +0.337 | 46 |
| 64 | +0.062 | +0.171 | 97 |
| 128 | +0.026 | +0.081 | 88 |
| 256 | +0.013 | +0.046 | 99 |
| 597/606 (full, `amortized_full`, closest full-scale reference -- not the identical `free`-mode diagnostic) | ~+0.007 | -- | 13 |

**A smooth, monotonic decay with repertoire count, not a cliff.** Val `tm` reduction
roughly halves with each doubling of repertoire count (64->128->256: 0.062 -> 0.026 ->
0.013), extrapolating cleanly towards the full scale's ~0.007-0.009. No sign of a sharp
threshold on the *TM* branch specifically -- contrast this with the *label+entropy*
branch's threshold-locating sweep later in this update, which shows a sharp cutoff (works at
606, collapses everywhere from 100 to 400) rather than a smooth decay. The two branches
respond to repertoire-count scale differently: TM degrades smoothly, label+entropy fails
abruptly.

**Synthetic-collapse debug: a new class-contrastive TM negative-sampling mode.** The S1/S2
synthetic runs -- meant to be the clean control -- have never escaped
`label_acc=0.5`/`H_theta`,`H_usage`~0, even after the `topic_usage_coef` fix and raised
patience (300 epochs/patience 60, still collapsed per the 2026-09-08 update). Read
`CompositeLoss._tm_loss` (`airrtm/losses/composite_loss.py`) directly to check what the TM
negative pool actually is: every sequence in the batch, regardless of repertoire or class
(`negatives_ST = seq_topic_logits_ST`, no label-awareness) -- the label-stratified
repertoire grouping only guarantees the *label loss* sees both classes per step, it plays no
role in the TM term.

Added `tm_negative_mode: "batch" | "opposite_class_pair"` (default `"batch"`, so every
existing Emerson config is unaffected). In the new mode, each repertoire's TM likelihood is
normalised only against its own sequences plus one paired repertoire of the *opposite*
class (`_opposite_class_partners`, cycling through the smaller class's pool when the split
is uneven), instead of the whole batch -- a direct "look like my own repertoire, not the
opposite-class one" signal rather than a generic partition function. This revives the spirit
of the `_sample_other_repertoire_ids` branch PLAN.md's defect 1.4 called redundant once
in-batch normalisation landed -- worth flagging as a deliberate reconsideration of that call,
not a regression: 1.4's actual defect was the *unnormalised* likelihood elsewhere, and the
opposite-class-pairing idea itself was never A/B tested against the working normalised term.
Implementation: `CompositeLoss._tm_loss` split into `_tm_log_likelihood_per_sequence`
(returns the per-sequence term, enabling direct testing) plus the mean; masking reuses the
existing `cross_RS` tensor rather than adding new computation. 7 new tests in
`tests/test_losses.py` (66 total, all passing), including a numeric invariance check that a
repertoire's TM likelihood is exactly unchanged by perturbing an unrelated (non-partner)
repertoire's topic logits in the same batch -- the property this mode is supposed to have
and the default `"batch"` mode does not.

Relaunched all three synthetic configs (`S1_w0.001_v3`, `S1_w0.005_v3`, `S2_w0.001_v3`,
GPUs 3/4/5) combining this new mode with the other lever discussed: `n_repertoires_in_batch`
raised 4 -> 8 (`n_sequences_per_repertoire_in_batch` halved to 4096 to hold the total batch
pool at 32768), on the theory that only ~100 total synthetic repertoires need a larger
per-step slice for both the contrastive pairing and the label-stratified grouping to have
enough diversity.

**Negative result.** `S1_w0.005_v3` finished (early-stopped epoch 76, best epoch 16):
`H_theta`/`H_usage` collapsed to ~0.001-0.003, `label_acc` pinned at exactly 0.5 -- the same
failure mode as every prior synthetic attempt (2026-09-08, 2026-09-09 updates), unchanged by
either the class-contrastive TM negative pool or the larger batch. `S1_w0.001_v3` and
`S2_w0.001_v3` show the identical pattern through epoch ~82 (collapsed since roughly epoch
16-20), with occasional single-epoch noise spikes (e.g. `S1_w0.001_v3` epoch 81: `H_theta`
0.02 -> 1.37, total loss 6.13 -> 0.018) that revert completely the very next epoch --
transient instability, not real learning; `label_acc` never moves off ~0.49-0.51 through any
of it. **Neither lever fixed the synthetic collapse.** Whatever mechanism keeps this small,
clean-signal dataset collapsing is still unexplained and evidently not (just) about batch
diversity or the shape of the TM negative pool -- the entropy terms and label loss are
finding the same trivial fixed point regardless. `S1_w0.001_v3` and `S2_w0.001_v3` were killed
early (best epochs 94-still-running and 48 respectively, both collapsed) once `S1_w0.005_v3`'s
full run confirmed the pattern, to free the GPUs for the next attempt below rather than
finish out 300 epochs of a result already established.

**Third lever: a CNN branch on the encoder.** Added `TransformerCNNEncoder`
(`airrtm/models/encoder.py`) -- the existing `TransformerEncoder` plus a single
`Conv1d` branch (its own embedding, same-padding so sequence length is preserved,
mask-aware mean+max pooled the same way as the transformer's `"mean_max"` pooling),
concatenated. Registered as `encoder_type: "transformer_cnn"` in
`airrtm/models/factory.py`; `AIRRTM_Model` sizes its latent layers from
`encoder.get_output_dim()` generically, so no other code needed to change. Rationale: a
local, fixed-width filter is a different inductive bias than self-attention, worth trying
since the planted signal in S1/S2 is a short, fixed motif rather than a long-range
dependency. 4 new tests in `tests/test_encoder.py` (70 total, all passing).

Launched isolated against the *original* baseline recipe (not stacked on the two failed
levers above): `n_repertoires_in_batch` back to 4, `tm_negative_mode` back to the default
`"batch"` -- only `encoder_type` changed, so this is a clean single-variable test.
`config_synthetic_{s1_w001,s1_w005,s2_w001}_cnn.yaml`, GPUs 3/4/5, running as
`S1_w0.001_cnn`, `S1_w0.005_cnn`, `S2_w0.001_cnn`.

**Third lever also negative -- and faster.** All three CNN runs collapsed
(`H_theta`/`H_usage` -> ~0, `label_acc` pinned at 0.5) by epoch 6-8 -- *faster* than the
`opposite_class_pair` attempts (epoch ~16-20) or the original baseline. Killed all three
(the pattern was already unambiguous) to free GPUs 3/4/5. **Three independent levers --
bigger batch, class-contrastive TM negatives, a different encoder inductive bias -- now
fail identically and fast.** All three changed something about the TM branch or the shared
encoder; none touched the label+entropy branch itself. That convergent, encoder- and
TM-independent failure is itself informative: it points at the label+entropy branch as the
more likely site of the fixed point, not at anything tried so far.

**Correction, prompted by the threshold-sweep lesson below: that CNN verdict was premature
and is being retested.** The threshold-locating-sweep saga later in this document found that
several Emerson slices looked identically collapsed at epoch 5-10 and only started a real,
sustained recovery after epoch 30-90+. The CNN runs above were killed at epoch 6-8 -- the
exact same too-early read that turned out to be wrong there. **The CNN encoder result should
be treated as unconfirmed, not negative.** Relaunched all three synthetic datasets with the
CNN encoder, this time with a second seed each (`_seed239`, `_seed1`) and an explicit
commitment not to judge before at least ~60-80 epochs: `S1_w0.001_cnn_{seed239,seed1}`,
`S1_w0.005_cnn_{seed239,seed1}`, `S2_w0.001_cnn_{seed239,seed1}`, GPUs 0-5.

**Confirmed negative, this time with real confidence.** All six ran to genuine completion
via early stopping (not killed early): best epochs 3-109 (`S1_w0.001_seed239/seed1`: 5/5;
`S1_w0.005_seed239/seed1`: 5/100; `S2_w0.001_seed239/seed1`: 109/3), stopping at epochs
63-169. Every single one lands at `label_acc` = exactly 0.5000, `H_theta`/`H_usage` near 0
(with the same transient single-epoch noise spikes seen everywhere else in this document,
reverting immediately). Wide variation in best epoch (3 to 109) with an identical outcome
regardless rules out "just needed more patience" as an explanation this time -- unlike the
premature epoch-6-8 kill, these had a genuine, generous budget. **The CNN encoder is a real
fourth confirmed failed lever** (alongside bigger batch, class-contrastive TM negatives, and
dropping TM entirely) -- all four converge on the identical trivial fixed point on synthetic
data, none of them the culprit specifically.

**Fourth attempt: isolate the label+entropy branch, mirroring the unopposed-TM diagnostic
but for the other side.** The 2026-09-09 diagnostic isolated TM by zeroing entropy/label
competition (`theta_mode: free`, `tm_likelihood_coef: 1.0`, everything else zeroed) and found
TM *can* be fit at small scale. Built the mirror image for synthetic:
`config_synthetic_{s1_w001,s1_w005,s2_w001}_notm.yaml`, `use_topic_model: false`
(drops phi/TM entirely, same construction as `config_notm_seed1.yaml`'s Emerson ablation)
with the entropy pair (`theta_entropy_coef: 0.5`, `topic_usage_coef: 5.0`) and label
supervision left as the only active terms. Question: can label+entropy alone produce a
non-trivial Theta on this small, clean-signal dataset when not competing against TM at all?
Launched on GPUs 3/4/5 as `S1_w0.001_notm`, `S1_w0.005_notm`, `S2_w0.001_notm`.

**Also negative.** All three show the identical collapse through epoch 40-45:
`H_theta`/`H_usage` at 0.001-0.05 (the same transient single-epoch noise spikes that
immediately revert, e.g. `S1_w0.005_notm` epoch 44: `H_theta` 0.001 -> 2.28 -> 0.0002 the
next epoch), `label_acc` pinned at exactly 0.5 throughout. **Four independent levers now
fail identically**: bigger batch, class-contrastive TM negatives, a CNN encoder branch, and
-- this one -- removing TM from the competition entirely. The last result is the sharpest:
it rules out "TM-vs-label competition" as the mechanism, since the collapse happens even
when label+entropy are the *only* active terms, unopposed. Whatever fixed point this is, it
belongs to the label+entropy branch itself on this dataset, not to any interaction with TM
or the encoder.

**New hypothesis, tying this to the repertoire-count scaling sweep above:** synthetic has
only ~100 total repertoires vs Emerson's 606 train. Is this actually a repertoire-*count*
effect -- the same axis the TM-fitting story turned on -- rather than something specific to
synthetic's data generating process? Built `config_v7_reps100.yaml`: the *full* flagship
recipe (`config_v7.yaml` verbatim: label+entropy+TM all active together, not an isolated
diagnostic), run on **Emerson** via `--repertoire_slice 0:100` to match synthetic's scale
(`n_batches_per_repertoire` raised to 24 so steps/epoch, ~600, stays close to the full run's
~604 rather than cratering with fewer repertoire groups). Two seeds (239, 1), GPUs 6/7.
If this *also* collapses, repertoire count -- not synthetic-vs-Emerson data structure -- is
the shared root cause across both this thread and the TM-fitting one. If it doesn't (label_acc
clears chance the way the real 606-repertoire flagship runs do), synthetic's collapse is
specific to its data, not just its size.

**Confirmed: it is a repertoire-count effect.** Both seeds collapse immediately -- by epoch
0 -- not gradually like the full-scale runs' epoch-16-90 collapses. `H_theta`/`H_usage` sit
at 0.001-0.06 from the very first epoch, and `label_acc` is frozen at exactly **0.5700**
across every epoch logged so far (0-5), identically in both seeds -- not merely near chance,
bit-for-bit the same value epoch after epoch, consistent with the label head collapsing to a
constant per-repertoire prediction (whichever class is more common in this 100-repertoire
slice) immediately, before training can move it at all. This is a faster, sharper collapse
than anything seen on synthetic data (which took 16-20 epochs). **The Emerson flagship
recipe -- label+entropy+TM all active together, the exact config that works at 606
repertoires -- fails the same way synthetic data always has, once restricted to ~100
repertoires.** This settles the open question from this thread and reframes the whole
synthetic-collapse investigation: it was never about synthetic data being different from
Emerson (wrong signal structure, wrong witness rate, wrong encoder inductive bias) -- every
lever tried against that framing (bigger batch, class-contrastive TM negatives, a CNN
branch, dropping TM entirely) failed for the same underlying reason. **Repertoire count is
the actual gate**, on the label+entropy branch as much as (per the earlier TM-fitting
thread) the TM branch -- both branches need enough repertoires in the corpus to avoid
collapsing onto a trivial constant solution, and ~100 is below that threshold for either
one. Where exactly that threshold sits (below 606, above 100) is now the natural next
question -- the repertoire-count scaling sweep above (16/64/128/256/597) was designed for
the TM branch specifically; an analogous sweep for the *label+entropy* branch (e.g.
`v7_reps` at 200/300/400) would locate it for the branch that actually matters for
classification.

**200 also collapses.** `v7_reps200_seed239` shows the identical signature through epoch
5: `H_theta`/`H_usage` near 0, `label_acc` frozen at exactly **0.5800** every epoch (a
different frozen value than reps100's 0.5700, consistent with "predicts the majority class
of whichever repertoire slice", not a fixed bug value). **The threshold sits somewhere
between 200 and 606 -- not close to 100.** Bisecting further: launched
`config_v7_reps300.yaml` / `_reps400.yaml` (`--repertoire_slice 0:300` / `0:400`), GPUs 3/4.

**300 and 400 also collapse.** Both show the identical signature through epoch 4-5:
`H_theta`/`H_usage` near 0 (with the same transient single-epoch noise spikes seen
throughout this document, e.g. `reps300` epoch 1: `H_theta` 0.05 -> 2.79 -> 0.07 the next
epoch), `label_acc` frozen at a constant value every epoch (0.5667 for 300, 0.5650 for 400 --
again a different constant per slice, consistent with "predicts the majority class of this
particular repertoire subset" rather than a fixed bug value). **The threshold sits above
400, not between 200 and 400 as the naive bisection assumed.** Unlike the TM branch's smooth
decay (previous entry), the label+entropy branch shows no partial credit anywhere tried so
far: 100, 200, 300, and 400 all fail identically and fast; only the full 606 is known to
work. Launched the next points now that the TM scaling sweep's GPUs freed up:
`v7_reps500_seed239`, `v7_reps500_seed1` (2 seeds, since this is the most informative point
so far) and `v7_reps550_seed239` (pre-emptive, to save a round-trip if 500 also collapses),
GPUs 0/1/2.

**500 collapses (both seeds), 550 shows a subtly different, ambiguous signature.**
`v7_reps500_seed239` and `_seed1` are both frozen at exactly `label_acc=0.5480` across all 5
epochs -- the same rigid pattern as every other collapsed slice. `v7_reps550_seed239` is the
first slice in this entire sweep whose `label_acc` is *not* bit-identical every epoch
(0.5584 -> 0.5566 -> 0.5602 -> 0.5566 -> 0.5584 -> 0.5584, small but real oscillation) --
though `H_theta` is still low (0.001-0.04), so this is not yet a clear "working" signal, just
a qualitatively different, less-rigid one than the flat collapse seen at every other slice.
Too early (5-6 epochs) and too subtle to call the threshold yet -- **the honest read right
now is "somewhere at or above 550, not yet confirmed," not "550 works."** All 8 GPUs were at
90-100% utilization when this was checked, so no bisection point beyond 550 has been launched
yet; the next step is watching 550 (and, ideally, a second seed of it) for more epochs before
committing to a number.

**Correction: not resolved after all -- 550 shows a slow, sustained recovery through epoch
28, unlike every other slice.** Epochs 0-11 looked like the same collapse everything else
showed (see above), but continuing to watch past that point changed the picture: from
epoch ~14, `H_usage` climbs steadily and monotonically (1.37 -> 1.31 -> 1.65 -> ... -> 2.33
-> 2.50 -> 2.60 by epoch 28, out of the healthy target `log(30)=3.4`) rather than reverting
the way every transient blip in every other slice (100-500) has. `total` loss trends
steadily more negative over the same span (0.95 -> -5.66), consistent with real sustained
optimization rather than noise. `label_acc` has *not* moved yet -- still oscillating
0.556-0.560, right at this slice's ~0.558 majority baseline -- so this is not yet a
confirmed working run, but it is unambiguously a different trajectory in kind from anything
else in this sweep. **Read this as: the threshold may not be a hard cliff exactly at some N,
but a slower basin of attraction that 550 happens to be escaping (slowly) while 100-500 do
not (yet) -- or 550 sits right at the edge where escape is possible but not guaranteed.**
Watching closely; will update again once `label_acc` either clears the majority baseline or
`H_usage` plateaus/reverts.

**Update, epoch 34: entropy escape continues, but no classification payoff yet.**
`H_usage` keeps climbing (2.60 at epoch 28 -> 2.79 -> 2.91 by epoch 34, out of the ~3.4
target) and `total` loss keeps improving (-5.66 -> -7.23 over the same span) -- this is not
a blip, it is sustained, multi-epoch progress, the first of its kind anywhere in this sweep.
`H_theta` has stopped falling and is plateauing around 0.28-0.42 rather than continuing
toward the ~0.15-0.3 seen in healthy full-scale runs. **Most important: `label_acc` has not
moved at all** -- still 0.556-0.560 through epoch 34, indistinguishable from this slice's
~0.558 majority baseline the entire time. So Theta is genuinely escaping the collapsed fixed
point (unlike 100-500, which show zero comparable movement), but that escape has not yet
translated into any repertoire-classification signal. Two readings remain open: (a) this is
a slow version of the same recovery every full-scale run eventually shows (some of which
have best epochs in the 40-90 range before label_acc moves), and label_acc will follow once
Theta separation is far enough along; or (b) Theta can partially de-collapse without ever
producing a useful label signal at this repertoire count, a distinct and more surprising
failure mode from straightforward collapse. Continuing to watch before calling it either
way.

**Update, epoch 40: the entropy climb has plateaued, not continued to the healthy target.**
`H_usage` leveled off around 2.8-2.95 from epoch 34 onward (not still rising, and short of
the ~3.4 target); `H_theta` similarly settled around 0.25-0.4 rather than continuing to
fall. `label_acc` shows a marginal uptick at epochs 39-40 (0.5657, 0.5620) against the prior
0.556-0.560 band -- but this is a ~1-point move on a metric that has wobbled by similar
amounts throughout without meaning anything, so it is not yet a confirmed signal either way.
Best-supported read at this point: 550 escaped the sharpest form of collapse (unlike
100-500) but has settled into a *partial*, plateaued state short of the full-scale runs'
healthy signature -- not a clean binary "550 works" result. Whether `label_acc` continues
climbing past this point remains the open question; will only be resolved by watching
further epochs or, more decisively, by evaluating the eventual best checkpoint against the
held-out test split the way every other reported number in this document is.

**Update, epoch 47: settled, not still moving.** `H_usage` has held flat at 2.85-2.98 and
`H_theta` at 0.22-0.28 for 13 straight epochs (34-47) -- a real plateau, not a slow
continuation. `label_acc` over the same span: 0.5584, 0.5566, 0.5602, 0.5511, 0.5529,
0.5639, 0.5547 -- bouncing with no trend, centered almost exactly on the ~0.558 majority
baseline. **Best-supported conclusion now: Theta partially de-collapsed (escaping the
sharpest failure mode that 100-500 never left) but plateaued at a state that still carries
no repertoire-classification signal.** This is a third, distinct outcome from "collapses"
and "works" -- a genuine intermediate regime this sweep hadn't produced before. Whether this
plateau is 550's actual ceiling or would eventually break through given more epochs (best
epoch so far looks to be in the 30-46 range judging by `total` loss, with patience 60 still
to run) remains open; the decisive check, once it stops, is `evaluate-model` against the
held-out test split -- train-loop `label_acc` at this repertoire count is itself measured on
only ~55 held-out-within-training repertoires (`val_size: 0.1` of 550), coarse enough that a
real signal could still be underpowered here even if the checkpoint would show one on the
full 152-repertoire test split.

**Methodological check, prompted by this result**: `--repertoire_slice 0:N` takes the
metadata's first `N` rows, not a random sample (`airrtm/cli/train.py`), so before reading
too much into a threshold this close to the full dataset, checked whether the ordering
itself is a confound. Label balance is stable across every slice tested (16 -> 0.563, 64 ->
0.484, 100 -> 0.430, 200 -> 0.420, 300 -> 0.433, 400 -> 0.435, 500 -> 0.452, 550 -> 0.442,
606 -> 0.444) -- no systematic drift, so label imbalance is not the explanation. Does not
rule out other ordering effects (cohort, sequencing batch) that might correlate with row
order; a genuinely random subsample at, say, N=550 would be a cheap robustness check if this
threshold is pursued further.

### Correction: the "100-550 collapse, 606 works" conclusion above was premature

Every check of the 100/200/300/400/500/550 slices above was made within the first ~5-11
epochs of training. All six jobs kept running (patience 60, none had early-stopped), and by
epochs 47-74 **every single slice** -- not just 550 -- shows the same delayed entropy
recovery: `H_usage` climbing from ~0 into the 1.9-3.1 range, `H_theta` settling around
0.2-0.4. The clean "sharp cliff between 550 and 606" conclusion recorded above does not
survive watching longer. This is a real methodological lesson for this specific
architecture/dataset combination: **these diagnostics need dozens of epochs before a
collapsed-looking run can be called collapsed for good** -- something already known in
general from this document's own history (v4/v5's collapse took until epoch ~18-80; the
full 606-repertoire runs' best epochs range 40-96), but under-weighted when characterizing
the smaller slices, which looked frozen solid for the first ~10 epochs in every case.

**What actually distinguishes the slices now is classification signal, not entropy
recovery** -- and it is not monotonic in repertoire count:

| slice | epoch (still running) | `H_usage` | `label_acc` | majority baseline (this slice) | verdict |
|---|---|---|---|---|---|
| 100 | 74 | ~2.0-2.3 | 0.55-0.58 | 0.57 | at/near baseline |
| 200 | 66 | ~2.6-2.8 | 0.54-0.58 | 0.58 | at/below baseline |
| 300 | 59 | ~2.7-2.9 | 0.52-0.55 | 0.567 | below baseline |
| 400 | 78 | ~2.85-2.94 | **0.61-0.62, stable** (0.58-0.60 at ep.64 -> 0.59-0.63 at ep.70 -> settled at 0.61-0.62 through ep.78) | 0.565 | **above baseline, holding steady -- only slice that is** |
| 500 | 49 | ~2.8-3.1 | 0.54-0.56 | 0.548 | at/below baseline |
| 550 | 47 (plateaued) | ~2.9 | 0.55-0.57 | 0.558 | at/below baseline |

400 is the outlier, showing real above-baseline signal and climbing; every other slice from
100-550 sits at or below its own majority baseline despite comparably recovered entropy.
Given none of these runs have early-stopped yet and every printed `label_acc` here is a
*validation*-split metric computed on a small, size-dependent held-out slice (~10 repertoires
for the 100-slice, ~55 for the 550-slice) -- exactly the kind of small-N noise this document
has flagged before ("training-loop numbers... do not reliably predict test AUC even when
they look the same across runs", 2026-09-09) -- **none of this should be read as settled**.
The decisive check, once each run actually early-stops, is `evaluate-model` against the real
152-repertoire held-out test split, not these in-training validation numbers.

### Second correction: the whole threshold-locating sweep was chasing noise -- there is no threshold

All six runs kept training for ~14+ hours without early-stopping (patience 60 tracks total val
loss, which kept improving via reconstruction quality alone -- `rec_acc` climbed past 0.95 for
several slices -- long after label-task performance had stopped moving; this is why patience
never triggered and why waiting for early stopping was the wrong strategy here). Rather than
wait longer, evaluated a recent mid-training checkpoint from each run
(`checkpoint_epoch_{100..145}.pt`, `checkpoint_every: 5`). These periodic checkpoints don't
carry `model_config` the way the final `model.pt` does, so a small scratch script
(`eval_checkpoint.py`, not committed) rebuilds the model from each run's own config yaml +
`au.load_repertoires`, loads the periodic `state_dict`, and calls `topic_proportion_features` /
`classify_repertoires` directly -- the same functions `evaluate-model` uses internally --
against the **real, unchanged 152-repertoire held-out test split**:

| slice | seed | checkpoint epoch | test ROC-AUC (single holdout) | test ROC-AUC (5-fold CV mean) |
|---|---|---|---|---|
| 100 | 239 | 145 | 0.651 | **0.716** |
| 100 | 1 | 145 | 0.489 | 0.606 |
| 200 | 239 | 135 | 0.628 | 0.562 |
| 300 | 239 | 125 | 0.675 | 0.645 |
| 400 | 239 | 120 | 0.659 | 0.628 |
| 500 | 239 | 110 | 0.671 | 0.624 |
| 500 | 1 | 110 | 0.655 | 0.614 |
| 550 | 239 | 125 | 0.693 | **0.717** |

(Test-split majority baseline: accuracy 0.539, same 152 repertoires used across this
document's entire 2026-09-08 -> 2026-09-13 evaluation history.)

**Every slice from 100 to 550 produces a real, working classifier, comfortably within this
session's flagship 5-seed range (0.572-0.740, 2026-09-09 entry).** There is no threshold, no
cliff, and no gating repertoire count for the label+entropy branch at all -- the entire premise
of this sweep (launched after `v7_reps100` appeared to collapse immediately) does not survive
contact with the real test split. What actually happened: each slice's *training-loop*
`label_acc` is computed on a tiny, size-dependent held-out fraction (`val_size: 0.1` of the
slice -- as few as 10 repertoires for the 100-slice), far too coarse to reliably detect real
signal -- exactly the caveat this document already raised in the 2026-09-09 entry
("training-loop numbers do not reliably predict test AUC even when they look the same across
runs") and then under-weighted for the smaller slices in this sweep. The apparent
"100/200/300 collapse, only 400/550 show signal" pattern from watching training-loop metrics
for hours was an artifact of that measurement, not a property of the model.

**Practical lesson for the rest of this investigation**: when the training-loop validation
metric is on a small repertoire count, do not trust it either way (collapsed-looking or
working-looking) -- pull a checkpoint and run the real evaluation instead of waiting on
patience-based early stopping, which can be driven by an unrelated loss component (here,
reconstruction) and may never trigger in a useful timeframe. The `eval_checkpoint.py` approach
above (rebuild model from config + state_dict, skip waiting for the final `model.pt`) is
reusable whenever a run's in-training signal is ambiguous and its `evaluate-model` output
isn't ready yet.

**What remains actually true from this whole thread**: dropping repertoire count from 606 to
100 does not break the label+entropy branch at all, in sharp contrast to the earlier
(separately confirmed, still-standing) finding that it *does* progressively degrade the TM
branch specifically (the smooth 16/64/128/256/597 decay in the "repertoire-count scaling
sweep" entry above). The two branches' sensitivity to repertoire count is genuinely different
-- just not in the way this sweep spent most of a day concluding.

### Representation probe landed (the deferred step from the prior section)

Ran the probe deferred above, now that the disjoint_theta/reps8/reps16 runs (all within
seed noise, no lever won) have landed. Extracted each repertoire's mean `z_mean_SL`
(`AIRRTM_Model.sequence_to_latent`, the shared encoder embedding before the phi/Theta
heads) for the matched TM-present/no-TM seed triples (`v7_attn_dualentropy`/`v7_seed1`/
`v7_seed2` vs. `notm_seed239`/`notm_seed1`/`notm_seed2`), for all 758 repertoires
(train+test). Built a target with nothing to do with CMV: K-means (k=5) over each
repertoire's V-gene usage histogram, cluster sizes `[149, 179, 133, 264, 33]` (majority
baseline accuracy 0.348). Fit a 5-fold cross-validated logistic-regression probe from the
raw 64-dim embedding to the cluster label. Script: `representation_probe.py` (scratchpad,
not committed -- same status as `fisher_overlap_check.py`).

| seed | TM-present probe accuracy | no-TM probe accuracy | delta |
|---|---|---|---|
| 239 | 0.364 | 0.359 | +0.005 |
| 1 | 0.372 | 0.348 | +0.024 |
| 2 | 0.383 | 0.369 | +0.013 |
| mean | **0.373** | **0.359** | **+0.014** |

**All 3/3 matched pairs favor TM-present**, same direction as the earlier clean paired
`theta_features` AUC comparison (2026-09-11/12, also 3/3 favoring TM by 0.01-0.05). But the
magnitude here is small and both conditions sit barely above the 0.348 majority baseline --
neither embedding strongly encodes V-usage structure. Read this as weak, consistent
supporting evidence for the shared-encoder-reshaping hypothesis (TM's gradient still reaches
the same encoder Theta's pooling head reads, even though `tm_loss` itself never converges
usefully) -- not as strong confirmation. The effect is real-directioned but small enough
that a single additional seed either way would not be surprising.

---

## Update, 2026-09-14: v1's actual formulation, implemented and launched

Four independent levers had failed identically on synthetic data (bigger batch, class-paired
TM negatives, a CNN encoder, dropping TM entirely -- all confirmed, the CNN result now with
real confidence after a full retest). Prompted to go back to the source: the user pointed at
the original v1 TensorFlow implementation (`archive/`, matching
[csi-greifflab/airrtm](https://github.com/csi-greifflab/airrtm/blob/main/model.py)), which is
known to have worked on this exact synthetic data. A structured read of `archive/model.py`,
`archive/losses.py`, `archive/train_model.py`, `archive/utils.py` against the current
`airrtm/models/airrtm_model.py` / `airrtm/losses/composite_loss.py` turned up two structural
differences neither previously tested:

1. **Theta is never softmax-normalised in v1.** `archive/model.py:220-224` -- the softmax is
   present in the code but commented out. Every run in this document's history forces Theta
   onto a probability simplex (`torch.log_softmax` in `airrtm_model.py`); v1's Theta is a raw,
   unconstrained score. The entropy-pair fight this whole investigation has managed since
   finding 5 (`theta_entropy_coef` vs `topic_usage_coef`) only exists *because* of that
   simplex constraint -- v1 has no equivalent degenerate attractor to collapse into, and in
   fact never uses entropy/decorrelation regularisation at all (`archive/model.py:65-66,90-91`
   are constructor args that are never read in `call()` -- dead code, confirming the simplex
   constraint is what created the need for them in the first place).
2. **The TM likelihood is a plain, clipped BCE on the raw ``theta . phi`` dot product**, not an
   in-batch softmax partition function. `archive/model.py:252,263` feeds an unconstrained `Dot`
   output straight into `BinaryCrossentropy(from_logits=False)` (`archive/losses.py:19-26`),
   which Keras clips into `(eps, 1-eps)` -- effectively a saturating, hinge-like objective, not
   a proper unbounded cross-entropy. Negatives are explicit: every sequence gets a duplicate
   row targeted at a repertoire drawn from the *opposite label class* (`archive/utils.py:
   184-193`), regenerated every epoch (`archive/train_model.py:89-98`). This session's earlier
   `tm_negative_mode="opposite_class_pair"` attempt used the same opposite-class pairing idea
   but layered on top of the still-softmax-normalised, still-partition-function objective --
   never v1's actual unnormalised formulation.

**Implemented both together**, since they're coupled (an unconstrained Theta only makes sense
paired with an unnormalised likelihood):
- `AIRRTM_Model(theta_normalization="none")` (`airrtm/models/airrtm_model.py`) -- Theta is the
  raw logit row, fed straight into the label head with no `exp()` when
  `label_input="repertoire"`, matching `archive/model.py:263` exactly.
- `CompositeLoss(tm_likelihood_family="raw_bce")` (`airrtm/losses/composite_loss.py`) -- a new
  `_tm_loss_raw_bce` method: clipped BCE on `(theta_own . phi)` against target 1 and
  `(theta_partner . phi)` against target 0, reusing the existing `_opposite_class_partners`
  pairing helper. `theta_is_normalized=False` forces `theta_entropy_coef`/`topic_usage_coef`
  to zero (they assume a simplex that no longer exists) -- the same guard style already used
  for `use_topic_model=False`.
- `phi_l2_coef` -- a new regulariser matching v1's `activity_regularizer=L2(...)` on raw phi
  (`archive/model.py:197`), previously unreplicated.
- `airrtm/training/train.py::_tm_random_baseline` now reports the correct chance value for
  this family (`log(2)` -- a BCE's chance level -- not `log(pool_size)`, which only applies to
  the batch-softmax family).

11 new tests (`tests/test_model.py`, `tests/test_losses.py`; 78/78 total passing), covering
the normalisation guard, the raw dot-product score, the label head reading raw Theta directly,
the `theta_is_normalized` guard, and a directional check that the raw-BCE loss actually rewards
alignment with a sequence's own repertoire over its opposite-class partner. Smoke-tested end
to end on a 16-repertoire slice (3 epochs, no crashes) before the real launch.

**Deliberately not done**: v1's literal free per-repertoire embedding (`theta_mode="free"`) is
blocked by the existing guard rejecting `label_input="repertoire"` + `theta_mode="free"` (the
"v1 memorisation shortcut" -- a lookup table that can't score held-out repertoires). Left that
guard in place and used `theta_mode="amortized"` for the first batch of runs instead; the
literal free-embedding replica remains a natural follow-up if the amortized hybrid doesn't
work either.

**Launched**: 8 runs, full factorial over the 3 axes most likely to matter -- coefficient
balance (v1's actual balance, `vae_coef=0.005`/`tm_likelihood_coef=0.99` from
`archive/model.py:406-436`, vs. this session's standard `0.05`/`0.7`), `phi_l2_coef` (v1's
default 0.001 vs. off), and `theta_pooling` (attention vs. mean) -- all on `S1_w0.001`
(the hardest witness rate), seed 239, `n_topics_signal/nonsignal=5/10` (unchanged from prior
synthetic runs), `n_epochs=300`, `patience=60`, GPUs 0-7:
`v1style_{v1exact,standard}_{l2on,l2off}_{attention,mean}`.

**Correction, ~1h in: wrong learning rate.** Every config in this document's entire history --
102 of them -- uses `learning_rate: 0.0003`. `archive/model.py:442` (`get_default_model`) and
`:453` (`load_model`) both actually use `1e-3`, over 3x higher; this was not one of the two
differences the v1-vs-v2/v3 comparison ranked, and slipped through into the first launch of
these 8 configs unnoticed. At epoch ~44, all 8 showed `tm` glued to the exact chance baseline
(0.693) and `total` loss barely moving epoch to epoch -- consistent with, though not proof of,
a learning rate too conservative to show visible progress in that many epochs. Killed all 8 and
relaunched with `learning_rate: 0.001` (GPUs 0-7, same 8-way grid otherwise unchanged). Results
pending. Note this discrepancy is not unique to these 8 runs -- every other config in this
document trained at 0.0003 too, though those were tuned/validated at that rate over many
iterations, unlike this fresh v1-replica attempt where the mismatch was never intentional.

**Update: the corrected LR unstuck `total`/`rec` but not `tm`/`label`, and that pattern has
now held for 170+ epochs across all 6 still-running configs** (2 of the original 8,
`standard_l2off_{attention,mean}`, were killed to free GPUs 6/7 for the VAE-warmup experiment
below -- their role in this grid was redundant with the other 6). At epoch ~170-174: `tm` is
still glued to exactly the chance baseline (0.693 +/- 0.002) in every one of the 6, `label_acc`
still pure noise around 0.5 (0.44-0.56) with no trend -- while `rec_acc` in the "standard"
(`vae_coef=0.05`) configs has climbed substantially further (0.72-0.84) than at the epoch-44
check, confirming the optimizer itself is working fine, just not for the TM/label branches
under this objective. **170+ epochs of flat `tm`/`label` is a much longer, more solidly
supported window than the 40-90 epochs where every other real signal in this document has
shown up** -- not declared fully final since none have early-stopped yet, but this is a
meaningfully stronger negative result than the epoch-44 read.

---

## Update, 2026-09-14: VAE-coefficient warm-start, both directions, on Emerson

Prompted by discussing the existing warm-start mechanism (`tm_likelihood_coef_start` /
`theta_entropy_coef_start` / `topic_usage_coef_start`, 2026-09-10): what if `vae_coef` itself
-- the *outer* split between the VAE branch and everything else
(`total = vae_coef * VAE + (1 - vae_coef) * non_vae`) -- were annealed the same way? Unlike
the existing warm-start, which only reshuffles weight *within* the non-VAE branch, this
reshuffles weight *between* VAE and everything else -- a strictly stronger lever, since at
either extreme one whole branch gets zero gradient regardless of its own coefficients.

Implemented as `vae_coef_start` / `reconstruction_loss_coef_start` / `vae_anneal_epochs`
(`airrtm/training/train.py::train_model`, same `_linear_schedule` mechanism as the existing
warm-start, threaded through `airrtm/cli/train.py`). Two directions, both real experiments:
- **"up"**: `vae_coef_start` near 0.0, ramping *up* to the target -- TM/label/entropy get an
  unopposed window with literally zero VAE gradient reaching the shared encoder, the same
  regime as the standalone "unopposed TM" diagnostic (2026-09-09), except VAE fades back in
  afterward instead of staying off for the whole run.
- **"down"**: `vae_coef_start` near 1.0, ramping *down* -- the model spends its first epochs
  as a plain sequence autoencoder before topic modelling/classification attach at all.

6 new tests (`_linear_schedule` unit tests + an integration test confirming the schedule
reaches the criterion's real target), 81/81 total passing.

**Launched both on Emerson** (killed `v1style_standard_l2off_{attention,mean}` on GPUs 6/7 to
free them -- those two configs' role in the synthetic sweep is redundant with the other 6
still running): `config_v7_vaewarmup_up.yaml` (`vae_coef_start: 0.0`) and
`config_v7_vaewarmup_down.yaml` (`vae_coef_start: 1.0`), both `vae_anneal_epochs: 40`,
otherwise the standard flagship recipe (`config_v7.yaml`: attention pooling, `label_input:
repertoire`, both entropy terms, `tm_likelihood_coef: 0.7`) with `vae_coef` still targeting
the standing default **0.05** in both -- deliberately not lowered to 0.01 or 0, so the two
runs differ only in warm-start direction, not final steady state.

**Sanity check confirmed the mechanism is working**: at epoch 0, `v7_vaewarmup_up`'s `total`
(7.35) tracks `0.7*tm + 0.3*label + regularizers` almost exactly, not reconstruction (which
is barely moving, `rec_acc` frozen at 0.357 for epochs 2-4 to four decimal places -- expected,
since it gets almost no gradient with `vae_coef~0`). `v7_vaewarmup_down`'s `total` (-0.02)
tracks `0.95*rec + 0.05*kl + regularizers`, not tm/label (frozen at their usual starting
values, as designed). One nuance worth flagging: `theta_entropy_coef`/`topic_usage_coef` are
*not* gated by `vae_coef` at all (they're additive top-level terms in `CompositeLoss`), so
they stay fully active in both directions regardless of which branch is suppressed.

**Early trend, epoch 0-4 (both still far too early to call anything settled -- ~15 min/epoch
at full Emerson scale, so the 40-epoch anneal window alone is ~10 hours)**:
`v7_vaewarmup_down`'s `H_usage` is climbing cleanly and quickly (0.33 -> 1.84 -> 2.41 -> 2.46
-> 2.52) toward the ~3.4 healthy target -- faster than any prior full-scale run in this
document reached a comparable level, because the entropy/usage regularizers are fully active
and completely unopposed by any TM/label competition for the shared encoder.
`v7_vaewarmup_up` shows the same fast collapse (`H_theta`/`H_usage` near 0 by epoch 1-4) every
ordinary flagship run shows this early -- unsurprising, since its non-VAE branch is fully
active from epoch 0, same as a normal run. Neither `tm`/`label`/`label_acc` has moved in
either run yet.

**Update, epoch 14 (~9 min/epoch on full Emerson scale):** `v7_vaewarmup_down`'s
reconstruction has essentially saturated (`rec_acc` 0.72 -> 0.98 in 14 epochs, unusually fast
-- expected with zero competition for the encoder), and `H_usage` has **plateaued** around
2.7-2.9 rather than continuing to the ~3.4 ceiling -- the same plateau level the
repertoire-count threshold-sweep runs settled at (2026-09-13/14 entries above), not a cleaner
result just because TM/label are absent. `v7_vaewarmup_up` remains stuck across every metric
through all 14 epochs (`rec_acc` frozen at exactly 0.357, `H_theta`/`H_usage` collapsed near
0) -- being "unopposed" from epoch 0 has not produced any acceleration yet. Results pending;
`vae_coef` doesn't reach its final target in either direction until epoch 40.

### Fix: regularisers were leaking through a "pure VAE" phase, and a wider variant grid

Prompted by the epoch-14 read above: `theta_entropy_coef`/`topic_usage_coef`/
`topic_l1_coef`/`topic_decorrelation_coef`/`phi_l2_coef` were flat top-level terms in
`CompositeLoss`, not gated by `vae_coef` at all -- so `v7_vaewarmup_down`'s climbing
`H_usage` (0.33 -> 2.9) during its nominally "pure VAE" phase was partly these regularisers
still pushing on Theta unopposed, not a clean VAE-only phase. Moved all five under the
`(1 - vae_coef)` branch alongside tm/label (`airrtm/losses/composite_loss.py`), so
`vae_coef=1.0` now genuinely stops all topic-model gradient, not just tm/label's. 2 new
tests (82/82 total passing), including one asserting `vae_coef=1.0` produces the identical
total loss regardless of how the four regularisers plus `phi_l2_coef` are set.

**Killed and relaunched everything with the fix**, plus a wider variant grid testing two
further axes: full-range endpoints (target `vae_coef` at the true extreme 0.0/1.0, not just
the flagship's small 0.05) and shorter anneal durations (5/15 epochs, alongside the existing
40). All 8 GPUs, `config_v7_vaewarmup_*.yaml`:

| run | `vae_coef_start` -> target | anneal epochs |
|---|---|---|
| `up` | 0.0 -> 0.05 | 40 (rerun with the fix) |
| `down` | 1.0 -> 0.05 | 40 (rerun with the fix) |
| `up_15_to005` | 0.0 -> 0.05 | 15 |
| `down_15_to005` | 1.0 -> 0.05 | 15 |
| `down_15_to0` | 1.0 -> **0.0** | 15 (full range -- VAE permanently dropped after warmup) |
| `up_15_to1` | 0.0 -> **1.0** | 15 (full range mirror -- ends fully VAE-dominant) |
| `down_5_to005` | 1.0 -> 0.05 | 5 |
| `up_5_to005` | 0.0 -> 0.05 | 5 |

**Fix confirmed at epoch 0**: every "down" variant now starts with `H_theta`/`H_usage` near
their maximum (~3.36/3.39, essentially `log(30)=3.40`) -- the untrained topic head's passive,
near-uniform initial state -- rather than the pre-fix run's climbing-from-0.33 signature,
confirming the regularisers no longer leak through the pure-VAE phase.

**First concrete post-anneal event, epoch 5-6**: `down_5_to005` (`vae_anneal_epochs: 5`) is
the first to cross into its post-anneal phase, where `vae_coef` reaches its final target
(0.05) and TM/label get real weight for the first time. Right at that transition,
reconstruction -- which had climbed to `rec_acc` 0.90-0.94 during the pure-VAE phase --
**collapsed back to 0.37-0.38** (the same floor the "up" variants never leave), with
`H_usage` correspondingly dropping from ~2.4 back to 0.5-0.8. This mirrors the existing
coefficient-level warm-start finding (`warmstart_16_long`, 2026-09-10: "the model actively
trades TM fit away in exchange for label accuracy") at the VAE-branch level instead, and much
faster given the short window. Only 1-2 post-anneal epochs so far -- a real, mechanism-
consistent event, not yet a settled trend. No `tm`/`label` movement in any of the 8 runs yet,
including the ones now fully at target weight -- too early regardless of weight.

**Update, epoch 9**: `v7_vaewarmup_up_5_to005` killed -- its 5-epoch anneal completed several
epochs ago and it shows the identical frozen state (`rec_acc` 0.357, `H_usage` ~0) as before
the anneal even started, no differentiation from the un-warmed-up baseline; least informative
of the 8 at this point. `v7_vaewarmup_up_15_to1` is the most actively evolving run right now:
`H_theta`/`H_usage` climbing from near-0 to ~1-2.4 and `kl_d` collapsing toward ~0.1 as
`vae_coef` (target 1.0) approaches the halfway point of its 15-epoch ramp.

**New run, replacing the killed one on GPU 5**: `config_v7_tmlabel_overfit.yaml` --
`vae_coef: 0.0` set as a plain static value, *not* a warm-start (no `vae_coef_start`/
`vae_anneal_epochs` at all). The VAE branch (reconstruction + KL) never gets any weight for
the entire run, not just a warm-up window -- TM/label/entropy have the shared encoder
permanently to themselves. With this session's regulariser-regrouping fix, this also means
the entropy/usage regularisers get their full intended weight throughout (same as the
flagship's normal steady state), unlike the pre-fix "unopposed TM" diagnostics from
2026-09-09/10 which zeroed those regularisers too to isolate TM alone -- this run keeps them
active, closer to "the label+entropy branch trained with zero VAE competition, permanently"
than a pure TM-only diagnostic. Results pending.

## 2026-09-14: encoder simplified (heads 8->2, pooling mean_max->mean), full relaunch

Current encoder (both flagship and all warm-start variants) was `attention_dim=64,
attention_dim_head=32, attention_heads=8, depth=4`, `pooling="mean_max"` -- 8-head attention
over 64 dims (8 heads x 32 head-dim, `attn_one_kv_head=False`), `mean_max` pooling
concatenating mask-aware mean and max over positions (output width `2*attention_dim`). Same
shape for the decoder. No architecture change has been tested since the original v7 flagship
recipe was picked.

**Change**: `attention_heads: 8 -> 2` (encoder and decoder both), encoder `pooling:
"mean_max" -> "mean"` (decoder has no `pooling` param). `attention_dim=64`,
`attention_dim_head=32` unchanged. Implemented `TransformerEncoder(pooling="mean")` as a
third pooling mode (`airrtm/models/encoder.py`) -- mask-aware mean only, half the output
width of `mean_max`; refactored the shared mean logic out into `_masked_mean()`, reused by
both `_masked_mean_max()` and the new mode. Two new tests in `tests/test_encoder.py`
(`test_mean_pooling_output_is_half_of_mean_max`, `test_mean_pooling_rejected_by_unsupported_
value`) confirm `"mean"` output equals the first half of `"mean_max"` output bit-for-bit
(same seed, `.eval()` to kill dropout nondeterminism) and that an unsupported pooling string
still raises. Full suite: 84/84 passing.

Note this is a **different pooling axis** than `theta_pooling` (repertoire-level
aggregation, FINDINGS.md finding 1) -- `theta_pooling` stays `"attention"` throughout,
unchanged. This change only touches how each transformer encoder collapses its own
per-position output.

**Killed** all 8 running jobs (`v7_vaewarmup_up`, `v7_vaewarmup_down`,
`v7_vaewarmup_up_15_to005`, `v7_vaewarmup_down_15_to005`, `v7_vaewarmup_down_15_to0`,
`v7_vaewarmup_up_15_to1`, `v7_vaewarmup_down_5_to005`, `v7_tmlabel_overfit`) and relaunched 8
new jobs with the simplified encoder (suffix `_simpleenc`), one per GPU 0-7:
`warmstart_full_medium` (the 0.740-AUC recipe), the 6 canonical VAE-warmup variants (dropping
`down_5_to005`, the least informative of the original 8 per the prior entry), and
`v7_tmlabel_overfit`. Each new config is a verified 3-line diff from its pre-existing
counterpart (`attention_heads: 8->2` x2, `pooling: "mean_max"->"mean"` x1) -- nothing else
changed, including all warm-start schedules and loss coefficients. All 8 confirmed alive and
past data-loading at ~1 minute in. Results pending.

**Check-in at ~24 min (epoch 5-7 for the fast/warmstart-subset jobs, epoch 1-2 for the
full-dataset `down`/`down_15_to0`/`down_15_to005` jobs, which run ~3x slower per epoch on the
larger dataset -- expected, not a stall). All 8 PIDs alive, no crashes.**

Clear split by warmup direction, consistent with the simplified encoder (heads=2,
mean-pooling):
- **`_down` variants healthy**: `v7_vaewarmup_down_simpleenc`, `_down_15_to005`,
  `_down_15_to0` all show H_theta/H_usage staying well above 0 (0.9-3.4) and rec_acc climbing
  fast (0.71->0.83->0.89 by epoch 2 for `down_simpleenc`). No collapse yet.
- **`_up` variants collapsing**: `v7_vaewarmup_up_simpleenc`, `_up_15_to005` both show
  H_theta/H_usage crashed to ~0.001-0.08 by epoch 5-7 and staying there -- classic topic
  collapse. `_up_15_to1` is oscillating (collapsed at ep2, recovered ep3, collapsed again
  ep4) rather than converging.
- **`v7_tmlabel_overfit_simpleenc` looks bad**: rec_acc actively *degrading* epoch over epoch
  (0.090 -> 0.037 -> 0.030) with H_theta/H_usage pinned near 0 and phi_l2 elevated (~8-10) --
  diverging, not just collapsed. Worth killing early if this doesn't turn around by the next
  check.
- **`warmstart_full_medium_simpleenc`** (the prior 0.740-AUC recipe) looked fine through
  epoch 5 (H_theta 0.63, rec_acc 0.37) then collapsed at epoch 6 (H_theta 0.63->0.09, H_usage
  0.92->0.11) in a single step. Watch whether this recovers or is a permanent collapse --
  this is the config that matters most since it's the best-known baseline.

`label_acc` is flat at ~0.55-0.56 across every job regardless of collapse state (vs. ~0.445 at
epoch 0) -- too early relative to the 60-300 configured epochs to compare against the 0.740
AUC baseline.

**Correction from the ~24min check-in above**: the earlier per-epoch extraction regex didn't
allow a leading `-` on numeric fields, so it silently dropped every epoch line once `total`
went negative (common once `rec` drops below ~1). That truncated most jobs' visible history
to their first 5-7 epochs. Re-extracted with a fixed regex at the ~51min check-in (epoch
16-19 across jobs); corrected picture below.

**Check-in at ~51 min (epoch 16-19 across all 8 jobs). All 8 PIDs alive, no crashes.**

- **`warmstart_full_medium_simpleenc` recovered**: the epoch-6 H_theta/H_usage collapse
  (0.63->0.09) was transient, not permanent -- by epoch 16-18, H_theta is back to 0.2-0.5 and
  H_usage to 2.0-2.9, rec_acc steady ~0.39. This is the best-known-baseline config (0.740
  AUC); it's healthy and progressing normally (18/60 configured epochs).
- **The earlier `_up`-collapses-`_down`-stays-healthy split did not hold** at this later
  checkpoint: `v7_vaewarmup_up_simpleenc` and `_up_15_to005` both recovered by epoch 16-18
  (H_theta 0.4-0.6, H_usage 2.3-2.9, rec_acc 0.37-0.43) -- the early collapse looks like a
  transient phase most jobs pass through, not a fixed fate tied to warmup direction.
- **`v7_vaewarmup_down_15_to0_simpleenc` is the one genuinely concerning job**: healthy and
  improving through epoch 13 (rec_acc climbed 0.71->0.93, H_usage ~2.7-2.9), then a sharp
  destabilization at epoch 14 -- rec jumped 0.43->4.88, rec_acc crashed 0.85->0.08,
  H_theta/H_usage collapsed to ~0.01-0.04. Epochs 15-18 show partial recovery (H_usage back
  up to 1.6 by epoch 18) but `rec` is still ~5-6, far worse than its pre-collapse best
  (~0.21). Worth continuing to watch; if it doesn't fully recover in the next check, it's the
  clearest kill/restart candidate of the 8.
- **`v7_tmlabel_overfit_simpleenc` correction**: this run is not evaluated on `rec_acc` (per
  user) -- it's designed to overfit the TM/label objective, so falling reconstruction
  accuracy here is expected, not a red flag. On its actual metrics: `label_acc` ~0.55-0.56
  (in line with every other job), `tm` stable ~10.39, `tm_gain` oscillating positive
  (0.03-0.62), H_theta/H_usage still pinned near 0 (0.003-0.12) through epoch 18 and
  `phi_l2` elevated (~11-16) -- consistent with a low-entropy/concentrated topic assignment,
  which may well be the intended behavior for this diagnostic run rather than a bug. Not
  flagging this as a kill candidate.

**Check-in at ~78 min (epoch 26-28 across all 8 jobs, warmstart_full_medium at 28/60 =
47% done). All 8 PIDs alive, no crashes.**

- **`v7_vaewarmup_down_15_to0_simpleenc` has NOT recovered -- flag as kill/restart
  candidate**: 14+ epochs after its epoch-14 destabilization, `H_theta`/`H_usage` have come
  back structurally (H_usage 1.7-2.9, close to its pre-collapse ~2.7-2.9) but `rec`/`rec_acc`
  have not -- still `rec`~5.5-6.2, `rec_acc`~0.27-0.28, versus its own pre-collapse best of
  `rec`~0.21, `rec_acc`~0.93 at epoch 8. It also had a fresh `kl_d` spike to 24.2 at epoch 26
  (back down to ~6.1-6.3 since). Reads as settled into a substantially worse local optimum
  after the shock rather than a genuine recovery. Not killed yet -- flagging for a decision
  rather than acting unilaterally, since it's still occupying a GPU and may or may not be
  worth restarting from a pre-collapse checkpoint (if one was saved) vs. left to keep running.
- **`warmstart_full_medium_simpleenc`** (0.740-AUC baseline): healthy, `label_acc` climbed to
  ~0.59-0.61 -- the first job to break past the ~0.55 plateau every job was stuck at through
  epoch 18. Progressing normally at 28/60 configured epochs.
- **`v7_tmlabel_overfit_simpleenc`**: `H_theta`/`H_usage` are no longer pinned near 0 --
  recovered to 0.42-0.50 / 2.46-2.53 by epoch 26-28 (were ~0.003-0.12 at epoch 18). `label_acc`
  ~0.56-0.57, in line with other jobs. (Per standing correction, not judging this run on
  `rec_acc`.)
- **Remaining 5 jobs** (`up`, `up_15_to005`, `down`, `down_15_to005`, `up_15_to1`) all look
  healthy: H_theta 0.2-0.9, H_usage 2.5-3.1, label_acc 0.53-0.59, no new instability.

**Check-in at ~104 min (epoch 36-38). All 8 PIDs alive, no crashes.**

- **`v7_vaewarmup_down_15_to0_simpleenc` confirmed plateaued, not recovering**: epoch 36-38
  shows `rec`~6.1-6.2 / `rec_acc`~0.27, essentially unchanged from epoch 26-28
  (`rec`~5.5-6.2 / `rec_acc`~0.27-0.28) despite H_theta/H_usage being fully healthy
  (0.15-0.20 / 3.0-3.1) the whole time. Ten epochs with a healthy topic model and flat,
  ~20x-worse reconstruction than its own pre-collapse best (rec_acc 0.93 @ epoch 8) is a
  settled bad state, not a slow recovery. Still running, still not killed -- this is a
  decision point for the user, not something to act on unilaterally.
- **`warmstart_full_medium_simpleenc`** (0.740-AUC baseline): 38/60 epochs (63%), still
  improving -- `rec_acc` climbed to ~0.48 (from ~0.39-0.41 at epoch 26-28), `label_acc`
  ~0.57-0.59. At its observed pace (~2.3 min/epoch) it has ~22 epochs / ~50 min left before
  hitting the 60-epoch ceiling -- next check should catch it at or near completion.
- All other jobs continuing along their established healthy trajectories, no new events.

**Check-in at ~135 min (epoch 48-50). All 8 PIDs alive, no crashes.**

- **`warmstart_full_medium_simpleenc`**: 50/60 epochs (83%), still improving -- `rec_acc`
  ~0.52, `label_acc` ~0.57-0.58. At its observed pace (~2.5 min/epoch) has ~10 epochs / ~25
  min left. Has not finished yet (PID 355144 / GPU 0 still active).
- **`v7_vaewarmup_down_15_to0_simpleenc` still flat**: `rec`~6.0-6.2 / `rec_acc`~0.275,
  unchanged for 22+ epochs since its epoch-14 collapse, versus its pre-collapse best of
  `rec_acc`=0.93. Confirmed settled, not recovering. Still running on its own GPU pending a
  user decision on whether to kill/restart it -- not touched yet.
- **`v7_tmlabel_overfit_simpleenc`**: now has the highest `label_acc` of any job (~0.58-0.59),
  H_theta/H_usage healthy (~0.28-0.30 / 2.87-2.96). (Not judged on rec_acc, per standing
  correction.)
- Remaining jobs unchanged in character from prior check-ins, no new instability.

**Check-in at ~161 min. `warmstart_full_medium_simpleenc` FINISHED its full 60 epochs and
exited cleanly (PID 355144 gone, GPU 0 free).** Restored the best-epoch checkpoint (epoch 51)
and wrote `/Warehouse/Andrei/emerson/model/warmstart_full_medium_simpleenc/model.pt`. Final
epoch 59 training metrics: rec_acc=0.543, label_acc=0.591. This is the config that produced
the 0.740-AUC baseline; **training-loop label_acc is not the same as held-out AUC** -- a real
eval run against the held-out split is needed before comparing this checkpoint to the 0.740
number. Not run yet, pending user request.

The other 7 (300-epoch configs) are all at epoch 60/300 (20%), all PIDs alive:
- **`v7_vaewarmup_down_15_to0_simpleenc` still flat**: rec_acc=0.255, unchanged in character
  since its epoch-14 collapse (~46 epochs ago now). Confirmed settled at a much worse state
  than its pre-collapse best (0.93). Still running, still untouched, still pending a user
  decision to kill/restart.
- **`v7_tmlabel_overfit_simpleenc`**: label_acc now 0.594, the best of any job. H_theta/H_usage
  healthy (0.30/3.09). (Not judged on rec_acc.)
- Rest (`up`, `down`, `up_15_to005`, `down_15_to005`, `up_15_to1`) all healthy, label_acc
  0.53-0.58, no new instability.

**Check-in at ~192 min. `v7_vaewarmup_up_15_to1_simpleenc` also finished (PID 355138 gone,
GPU freed) -- not a crash, it early-stopped at epoch 66 of its 300-epoch budget.** Notably,
"Restored the model state at the best epoch (6)" -- its early-stop criterion picked epoch 6
as best, meaning validation performance never improved past that point in 60 further epochs
of training, even though training `rec_acc` kept climbing to 0.74 by epoch 66. Checkpoint
written to `/Warehouse/Andrei/emerson/model/v7_vaewarmup_up_15_to1_simpleenc/model.pt`.

The remaining 6 jobs are all at epoch ~72/300 (24%), all PIDs alive:
- **`v7_vaewarmup_down_15_to0_simpleenc`**: rec_acc=0.276, still exactly flat -- now 58+
  epochs (nearly an hour of wall time) since its epoch-14 collapse with zero net movement.
  This has moved past "might still recover" -- it's a settled failure state. Recommend
  killing and restarting from a pre-collapse checkpoint (if one exists) or just letting the
  GPU go to something else, rather than continuing to let it run unproductively. Still not
  touched -- explicit user call needed.
- Rest continuing healthy, no new instability. `v7_tmlabel_overfit_simpleenc` label_acc
  0.589-0.596, still the group's best.

**Check-in at ~223 min (epoch 82-84/300, 27-28%). All 6 remaining PIDs alive, no new
finishes/crashes.** No material change from the prior check-in.

- **`v7_vaewarmup_down_15_to0_simpleenc`**: rec_acc=0.29, still in the same flat 0.25-0.29
  band it's been in since the epoch-14 collapse (70+ epochs now). Still a settled failure
  state, still recommended for kill/restart, still untouched -- awaiting explicit user
  decision.
- All other 5 jobs continuing their established healthy trajectories.

**Check-in at ~254 min. `v7_vaewarmup_up_simpleenc` also finished (PID 355143 gone, GPU
freed) -- clean early stop at epoch 88, restored best checkpoint from epoch 28.** Same
pattern as `v7_vaewarmup_up_15_to1_simpleenc`: substantial further training after the
recorded "best" epoch, restored checkpoint reverts well back. Checkpoint written to
`/Warehouse/Andrei/emerson/model/v7_vaewarmup_up_simpleenc/model.pt`.

5 jobs remain, epoch 94-96/300 (~32%), all PIDs alive. `v7_vaewarmup_down_15_to0_simpleenc`
still flat (rec_acc 0.29-0.30, unchanged), still a settled failure state, still untouched.
No other news.

**Check-in at ~285 min (epoch 106-108/300, ~36%): steady progress, no finishes, no crashes,
`down_15_to0` still flat (rec_acc~0.27-0.28). No news.**

**First real held-out `evaluate-model` results on the simplified encoder (heads 8->2,
mean-only pooling) -- and they are a clear regression from the 0.740-AUC baseline:**

| run | `theta_features.roc_auc` | `theta_features_cv.roc_auc_mean` | `quantile_features.roc_auc` |
|---|---|---|---|
| `warmstart_full_medium` (original, pre-simpleenc, reference) | **0.7401** | -- | -- |
| `warmstart_full_medium_simpleenc` (same recipe, simplified encoder) | **0.6007** | 0.6447 | 0.5028 |
| `v7_vaewarmup_up_simpleenc` | 0.6194 | 0.6696 | 0.4631 |
| `v7_vaewarmup_up_15_to1_simpleenc` | 0.6660 | 0.6468 | 0.5094 |

`warmstart_full_medium_simpleenc` is the most direct comparison -- identical recipe to the
0.740-AUC run except `attention_heads: 8->2` and `pooling: "mean_max"->"mean"` -- and it
dropped 14 points of AUC (0.740 -> 0.601). The other two simpleenc runs land slightly higher
(0.62-0.67) but all three are well below the original encoder's baseline. This strongly
suggests the encoder simplification (fewer attention heads and/or dropping the max-pooling
component) cost real held-out classification signal, even though train-loop metrics
(`label_acc` ~0.55-0.62 across these runs) looked broadly comparable across the whole
session and gave no early warning of this gap -- consistent with this document's standing
lesson that training-loop numbers don't reliably predict test AUC. `quantile_features` (the
per-sequence phi-score-based features) are near chance (0.46-0.51) for all three, same as
theta features carrying essentially all the signal here.

Two more simpleenc checkpoints (`v7_vaewarmup_up_15_to005_simpleenc` at epoch ~108,
`v7_vaewarmup_down*` variants) have not finished/been evaluated yet -- worth checking whether
any of the `_down` variants (healthier training dynamics throughout) close the gap any more
than `_up` did, but the `warmstart_full_medium_simpleenc` result alone is discouraging enough
that reverting to the original 8-head/mean_max encoder for the next iteration is worth
considering.

**Killed** all 5 remaining `_simpleenc` (heads=2, pooling="mean") jobs (`v7_vaewarmup_down`,
`down_15_to005`, `up_15_to005`, `down_15_to0`, `v7_tmlabel_overfit`) and relaunched all 8
recipes with a new isolated variant (suffix `_simpleenc_mm`): `attention_heads: 8->2` kept,
but `pooling` reverted to `"mean_max"` (unchanged from the original) -- to separate which of
the two bundled changes caused the AUC regression. Each new config is a verified 2-line diff
from the *original* (pre-simpleenc) config: only `attention_heads: 8->2` in both encoder and
decoder params, nothing else. All 8 launched on GPUs 0-7, confirmed alive and past
data-loading. The 3 finished mean-only-pooling checkpoints (`warmstart_full_medium_simpleenc`
0.601 AUC, `v7_vaewarmup_up_simpleenc` 0.619, `v7_vaewarmup_up_15_to1_simpleenc` 0.666) and
their eval reports remain on disk for comparison once the `_mm` counterparts finish and are
evaluated.

**Check-in at ~21 min (epoch 0 across all 8 `_mm` jobs). All 8 PIDs alive.** Same early
pattern as the mean-only run's epoch 0: `_down`/`_down_15_to005`/`_down_15_to0` start healthy
(H_theta~3.36, H_usage~3.38, rec_acc~0.82), `_up`/`_up_15_to005`/`_up_15_to1`/
`tmlabel_overfit` start with H_theta/H_usage pinned near 0 (~0.012), consistent with their
warmup schedule at epoch 0, not a bad sign this early -- the mean-only run showed the exact
same epoch-0 split and most `_up` variants recovered by epoch 5-20 last time. Too early to
draw any conclusion about whether `mean_max` pooling fixes the AUC regression.

**Check-in at ~52 min (epoch 3-5). All 8 PIDs alive.** Trajectory so far mirrors the
mean-only run closely: `_down`/`_down_15_to005`/`_down_15_to0` healthy (rec_acc~0.98,
H_theta~0.5-0.8, H_usage~2.4-2.7), `_up`/`_up_15_to005`/`tmlabel_overfit` still collapsed
(H_theta/H_usage ~0-0.02), `warmstart_full_medium_simpleenc_mm` starting to recover (H_theta
0.06->0.54 between epoch 4-5, same shape as the mean-only run's epoch-5-6 transition). `label_acc`
flat ~0.55 everywhere, same as the mean-only run at this stage. No distinguishing signal
between `mean_max` and `mean` pooling yet -- still too early.

**Check-in at ~100 min (epoch 10-11). All 8 PIDs alive, no finishes yet -- width-experiment
trigger (see standing instruction below) not yet applicable.** Trajectories continuing to
develop; one notable blip: `v7_vaewarmup_up_simpleenc_mm` label_acc jumped to 0.6705 at
epoch 11 (vs. the ~0.55 plateau every job has shown at this stage all session) alongside
H_theta/H_usage recovering (0.25->0.47 / 1.39->2.00). Could be small-validation-split noise
(same caveat this document has raised before about training-loop numbers) or a real signal --
watch whether it holds.

**Standing instruction (2026-09-14 ~21:16 UTC)**: once these 8 `_mm` jobs finish and are
evaluated, if none beat the 0.7401 baseline, launch a widened-transformer follow-up
(`attention_dim: 64->128`, `attention_heads: 2->4`, `attention_dim_head` stays 32, pooling
stays `mean_max`) on the same 8 recipes, suffix distinct from existing names, GPUs from
natural completions only (no killing). If any `_mm` result beats 0.7401, report and ask
before proceeding further.

**Check-in at ~131 min (epoch 13-15). All 8 PIDs alive, no finishes yet -- width-experiment
trigger still not applicable.**

- **`v7_vaewarmup_up_simpleenc_mm`'s label_acc jump held and grew**: 0.62 (ep13) -> 0.61
  (ep14) -> 0.60 (ep15), sustained well above the ~0.55 plateau every job has shown all
  session, alongside real H_theta/H_usage recovery (0.35-0.39 / 2.16-2.33). Four epochs
  sustained now -- looks like a real signal, not single-epoch noise. `v7_vaewarmup_up_15_to005_simpleenc_mm`
  shows the same pattern emerging (0.55 -> 0.61 -> 0.62 over ep13-15). Worth watching whether
  this survives to a finished/evaluated checkpoint -- train-loop `label_acc` this elevated
  hasn't been seen from any `_up` variant since the original (non-simplified) architecture.
- **`v7_vaewarmup_down_15_to0_simpleenc_mm` destabilized again**: healthy through epoch 13
  (rec_acc 0.88) then a sharp break at epoch 14 (rec 0.34->4.40, rec_acc 0.88->0.21),
  mirroring the exact same failure this recipe showed in the mean-only run (which never
  recovered). Watch whether this one recovers or repeats that outcome.
- **`v7_tmlabel_overfit_simpleenc_mm` anomaly at epoch 15**: `tm_gain` spiked to -71.3 (vs.
  ~0.1-0.5 every prior epoch), `tm` jumped 10.39->11.11, `total` dropped 7.38->1.00, `H_theta`
  jumped from ~0.0002 to 1.19 in one step. Large single-step discontinuity -- not judged on
  rec_acc per standing correction, but worth confirming next check whether this is a one-off
  numerical event or a new regime.

**Check-in at ~162 min (epoch 17-20). All 8 PIDs alive, no finishes yet -- width-experiment
trigger still not applicable.** All three open questions from the last check-in resolved:

1. **The elevated label_acc held**: `v7_vaewarmup_up_simpleenc_mm` epoch 17-19 =
   0.6175/0.6109/0.6192, `v7_vaewarmup_up_15_to005_simpleenc_mm` = 0.5993/0.5911/0.6010 --
   sustained for 7+ epochs now since first appearing around epoch 11-13. This looks like a
   genuine, durable improvement over the ~0.55 plateau every job (including the mean-only
   run) showed all session, not noise.
2. **`v7_vaewarmup_down_15_to0_simpleenc_mm` did not recover**: rec_acc still stuck at
   ~0.27-0.29 through epoch 19, versus its pre-collapse best of 0.90+. Confirmed repeating
   the exact same settled-failure outcome the mean-only run's `down_15_to0` showed.
3. **`v7_tmlabel_overfit_simpleenc_mm`'s epoch-15 anomaly was a real regime shift, not a
   one-off**: H_theta/H_usage settled at 0.80-1.04 / 1.61-2.28 through epoch 17-19 (up from
   ~0.0002 pinned near-zero the whole run before that point), tm_gain back to a normal range
   (0.36-0.46). The topic model broke out of its near-zero-entropy collapse into genuine
   topic diversity partway through training.

**Check-in at ~193 min (epoch 21-24). All 8 PIDs alive, no finishes yet.** Steady
continuation, no new events: `v7_vaewarmup_up_simpleenc_mm` label_acc still holding
0.60-0.61 (9+ epochs sustained now); `warmstart_full_medium_simpleenc_mm` itself climbed to
0.5977 at epoch 24, its highest yet and creeping toward the same elevated band; `down_15_to0`
still flat (rec_acc~0.30, unrecovered). No jobs close to finishing (all still well under
their configured epoch budgets).

**Check-in at ~224 min (epoch 26-28). All 8 PIDs alive, no finishes, no new events.** Steady
continuation of the same picture: `up_simpleenc_mm`/`warmstart_full_medium_simpleenc_mm`
holding ~0.57-0.58 label_acc, `down_15_to0` still flat (rec_acc~0.30-0.31).

**Check-in at ~255 min (epoch 30-32). All 8 PIDs alive, no finishes.** Continuing to climb
gradually: `warmstart_full_medium_simpleenc_mm` 0.581, `v7_vaewarmup_up_simpleenc_mm` 0.571-0.591,
`up_15_to005` 0.588. New wrinkle: `down_15_to0` -- still stuck on reconstruction (rec_acc
~0.31, unrecovered since its epoch-14 collapse) -- now also shows label_acc climbing to
~0.60 (0.6010/0.6026), decoupled from its broken reconstruction. No finishes yet.

**Check-in at ~286 min (epoch 34-36). All 8 PIDs alive, no finishes.** Same gradual climb
continuing: `warmstart_full_medium_simpleenc_mm` 0.579-0.586, `v7_vaewarmup_up_simpleenc_mm`
0.596-0.599, `up_15_to005` 0.583-0.591. No finishes yet, no new events.

**Check-in at ~317 min (epoch 38-40). All 8 PIDs alive, no finishes.** Same gradual climb:
`warmstart_full_medium_simpleenc_mm` 0.576-0.583, `v7_vaewarmup_up_simpleenc_mm` 0.580-0.588.
Minor blip: `down_15_to0`'s `kl_d` spiked to 33.3 at epoch 39 (vs. its typical 8-13 range),
back to normal-ish by the surrounding epochs -- one-off, not treated as urgent.

**Check-in at ~348 min (epoch 42-44). All 8 PIDs alive, no finishes.** Steady climb
continuing (`warmstart_full_medium_simpleenc_mm` 0.573-0.586, `up_simpleenc_mm` 0.583-0.584).
**`down_15_to0`'s `kl_d` instability is escalating, not a one-off**: 20.6 (ep42) -> 48.6
(ep43), well beyond the earlier single 33.3 blip -- still no rec_acc recovery (~0.30). Watch
whether this keeps growing or settles; this job remains the clearest ongoing problem case
among the 8.

**Check-in at ~379 min (epoch 45-48). All 8 PIDs alive, no finishes.** `down_15_to0`'s kl_d
instability self-corrected: peaked at 38.0 (ep45) then 33.0 (ep46), back down to 10.9 by
epoch 47 -- no crash, no NaN, settled back into its usual (still rec_acc~0.30-0.31 flat)
state. Not an escalating problem after all. Rest of the jobs unchanged in character:
`warmstart_full_medium_simpleenc_mm` and `up_simpleenc_mm` still in the 0.58-0.59 label_acc
band.

**Check-in at ~410 min (epoch 50-52). All 8 PIDs alive, no finishes.** Steady, no new events.
`down_15_to0`'s kl_d fully back to normal (10.3-11.9). `warmstart_full_medium_simpleenc_mm`
and `up_simpleenc_mm` holding 0.58-0.58/0.58-0.59 label_acc respectively.

**Check-in at ~441 min (epoch 55-56). All 8 PIDs alive, no finishes, no new events.** Same
steady state as the last few checks.

**Check-in at ~472 min: `warmstart_full_medium_simpleenc_mm` FINISHED** -- full 60 epochs,
best-epoch checkpoint restored (epoch 57), GPU 0 free. This is the direct heads=2/mean_max
counterpart to the 0.7401-AUC original baseline. First launch attempt (PID 409523) silently
failed -- venv wasn't activated in that shell, `nohup: failed to run command 'evaluate-model':
No such file or directory` -- caught at the next check-in when the PID was gone but no
report.json existed. Relaunched correctly (PID 410173, GPU 0, venv activated first), result
pending. Other 7 jobs all alive and unchanged in character (epoch 59-61).

**Check-in at ~511 min: warmstart_full_medium_simpleenc_mm eval reached 100% theta scoring**
(PID 410173 still alive, now in the silent post-scoring stage), report.json not yet written.
**`v7_vaewarmup_up_15_to1_simpleenc_mm` also finished** (early-stopped at epoch 67 of 300,
best checkpoint restored from epoch 7 -- same pattern as its mean-only counterpart, which
also reverted to a very early best epoch). GPU 5 freed; `evaluate-model` launched (PID
412738, venv activated correctly this time). 6 jobs remain running.

**Check-in at ~522 min: `warmstart_full_medium_simpleenc_mm` eval landed --
`theta_features.roc_auc = 0.6767`** (cv mean 0.6824, quantile_features 0.6047). Better than
the mean-only variant (0.601) but still 6+ points below the 0.7401 original baseline -- not
close/ambiguous. Since `warmstart_full_medium` is the single most directly comparable config
to the baseline, this is decisive enough per the standing instruction: not launching a
width experiment is not warranted here, so proceeding with it.

**Widened-transformer experiment launched** (triggered by the above -- mean_max pooling alone
did not recover the lost AUC). Built 8 new configs (suffix `_simpleenc_mm_wide`) from the
`_simpleenc_mm` configs: `attention_dim: 64->128`, `attention_heads: 2->4` in both
encoder_params and decoder_params (attention_dim_head stays 32, pooling stays `mean_max`).
Verified each diff is exactly the intended 4-line change. Only 1 GPU (0) was free at launch
time -- `warmstart_full_medium_simpleenc_mm_wide` launched there (PID 413848, confirmed alive
past data-loading); the other 7 wide configs are queued and will launch as more GPUs free up
from the remaining `_mm` jobs finishing/early-stopping.

**Check-in at ~544 min: `v7_vaewarmup_up_15_to1_simpleenc_mm` eval landed --
`theta_features.roc_auc = 0.6282`** (cv mean 0.6672, quantile_features 0.4641). Second
mean_max-pooling result, also below the 0.7401 baseline (and below `warmstart_full_medium`'s
own 0.6767) -- confirms the pattern rather than changing it. `warmstart_full_medium_simpleenc_mm_wide`
(the widened-transformer job) is at epoch 0-1 on GPU 0, too early to read anything. The other
6 `_mm` (non-wide) training jobs are all still alive, no new finishes this check -- their
corresponding `_wide` configs remain queued for whenever GPUs free up.

**Check-in at ~564 min: launched second wide job.** `v7_vaewarmup_up_15_to1_simpleenc_mm`'s
eval finished, freeing GPU 5 -- `v7_vaewarmup_up_15_to1_simpleenc_mm_wide` launched there
(PID 416765), confirmed alive past data-loading. `warmstart_full_medium_simpleenc_mm_wide`
(GPU 0) at epoch 3, running noticeably slower per-epoch than the non-wide runs (expected --
2x attention_dim, 2x heads means more compute per step); label_acc still flat ~0.55, H_theta
low (0.006-0.17) -- normal early-training collapse phase, nothing concerning yet. Other 5
`_mm` (non-wide) training jobs unchanged, no new finishes. 5 wide configs remain queued
(up_simpleenc_mm, down_simpleenc_mm, up_15_to005, down_15_to005, down_15_to0,
tmlabel_overfit).

**Check-in at ~586 min: `v7_vaewarmup_down_15_to0_simpleenc_mm` finished** (early-stopped at
epoch 79 of 300, best checkpoint restored from epoch 19 -- notably *before* it reached the
kl_d instability episodes flagged in earlier check-ins, and its permanently-degraded
reconstruction plateau; the early-stop selection picked a point pre-dating that whole
saga). GPU 6 freed -- `v7_vaewarmup_down_15_to0_simpleenc_mm_wide` launched there (PID
418199), confirmed alive past data-loading. No free second GPU for its eval this check, so
that's queued for whenever one opens up. Both existing wide jobs (`warmstart_full_medium`
epoch 6, `up_15_to1` epoch 1) progressing normally, nothing conclusive yet. 5 wide configs
remain queued (up_simpleenc_mm, down_simpleenc_mm, up_15_to005, down_15_to005,
tmlabel_overfit).

**Check-in at ~608 min: no new finishes, no new events.** All 3 wide jobs progressing
normally (warmstart_full_medium epoch 8, up_15_to1 epoch 3, down_15_to0 epoch 1) -- no red
flags in any. All 5 remaining `_mm` (non-wide) jobs alive, unchanged. No GPU free for the
queued `down_15_to0` eval.

**Check-in at ~630 min: no new finishes, no free GPU.** Wide jobs showing the familiar
early collapse/health split by warmup direction: `warmstart_full_medium_wide` and
`up_15_to1_wide` both dipped to H_theta/H_usage near 0 at their latest epochs (10 and 5
respectively) after brief earlier flashes of recovery; `down_15_to0_wide` staying healthy
(H_theta 0.71-1.20, rec_acc~0.99). Consistent with the pattern every prior batch has shown
at this early stage -- not treated as a red flag.

**Check-in at ~651 min: no new finishes, no free GPU.** `up_15_to1_wide` recovering (H_theta
0.70->1.15, H_usage 1.21->2.41 over epoch 6-8). `down_15_to0_wide` staying healthy
(H_theta~0.6-0.7, rec_acc~0.99). `warmstart_full_medium_wide` still oscillating near-collapse
(H_theta 0-0.08) at epoch 10-12. Same pattern trajectory as every prior batch at this stage.

**Check-in at ~672 min: no new finishes, no free GPU.** All three wide jobs continuing
along their established trajectories: `warmstart_full_medium_wide` still hovering low
(H_theta 0.09-0.22), `up_15_to1_wide` stabilizing healthy (H_theta 0.73-1.15, H_usage
2.2-2.5), `down_15_to0_wide` staying healthy (rec_acc~0.99). No new events.

**Check-in at ~693 min: no new finishes, no free GPU.** Same trajectories continuing:
`warmstart_full_medium_wide` still low H_theta (0.05-0.09), `up_15_to1_wide` improving
(H_theta up to 0.98, rec_acc up to 0.52), `down_15_to0_wide` steady (rec_acc~0.98-0.99).

**Check-in at ~714 min: `v7_vaewarmup_down_15_to005_simpleenc_mm` finished** (early-stopped
at epoch 101 of 300, best checkpoint restored from epoch 41). GPU 4 freed --
`v7_vaewarmup_down_15_to005_simpleenc_mm_wide` launched there (PID 426642), confirmed alive
past data-loading. No second GPU freed, so `down_15_to0`'s queued eval still waits (now the
longest-waiting item). 4 wide configs remain queued (up_simpleenc_mm, down_simpleenc_mm,
up_15_to005, tmlabel_overfit).

**Check-in at ~735 min: no new finishes, no free GPU.** `warmstart_full_medium_wide`
recovering (H_theta up to 0.22, rec_acc 0.43), `up_15_to1_wide` healthy and stable (H_theta
~1.1, rec_acc~0.58). **`down_15_to0_wide` wobbled**: rec_acc dropped 0.49->0.23 at epoch 15
(rec jumped 3.16->5.13) -- only one epoch of data, but worth watching since its non-wide
counterpart eventually settled into a permanent version of this exact failure. New wide job
`down_15_to005_wide` at epoch 0-1, too early to read.

**Check-in at ~756 min: `down_15_to0_wide` confirmed collapsing, not recovering.** rec_acc
kept falling epoch over epoch -- 0.49 (ep14) -> 0.23 (ep15) -> 0.09 (ep16) -> 0.07 (ep17),
H_theta/H_usage now pinned at 0. This is the same failure mode its non-wide counterpart hit,
at nearly the same epoch (14). Not touched -- letting it run per the same policy as before
(flag, don't kill unilaterally). Other 3 wide jobs continuing to improve normally
(`warmstart_full_medium_wide` H_theta up to 0.34, `up_15_to1_wide` stable at rec_acc~0.60-0.61,
`down_15_to005_wide` early, mixed epoch 2-3). No new finishes among the 3 remaining `_mm`
jobs, no free GPU -- `down_15_to0`'s eval still queued.

**User decision: killed `v7_vaewarmup_down_15_to0_simpleenc_mm_wide`** (PID 418199, GPU 6) --
confirmed collapsing with no recovery in sight, not worth burning the GPU on. GPU 6 used to
finally launch `v7_vaewarmup_down_15_to0_simpleenc_mm`'s long-queued `evaluate-model` (PID
430018), which had been queued since ~06:15 UTC waiting for a GPU. `down_15_to0_wide`'s config
remains available for a future relaunch if wanted, but nothing queued to replace it right
now -- one fewer wide job running (3 of 8 total, plus 4 queued waiting on GPUs).

**`v7_vaewarmup_down_15_to0_simpleenc_mm` eval landed -- `theta_features.roc_auc = 0.7027`**
(cv mean 0.7032, quantile_features 0.5970). By far the best mean_max-pooling result yet,
much closer to the 0.7401 baseline than the other three (`warmstart_full_medium` 0.6767,
`up_15_to1` 0.6282, and this one's own wide counterpart was killed for collapsing). Notable:
this is the SAME checkpoint whose reconstruction was flagged repeatedly all session as a
"settled failure" (rec_acc stuck ~0.30 after an early collapse) -- yet its topic-proportion
features still carry strong classification signal despite broken reconstruction. This
decouples "reconstruction quality" from "classification usefulness" for this architecture in
a way not previously seen this clearly.

**`v7_vaewarmup_up_simpleenc_mm` also finished** (early-stopped at epoch 114 of 300, best
checkpoint from epoch 54). GPU 1 freed -- `v7_vaewarmup_up_simpleenc_mm_wide` launched there
(PID 435021). Its own `evaluate-model` also launched on GPU 6 (PID 435022, freed by the
down_15_to0 eval finishing). Both confirmed alive. 3 wide configs remain queued
(down_simpleenc_mm, up_15_to005, tmlabel_overfit).

**`v7_vaewarmup_up_simpleenc_mm` eval landed -- `theta_features.roc_auc = 0.6733`** (cv mean
0.6771, quantile_features 0.5526). Below the 0.7401 baseline, in the middle of this round's
pack (0.6282-0.7027). GPU 6 freed -- rather than wait for a specific counterpart to finish,
launched the next queued wide config there directly: `v7_vaewarmup_down_simpleenc_mm_wide`
(PID 440630), confirmed alive past data-loading. 2 wide configs remain queued
(up_15_to005, tmlabel_overfit).

**Check-in at ~880 min: no new finishes, no new events.** All 5 wide jobs progressing along
established trajectories (`warmstart_full_medium_wide` still ~0.62 label_acc, `up_15_to1_wide`
healthy rec_acc~0.84, `down_15_to005_wide` steady, `up_wide` in early collapse phase (normal),
`down_wide` early and healthy).

**Check-in at ~840 min: no new finishes, `up` eval still running (471 CPU-min, no report
yet -- consistent with the known slow silent post-scoring stage).** `warmstart_full_medium_wide`
climbing further -- label_acc 0.60-0.62, its highest yet. `up_15_to1_wide` stable
(rec_acc~0.76). `down_15_to005_wide` steady. New `up_wide` job at epoch 0, too early to read.

**Check-in at ~901 min: no new finishes, no new events.** All 5 wide jobs continuing along
their established trajectories, nothing new to report.

**Check-in at ~922 min: `v7_tmlabel_overfit_simpleenc_mm` finished** (early-stopped at epoch
144 of 300, best checkpoint from epoch 84). GPU 7 freed -- `v7_tmlabel_overfit_simpleenc_mm_wide`
launched there (PID 447608), confirmed alive past data-loading. This is the 6th of 8 wide
jobs launched; only `up_15_to005_wide` remains queued, waiting on its own non-wide
counterpart (still training). No free second GPU for `v7_tmlabel_overfit_simpleenc_mm`'s
eval this check. Other 5 wide jobs unchanged, all steady, no red flags.

**Check-in at ~944 min: `v7_vaewarmup_up_15_to005_simpleenc_mm` finished** (early-stopped at
epoch 152 of 300, best checkpoint from epoch 92). GPU 3 freed -- `v7_vaewarmup_up_15_to005_simpleenc_mm_wide`
launched there (PID 450257), confirmed alive past data-loading. **All 8 widened-transformer
jobs are now running** (GPUs 0,1,3,4,5,6,7 plus GPU 2 still held by a non-wide job -- see
correction below). No GPU free for pending evals this check.

**Correction at ~966 min: `v7_vaewarmup_down_simpleenc_mm` was never finished** -- an earlier
check-in wrongly reported it as finished and queued for eval; it has been actively training
the whole time (PID 379132, GPU 2, currently at epoch 158 of 300). Only 2 non-wide checkpoints
are actually finished and awaiting `evaluate-model`: `v7_tmlabel_overfit_simpleenc_mm` and
`v7_vaewarmup_up_15_to005_simpleenc_mm`. All 8 GPUs are occupied (7 wide jobs + this one
still-training non-wide job), so neither pending eval has a free GPU yet. All 7 wide jobs
continuing steadily, no red flags.

**Check-in at ~985 min: no new finishes, no new events.** `down_simpleenc_mm` (non-wide) now
at epoch 165/300. All 7 wide jobs progressing along established trajectories, still no free
GPU for the 2 queued evals.

**Check-in at ~1008 min: `warmstart_full_medium_simpleenc_mm_wide` finished** -- completed
its full 60-epoch budget (config `n_epochs: 60`, unchanged from the narrow/original recipes),
best checkpoint restored from epoch 54. This is the widened-transformer counterpart to the
0.7401 original baseline and the 0.6767 narrow (heads=2) result -- the single most important
comparison in this whole width experiment. GPU 0 freed -- `evaluate-model` launched
immediately (PID 457096), result pending. The 2 previously-queued evals
(`v7_tmlabel_overfit_simpleenc_mm`, `v7_vaewarmup_up_15_to005_simpleenc_mm`) remain queued;
this GPU went to the more urgent flagship-wide result instead.

**Check-in at ~1023 min: `v7_vaewarmup_down_simpleenc_mm` (non-wide) finished** (early-stopped
at epoch 171 of 300, best checkpoint from epoch 111). GPU 2 freed -- used to launch the
longest-queued pending eval, `v7_tmlabel_overfit_simpleenc_mm` (PID 459272), rather than this
job's own eval (judgment call: the tmlabel_overfit eval had been waiting since ~11:35).
`v7_vaewarmup_down_simpleenc_mm`'s own eval and `v7_vaewarmup_up_15_to005_simpleenc_mm`'s eval
remain queued. The flagship `warmstart_full_medium_simpleenc_mm_wide` eval finished theta
scoring (100%) and is now in the slow silent stage (PID 457096, 308+ CPU-min, alive) --
result still pending. Other 6 wide jobs all still running, unchanged.

**Check-in at ~1038 min: the decisive flagship-wide result landed, and it's negative --
widening the transformer made AUC worse, not better.**

| variant (all `warmstart_full_medium` recipe) | heads | attention_dim | pooling | test ROC-AUC |
|---|---|---|---|---|
| original | 8 | 64 | mean_max | **0.7401** |
| narrow (`_simpleenc_mm`) | 2 | 64 | mean_max | 0.6767 |
| **wide (`_simpleenc_mm_wide`)** | **4** | **128** | mean_max | **0.6346** |

`theta_features_cv` mean 0.6329, `quantile_features` 0.4577. Doubling both head count and
attention_dim relative to the narrow variant did not recover any of the AUC lost from the
original 8-head reduction -- it lost further ground, landing below even the narrow result.
This echoes this document's much earlier finding from `overfit_bigenc_full` (a 2x-deeper,
2x-wider transformer on the unopposed-TM diagnostic, months ago): **transformer
encoder/decoder capacity is not the bottleneck for this architecture on Emerson.** Combined
with the mean_max-vs-mean pooling result (mean_max recovers some but not all of the gap), the
best-supported read now is that **head count specifically** (not width, not pooling mode
alone) is where most of the original 0.7401 result's advantage lived -- worth testing
directly once this batch finishes (e.g. attention_dim=64, heads=8, reverting only pooling to
mean_max, isolating heads from the encoder-capacity confound this wide run just ruled out).

GPU 0 freed after this eval -- launched the next queued eval, `v7_vaewarmup_down_simpleenc_mm`
(PID 461484). `v7_tmlabel_overfit_simpleenc_mm`'s eval (PID 459272) still running. Remaining
queue: `v7_vaewarmup_up_15_to005_simpleenc_mm`'s own eval. 6 wide jobs still training,
unchanged.

**Check-in at ~1055 min: `v7_tmlabel_overfit_simpleenc_mm` eval landed --
`theta_features.roc_auc = 0.6841`** (cv mean 0.6889, quantile_features 0.5467). Mid-pack,
below the 0.7401 baseline, in line with the rest of this batch. GPU 2 freed -- launched the
last remaining queued eval, `v7_vaewarmup_up_15_to005_simpleenc_mm` (PID 463499).
`v7_vaewarmup_down_simpleenc_mm`'s eval (PID 461484) still running. **All 8 non-wide
`_simpleenc_mm` checkpoints now have an eval either landed or in progress** -- once these
last two land, the full narrow-batch picture is complete. 6 wide jobs still training,
unchanged.

**Full narrow-batch (heads=2, mean_max) leaderboard so far** (vs. 0.7401 baseline):
`down_15_to0` 0.7027, `tmlabel_overfit` 0.6841, `up_simpleenc_mm` 0.6733,
`warmstart_full_medium` 0.6767, `up_15_to1` 0.6282, `down`/`up_15_to005` pending. None has
beaten baseline; the wide (heads=4) counterpart of the flagship config scored even lower
(0.6346) than its own narrow version, reinforcing that head count -- not width -- is the
lever that mattered in the original 0.7401 result.

**`v7_vaewarmup_down_simpleenc_mm` eval landed -- `theta_features.roc_auc = 0.6874`** (cv
mean 0.6556). Mid-pack, below baseline, consistent with the rest of the narrow batch. All 8
non-wide `_simpleenc_mm` checkpoints now have an eval landed except `up_15_to005` (still
running).

**User-directed "small" variant launched on GPU 0**: `warmstart_full_medium_simpleenc_mm_small`
(PID 465687) -- same recipe as the flagship, further shrinking `attention_dim: 64->32` and
`attention_dim_head: 32->16` (both encoder and decoder), `attention_heads` unchanged at 2,
pooling still `mean_max`. Diff verified as exactly the intended 4-line change. Tests whether
shrinking further continues the pattern seen so far (both narrow heads=2 and wide heads=4
scored below the 8-head original) or reverses it. Note: `attention_dim_head=16` is below the
`x_transformers` rotary-embedding recommendation of >=32 (a warning, not an error, per this
run's own config comment and prior session logs) -- not expected to be fatal but worth
watching for anomalies traceable to that specifically. Confirmed alive past data-loading.

**`v7_vaewarmup_up_15_to005_simpleenc_mm` eval landed -- `theta_features.roc_auc = 0.6804`**
(cv mean 0.7135, notably higher than the single-split test AUC -- a wide gap worth flagging
but not alarming, consistent with this document's repeated finding that single-split test
numbers carry real noise). **This completes the full narrow-batch (heads=2, mean_max)
leaderboard, all 8 evaluated:**

| run | test ROC-AUC |
|---|---|
| `down_15_to0` | **0.7027** |
| `down` | 0.6874 |
| `tmlabel_overfit` | 0.6841 |
| `up_15_to005` | 0.6804 |
| `warmstart_full_medium` (flagship) | 0.6767 |
| `up` | 0.6733 |
| `up_15_to1` | 0.6282 |

Range 0.628-0.703, mean ~0.674 -- every one of the 8 lands 0.04-0.11 below the 0.7401
original (8-head) baseline, none comes close to matching it. Combined with the wide
(heads=4) flagship scoring even lower (0.6346), the narrow-batch spread itself (0.628-0.703)
is comparable in size to the gap from wide to narrow (0.635 to 0.677) -- reinforcing that
this is mostly a real head-count effect, not just seed/config noise swamping a smaller true
difference.

6 wide jobs still training, all steady, no red flags. `warmstart_full_medium_simpleenc_mm_small`
(the user-directed attn_dim=32/dim_head=16 variant) running on GPU 0, too early to read.

**Implemented a new `"attention"` pooling mode for `TransformerEncoder`** (per user
request, in `airrtm/models/encoder.py`): a single learned query attends over positions
(mask-aware softmax), replacing `mean`/`mean_max`'s unweighted averaging with a weighted
one. Mirrors `TopicAttentionPooling`'s rationale (rare/informative elements shouldn't be
diluted by uniform averaging) but at the per-position rather than per-repertoire level --
simpler since positions are a fixed, padded axis (no grouped/ragged softmax needed). Two new
tests added (`tests/test_encoder.py`), full suite passes 86/86. Not yet used in any launched
config -- available as `pooling: "attention"` whenever wanted.

**Check-in at ~1082 min: `v7_vaewarmup_up_15_to1_simpleenc_mm_wide` finished** (early-stopped
at epoch 68 of 300, best checkpoint from epoch 8 -- mirrors its narrow counterpart's very-early
best-epoch pattern). GPU 5 freed -- `evaluate-model` launched (PID 469354), result pending.
`warmstart_full_medium_simpleenc_mm_small` (the attn_dim=32/dim_head=16 variant) at epoch
7-9, showing the same early-collapse trajectory every other run has shown at this stage --
no anomaly traceable to the sub-32 `attention_dim_head`. Other 5 wide jobs unchanged, steady.

**Check-in at ~1103 min: no new finishes, no new events.** `up_15_to1_wide`'s eval still
running (379+ CPU-min, no report yet). `warmstart_full_medium_simpleenc_mm_small` recovering
normally (H_theta up to 0.36, label_acc up to 0.60 by epoch 18) -- still no anomaly from the
low dim_head. Other 5 wide jobs unchanged, steady.

**Check-in at ~1124 min: `v7_vaewarmup_up_15_to1_simpleenc_mm_wide` eval landed --
`theta_features.roc_auc = 0.6849`** (cv mean 0.6701). Mid-pack, consistent with the rest of
this batch and the narrow-batch range.

**`v7_tmlabel_overfit_simpleenc_mm_wide` CRASHED** -- `RuntimeError: Non-finite loss` (all
metrics NaN). Not a clean early-stop; a real divergence. The epochs leading up to it show a
clear signature: H_theta/H_usage pinned at exactly 0.0000 from epoch 22 onward (fully
collapsed) while `kl_d` climbed steadily (14.0->16.6 over epochs 22-29) and `phi_l2` grew
even faster (30.9->38.0, visibly accelerating) -- a collapsed topic model combined with an
unchecked weight/KL blowup, eventually overflowing to NaN sometime after epoch 29. GPU 7 is
now free from the crash; not relaunched automatically (crashes from divergence would likely
recur without a fix, e.g. tighter grad clipping or an LR/coefficient adjustment) -- flagging
for a decision rather than restarting blind.

`warmstart_full_medium_simpleenc_mm_small` continuing to look strong -- label_acc up to
0.6109 by epoch 30, the highest of any run in this whole width/size experiment so far. Other
4 wide jobs (down_15_to005, up, down, up_15_to005) unchanged, steady.

**Check-in at ~1145 min: no new finishes, no new events.** `warmstart_full_medium_simpleenc_mm_small`
holding its lead -- label_acc steady at 0.61-0.61 (ep39-41), `rec_acc` climbing to 0.69, still
the best of any run this whole width/size experiment. Other 4 wide jobs unchanged, no red
flags. GPU 7 still idle, pending a user decision on the crashed `tmlabel_overfit_wide`.

**Check-in at ~1166 min: no new finishes.** `warmstart_full_medium_simpleenc_mm_small`
plateauing on label_acc (0.60-0.61, ep50-52) while rec_acc keeps climbing (0.75-0.76) --
still the best label_acc of any run this experiment. Other 4 wide jobs unchanged, steady.
GPU 7 still idle.

**Check-in at ~1187 min: `warmstart_full_medium_simpleenc_mm_small` FINISHED** -- completed
its full 60-epoch budget, best checkpoint restored from epoch 38 (label_acc=0.611 at that
epoch, the highest training-loop label_acc of anything run this whole width/size experiment).
GPU 0 freed -- `evaluate-model` launched immediately (PID 482275), result pending. This is
the most anticipated eval of the session: does the training-loop lead translate to held-out
AUC, or repeat this document's standing lesson that train-loop numbers don't reliably predict
test AUC? Other 4 wide jobs unchanged, steady, no red flags. GPU 7 still idle pending user
decision on the crashed job.

**`warmstart_full_medium_simpleenc_mm_small` eval landed -- `theta_features.roc_auc = 0.6584`**
(cv mean 0.6550, quantile_features 0.4815). **The training-loop lead did not translate to
held-out AUC.** Despite having the highest training-loop label_acc (0.611) of anything run in
this whole width/size experiment, its test AUC (0.6584) is actually *below* its own narrow
(heads=2, attn_dim=64) counterpart's 0.6767, and below most of the narrow-batch range
(0.628-0.703). This is a clean, direct confirmation of this document's oldest and most
repeated lesson -- training-loop numbers, however striking, do not reliably predict held-out
test AUC. Shrinking further (attn_dim=32, dim_head=16) did not reverse the pattern any more
than widening did; if anything it sits between the narrow and wide results, roughly where a
naive interpolation would put it.

**Updated leaderboard across all three size regimes** (heads/attn_dim/pooling; all mean_max
except where noted; baseline 0.7401):

| run | AUC |
|---|---|
| `down_15_to0` (narrow, heads=2/dim=64) | **0.7027** |
| `down` (narrow) | 0.6874 |
| `up_15_to1_wide` (heads=4/dim=128) | 0.6849 |
| `tmlabel_overfit` (narrow) | 0.6841 |
| `up_15_to005` (narrow) | 0.6804 |
| `warmstart_full_medium` (narrow flagship) | 0.6767 |
| `up` (narrow) | 0.6733 |
| `warmstart_full_medium_small` (heads=2/dim=32/dim_head=16) | 0.6584 |
| `warmstart_full_medium_wide` (heads=4/dim=128 flagship) | 0.6346 |
| `up_15_to1` (narrow) | 0.6282 |

No config in any size regime has come within 0.037 AUC of the original 8-head baseline. GPUs
0, 2, 5, 7 free; 4 wide jobs (down_15_to005, up, down, up_15_to005) still training on GPUs
1/3/4/6; GPU 7 still idle pending user decision on the crashed `tmlabel_overfit_wide` --
nothing new launched without direction.

**4 new jobs launched on the freed GPUs, per user request -- a new pooling/sampling axis
orthogonal to the size sweep above, at both narrow (attn_dim=64/dim_head=32/heads=2) and
small (attn_dim=32/dim_head=16/heads=2) scale, keeping `theta_pooling: "attention"` (the
repertoire-level mechanism) and mean_max unchanged except where noted:**

- **`warmstart_full_medium_simpleenc_mm_attnpool`** (GPU 0, PID 492992): narrow recipe with
  the new per-position `pooling: "attention"` mode (just implemented in
  `airrtm/models/encoder.py`, see above) replacing `mean_max`. Verified 1-line diff.
- **`warmstart_full_medium_simpleenc_mm_contrastive`** (GPU 2, PID 492991): narrow recipe,
  `pooling` unchanged (`mean_max`), with `tm_negative_mode: "opposite_class_pair"` added to
  `loss_config` -- clarified with the user first, since the current baseline already runs the
  *non*-contrastive default (`"batch"`) implicitly; this is the actual contrastive comparison
  point, not a redundant no-op. Verified 1-line diff (an insertion).
- **`warmstart_full_medium_simpleenc_mm_small_attnpool`** (GPU 5, PID 492990): small-encoder
  counterpart of the attnpool run.
- **`warmstart_full_medium_simpleenc_mm_small_contrastive`** (GPU 7, PID 492989):
  small-encoder counterpart of the contrastive run.

All 4 confirmed alive, past data-loading. Results pending. 4 wide jobs (down_15_to005, up,
down, up_15_to005) continue training unaffected on GPUs 1/3/4/6.

**Check-in at ~22 min into the new batch. All 8 jobs alive, no crashes.** `attnpool` jobs
(narrow and small) look completely normal at epoch 0-2 -- no anomaly traceable to the new
per-position attention pooling mode; H_theta/H_usage, rec_acc, label_acc all in the usual
early-epoch ranges. `contrastive` jobs show one expected quirk worth noting so it isn't
re-flagged later: `tm` starts at ~9.70 instead of the usual ~10.39-10.40 uniform baseline,
and `tm_gain` sits around **69** (vs. every other run's -1 to +1 range). This is architectural,
not a bug -- `tm_negative_mode="opposite_class_pair"` restricts the TM partition function's
negative pool to same+opposite-class repertoires instead of the whole batch, which shifts
both the absolute scale of `tm` and whatever `tm_gain` measures relative to. Everything else
(rec_acc, label_acc, H_theta/H_usage) looks ordinary for both contrastive jobs. 4 wide jobs
unchanged, steady, no new finishes.

**Check-in at ~169 min into the new batch: `v7_vaewarmup_down_15_to005_simpleenc_mm_wide`
finished** (early-stopped at epoch 113 of 300, best checkpoint from epoch 53). GPU 4 freed --
`evaluate-model` launched (PID 512443), result pending. `warmstart_full_medium_simpleenc_mm_contrastive`'s
label_acc is now sustained at 0.60-0.61 across three consecutive epochs (22-24, from
0.5993/0.5877 earlier) -- no longer a single-epoch blip, a real durable elevation above the
~0.55 plateau everything else this session has shown. Attnpool and small_contrastive jobs
otherwise unremarkable -- normal early-training fluctuation. Other 3 wide jobs unchanged,
steady.

**Check-in at ~190 min: `down_15_to005_wide` eval still running (513 CPU-min, no report
yet), all other 7 jobs alive, no new finishes.** `contrastive`'s label_acc sustained 5
consecutive epochs now (22-27, 0.593-0.601) -- increasingly looks like a real, durable signal
rather than noise, though this document's standing lesson (training-loop numbers don't
reliably predict test AUC) still applies until this checkpoint is actually evaluated. Other
jobs unremarkable, steady progress.

**`v7_vaewarmup_down_15_to005_simpleenc_mm_wide` eval landed -- `theta_features.roc_auc =
0.6614`** (cv mean 0.6594). Mid-pack for the wide batch, still below the 0.7401 baseline. GPU
4 freed, currently idle (no queued config for it). `contrastive`'s label_acc still holding
at ~0.594 (9 epochs sustained now, 22-31) -- durable but not yet evaluated against held-out
AUC. Other 6 jobs unchanged, steady, no new finishes.

**Check-in at ~230 min: `contrastive`'s label_acc jumped to 0.6507 at epoch 35** (from the
~0.59-0.60 plateau it had held for 9+ epochs) -- the highest of any run this whole session.
But H_theta/H_usage are declining at the same time (H_theta 0.24->0.21->0.17, H_usage
2.71->2.44->1.82 over epochs 33-35) -- ambiguous whether this is a genuine improvement or the
onset of a new collapse phase (this document has seen both patterns produce similar-looking
early trajectories before). Watching closely. Other 6 jobs unchanged, no new finishes. GPU 4
still idle.

**Resolved: `contrastive`'s epoch-33-35 dip was NOT a collapse.** H_usage recovered (1.82 ->
2.34 -> 2.51 -> 2.44 over epochs 35-38) instead of continuing to fall toward 0, and label_acc
settled into a genuinely elevated range (0.586-0.616 across epochs 35-38) rather than reverting
to the ~0.55 plateau -- the epoch-35 0.6507 reading was a one-epoch peak within noise, not
the new steady state, but the surrounding band is still clearly above every other run's
plateau this session. Best read: this is a real, durable improvement from the contrastive TM
sampling, not an artifact. Still unconfirmed against held-out AUC -- that remains the
decisive check once this run finishes. Other 6 jobs unchanged, steady, no new finishes.

**Check-in at ~250 min: no new finishes.** `contrastive` continuing to hold in the elevated
band (label_acc 0.58-0.62, H_usage stable 2.6-2.7 over epochs 40-42) -- still looking durable.
Other 6 jobs unchanged, steady.

**Check-in at ~270 min: no new finishes.** `contrastive` climbing further -- label_acc
0.60-0.62 over epochs 44-46, its best range yet, H_usage stable ~2.7-2.8. Other 6 jobs
unchanged, steady.

**Check-in at ~291 min: `warmstart_full_medium_simpleenc_mm_small_contrastive` and
`warmstart_full_medium_simpleenc_mm_small_attnpool` both finished** -- both completed their
full 60-epoch budgets (best checkpoints epoch 52 and 56 respectively). GPUs 7 and 5 freed --
`evaluate-model` launched on both immediately (PID 528390 small_contrastive, PID 528391
small_attnpool), results pending. `contrastive` (narrow, still training) continuing to hold
in its elevated band, label_acc 0.60-0.61 over epochs 52-54. Other 4 jobs unchanged, steady.

**Check-in at ~306 min: both small-encoder evals still running (normal pace, no report yet).**
`contrastive` still holding strong (label_acc 0.59-0.62 over epochs 55-57). Other 4 jobs
unchanged, no new finishes.

**Check-in at ~322 min: four events at once.**

1. **`small_contrastive` eval landed -- `theta_features.roc_auc = 0.6482`** (cv mean 0.6485).
   Mid-pack, below baseline.
2. **`small_attnpool` eval landed -- `theta_features.roc_auc = 0.6456`** (cv mean 0.6894 --
   another notably wide train/cv-vs-single-split gap, as seen before this session).
3. **`warmstart_full_medium_simpleenc_mm_attnpool` (narrow) finished** -- completed its full
   60-epoch budget, best checkpoint from epoch 58. GPU 0 freed -- `evaluate-model` launched
   (PID 532433), result pending.
4. **`warmstart_full_medium_simpleenc_mm_contrastive` (narrow) finished** -- the run held/
   climbed in an elevated label_acc band (0.59-0.62) for the better part of 300 minutes, the
   most durable training-loop signal of the whole session. Completed its full 60-epoch
   budget, best checkpoint from epoch 49. GPU 2 freed -- `evaluate-model` launched immediately
   (PID 532432) as the top-priority result. **This is the decisive check on whether the
   contrastive TM sampling's sustained training-loop advantage is real or another instance of
   this document's standing lesson that training-loop numbers don't predict test AUC.**

3 wide jobs (up, down, up_15_to005) unchanged, steady, no new finishes.

**Check-in at ~338 min: the decisive `contrastive` result landed -- `theta_features.roc_auc
= 0.6819`** (cv mean 0.6720, quantile_features 0.4791). **The sustained training-loop signal
(label_acc held 0.59-0.62 for ~300 minutes, the most durable elevation seen all session) did
translate into a real improvement -- but a modest one, not a breakthrough.** 0.6819 beats the
mean_max narrow flagship's own 0.6767 by a small margin, and is roughly mid-pack overall,
well short of both the original 0.7401 baseline and the batch's actual best (`down_15_to0`'s
0.7027). Read: `tm_negative_mode="opposite_class_pair"` does appear to help a little over the
default `"batch"` negative sampling on this recipe, but the magnitude of the training-loop
elevation (a full ~0.05-0.07 label_acc lift, sustained for hours) did not translate into a
proportionally large AUC gain (+0.005 over the flagship) -- another instance of this
document's standing lesson that training-loop numbers, however striking, do not reliably
predict the *size* of a held-out AUC change, even when they correctly predict its *direction*.

**`warmstart_full_medium_simpleenc_mm_attnpool` eval landed -- `theta_features.roc_auc =
0.6600`** (cv mean 0.6539, quantile_features 0.5430 -- notably higher than every other run's
quantile_features this session, worth a second look if the per-sequence ranking use case
comes up again). Below baseline, unremarkable relative to the rest of the pooling/sampling
batch.

**Full pooling/sampling-axis leaderboard** (narrow encoder, attn_dim=64/heads=2 unless noted;
vs. 0.7401 baseline and the mean_max flagship's 0.6767):

| variant | AUC |
|---|---|
| `contrastive` (`tm_negative_mode="opposite_class_pair"`) | **0.6819** -- best of this axis |
| `attnpool` (per-position attention pooling) | 0.6600 |
| `small_contrastive` (attn_dim=32/dim_head=16) | 0.6482 |
| `small_attnpool` (attn_dim=32/dim_head=16) | 0.6456 |

3 wide jobs (up, down, up_15_to005) unchanged, steady, no new finishes.

**Check-in at ~358 min: 3 wide jobs (up epoch 140, down epoch 135, up_15_to005 epoch 125) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~378 min: 3 wide jobs (up epoch 143, down epoch 139, up_15_to005 epoch 129) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~398 min: 3 wide jobs (up epoch 147, down epoch 143, up_15_to005 epoch 133) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~418 min: 3 wide jobs (up epoch 151, down epoch 147, up_15_to005 epoch 137) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~438 min: 3 wide jobs (up epoch 155, down epoch 151, up_15_to005 epoch 141) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~458 min: 3 wide jobs (up epoch 159, down epoch 155, up_15_to005 epoch 145) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~478 min: 3 wide jobs (up epoch 163, down epoch 158, up_15_to005 epoch 149) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~498 min: 3 wide jobs (up epoch 167, down epoch 162, up_15_to005 epoch 153) all
alive and steady, no new finishes.** GPUs 0 and 2 idle per standing instruction.

**Check-in at ~518 min: `v7_vaewarmup_up_simpleenc_mm_wide` finished** -- stopped at epoch 168,
best checkpoint epoch 108. GPU 1 freed -- `evaluate-model` launched immediately (PID 567010,
`--skip_baseline`), result pending. `down` (epoch 166) and `up_15_to005` (epoch 157) unchanged,
steady. GPUs 0 and 2 idle per standing instruction.

**Check-in at ~538 min: `v7_vaewarmup_up_simpleenc_mm_wide` eval landed --
`theta_features.roc_auc = 0.6618`** (cv mean 0.6477, quantile_features 0.5732). Below the 0.7401
baseline, unremarkable relative to the rest of this session's wide-batch results. GPU 1 now free
(idle, not reused without request). `down` (epoch 170) and `up_15_to005` (epoch 161) both alive,
steady, no new finishes. GPUs 0 and 2 idle per standing instruction.

**Check-in at ~558 min: `down` (epoch 174) and `up_15_to005` (epoch 164) both alive, steady, no
new finishes.** GPUs 0, 1, and 2 idle per standing instruction.

**Check-in at ~578 min: `down` (epoch 178) and `up_15_to005` (epoch 168) both alive, steady, no
new finishes.** GPUs 0, 1, and 2 idle per standing instruction.

**Check-in at ~598 min: `down` (epoch 182) and `up_15_to005` (epoch 172) both alive, steady, no
new finishes.** GPUs 0, 1, and 2 idle per standing instruction.

**Check-in at ~618 min: `v7_vaewarmup_up_15_to005_simpleenc_mm_wide` finished** -- stopped at
epoch 174, best checkpoint epoch 114. GPU 3 freed -- `evaluate-model` launched immediately (PID
578648, `--skip_baseline`), result pending. `down` (epoch 186) unchanged, steady. GPUs 0, 1, and
2 idle per standing instruction.

**Check-in at ~638 min: `v7_vaewarmup_up_15_to005_simpleenc_mm_wide` eval landed --
`theta_features.roc_auc = 0.6713`** (cv mean 0.6647, pr_auc 0.7018). Below the 0.7401 baseline,
unremarkable relative to the rest of this session's wide-batch results. GPU 3 now free (idle, not
reused without request). `down` (epoch 190) unchanged, alive and steady -- the last remaining job
of the wide-batch (heads=4, attn_dim=128) sweep. GPUs 0, 1, 2, and 3 idle per standing
instruction.

**Check-in at ~658 min: `down` (epoch 194) alive, steady, no new finish.** GPUs 0, 1, 2, and 3
idle per standing instruction.

**Check-in at ~678 min: `down` (epoch 198) alive, steady, no new finish** -- approaching its
`n_epochs`/patience budget, likely to stop soon. GPUs 0, 1, 2, and 3 idle per standing
instruction.

**Check-in at ~698 min: `down` (epoch 202) alive, steady, no new finish** -- past epoch 200, so
its budget is evidently higher than expected; still running. GPUs 0, 1, 2, and 3 idle per
standing instruction.

**Check-in at ~718 min: `down` (epoch 206) alive, steady, no new finish.** GPUs 0, 1, 2, and 3
idle per standing instruction.

**Check-in at ~738 min: `down` (epoch 210) alive, steady, no new finish.** GPUs 0, 1, 2, and 3
idle per standing instruction.

---

## 2026-09-16: two claims in this document are wrong, and the fix for one of them is launched

A fresh read of the model/loss/sampling code against this document's own run logs
turned up three errors in what is recorded above. Full write-up in
[`PLAN_v4_2026-09-16.md`](PLAN_v4_2026-09-16.md); the corrections themselves belong here.

### Correction 1: `tm` is not flat at full scale -- it is undertrained

"The TM loss essentially never leaves 10.397 at 606 repertoires" quotes
`overfit_amortized_full` (gain 0.0078). The matched `theta_mode: free` run,
`overfit_free_full`, reached **0.0324 nats and was still climbing monotonically at
epoch 446** (0.0058 @ep39 -> 0.0160 @ep79 -> 0.0254 @ep239 -> 0.0306 @ep439). Use
`overfit_free_full` as the full-scale reference, not `overfit_amortized_full`.

### Correction 2: the repertoire-count scaling sweep confounded N with optimisation budget

The 2026-09-13 sweep (16/64/128/256) was described as "identical config, only
`--repertoire_slice` varied". It was not: steps/epoch and epoch count were both held
fixed while N varied, so **steps-per-repertoire (= steps x 4 / N) fell as 1/N by
construction**. `config_overfit_free_full.yaml` also ran 604 steps/epoch for 447
epochs = 270k steps, against 17-21k for every other point.

Re-index those runs by steps-per-repertoire and the "gain halves per doubling of N"
pattern disappears:

| steps/rep | 300 (N=256) | 347 (N=606 @ep87) | 600 (N=128) | 1300 (N=64) | 1782 (N=606 @ep447) | 4350 (N=16) |
|---|---|---|---|---|---|---|
| gain (nats) | 0.0132 | 0.0166 | 0.0262 | 0.0618 | 0.0324 | 0.1593 |
| gain / steps-per-rep | 4.4e-5 | 4.8e-5 | 4.4e-5 | 4.8e-5 | 1.8e-5 (saturating) | 3.7e-5 |

Five of six sit at **4.4e-5 +/- 12%**, and at matched budget **N=606 beats N=256 by
1.26x** -- the opposite direction from a capacity story. A rank/topic-count
explanation was considered and rejected: `_tm_log_likelihood_per_sequence` normalises
over the **4 repertoires in the batch**, so Theta never has to separate 606
repertoires, and the rank-30 ceiling at N=606 is ~1.1 nats rather than the ~0.005
observed -- slack by ~200x.

Also: `split_sequences` holds out 10% of each repertoire's *distinct clonotypes*, so
val gain carries no memorisation -- yet `free_16`'s val gain is 0.159, **3x above the
0.052-nat V/J/length "ceiling"** quoted in finding 3. That ceiling ignores CDR3
composition and was not measured under this sampling distribution or the K=4 in-batch
protocol. **Stop quoting 0.052 as a ceiling** until it is recomputed under the
matching protocol.

### Correction 3: clonal abundance is applied twice, and Theta effectively reads ~2-25 clonotypes

`_sample_indices` (`training/train.py:567-593`) draws 8192 indices
`multinomial(p ∝ duplicate_count, replacement=True)`. `_Columns.add`
(`train.py:556-557`) then collects `dataset.weights[indices]` **regardless of how
those indices were sampled**, and those weights reach `_pool_topic_logits`
(`airrtm_model.py:415-431`), where they become `+log(count)` inside
`_grouped_softmax` or a multiplicative weight in `_grouped_mean`.

Effective pooling weight on a clonotype therefore goes as **count^2**. Measured
effective sample size (`1/sum(q^2)`) of that distribution on five random repertoires:
**10.0, 1.7, 24.8, 3.3, 11.4**. Theta -- the only thing `label_input="repertoire"`
reads, and the only thing `theta_features` evaluates -- is determined by roughly
**2 to 25 clonotypes per repertoire**, against the Fisher burden baseline's
~143k-300k distinct clonotypes, abundance ignored. This contaminates every run in
this document.

Note what is *not* a lever: uniform-vs-abundance sampling changes distinct-clonotype
coverage by only 2-6% (measured: 7,372-7,873 distinct drawn in 8192 abundance-weighted
vs 7,846-8,081 uniform). The value of switching is that it stops abundance entering
twice, not that it covers more.

### Cross-cutting: the test split is too small for most of this document's comparisons

70 positive / 82 negative. Hanley-McNeil SE at AUC 0.74 is **+/-0.041**. The encoder
sweep (0.740 / 0.677 / 0.635 / 0.658) spans ~2.5 SE across single runs, so **"head
count is the lever" and "wider is worse" are not established**; and 0.740 is the
maximum over dozens of configs, so winner's-curse puts its true value plausibly at
~0.67-0.70. Standing rule from here: **no single-run comparison under ~0.10 AUC
counts.** Three seeds minimum.

### Code changes

| change | file |
|---|---|
| `AIRRTM_Model(theta_pooling_weights=False)` -- pool Theta over the drawn sequences unweighted, so abundance cannot enter a second time | `models/airrtm_model.py` |
| `CompositeLoss(abundance_weighted_losses=True)` -- restore the abundance-proportional target measure inside the TM and reconstruction likelihoods, normalised to mean 1 *within each repertoire* | `losses/composite_loss.py` |
| `weights` added to `AIRRTM_ModelTarget`, threaded from `Batch` in `_step` | `types.py`, `training/train.py` |

7 new tests (93/93 passing), including a drop-in check that equal abundances reproduce
the unweighted loss exactly, and a check that `theta_pooling_weights=False` makes
Theta invariant to a 1000x clone under both `mean` and `attention` pooling.

### Launched

All 7 free GPUs, plus one CPU job. Smoke-tested on a 12-repertoire slice first.

**Experiment 1 -- abundance applied once** (`config_dedupfix_seed{239,1,2}.yaml`, GPUs
0/1/2): the `warmstart_full_medium` recipe (the 0.7401 run) with
`abundance_weighted_sampling: false` + `theta_pooling_weights: false` +
`abundance_weighted_losses: true`. Verified 3-setting diff, nothing else changed.
Paired against that recipe's existing seeds 239/1/2 (0.740 / 0.734 / 0.700).

**Experiment 3 -- budget-matched scaling at T=60**
(`config_budget_t60_reps{64,128,256,597}.yaml`, GPUs 3/4/5/7): the unopposed-TM
diagnostic with **~600 gradient steps per repertoire held constant across N**
(46/100/200/150 epochs respectively) and `n_topics` 30 -> 60. Prediction under
correction 2: a **flat** gain curve in N. T=60 at matched budget is also directly
comparable to the existing T=30 points, giving the topic-count contrast the original
sweep never had. `patience` = `n_epochs` so each run spends its full budget --
early stopping tracks `total_loss`, which keeps improving on reconstruction long
after `tm_gain` plateaus.

**Experiment 2(a) -- the sampling ceiling** (`analysis/ceiling_sampling.py`, CPU,
committed rather than left in a scratchpad like `fisher_overlap_check.py` and
`representation_probe.py`): gives an *oracle* -- the Fisher-selected clonotype set --
the same bag the model gets, and reports ROC-AUC vs bag size for both sampling
schemes, plus the ESS of the uniform/count/count^2 pooling weights. If the oracle
scores ~0.70 from an 8192 bag, then 0.740 is already at the sampling ceiling and the
lever is bag size or vocabulary restriction, not architecture. Results pending.

### Experiment 2(a) landed: the sampling ceiling is not a ceiling, and abundance sampling *helps*

`analysis/ceiling_sampling.py`, 606 train / 152 test, Fisher selecting 528 clonotypes
at p<1e-3. The oracle is the selected set itself, scored as
`hits / distinct_in_bag` -- i.e. `BurdenScoreClassifier.score`'s own statistic
computed on the bag the model actually gets, 20 draws per repertoire.

| bag | abundance-weighted (w/ replacement) | uniform over distinct (w/o replacement) |
|---|---|---|
| 1,024 | 0.868 | 0.736 |
| 4,096 | 0.917 | 0.783 |
| **8,192** (every training config) | **0.921** | 0.869 |
| 16,384 | 0.921 | 0.921 |
| 32,768 | 0.938 | 0.897 |
| **65,536** (4x16384, evaluation time) | **0.937** | 0.899 |
| whole repertoire (Fisher's own view) | 0.896 | -- |

Three results, two of them surprising:

1. **There is no subsampling ceiling.** An 8192-sequence bag supports oracle AUC
   **0.921** -- *above* the whole-repertoire 0.896, and far above the 0.740 the best
   model achieves. The hypothesis that 0.740 is at the sampling ceiling is dead: the
   information is in the bag and the model is not extracting it. The gap is
   architecture/objective, not sampling. (This also closes the old "bag 1024 is too
   small" concern -- the oracle gets 0.868 at 1024.)
2. **Fixed-size subsampling beats reading the whole repertoire**, because
   `hits / n_distinct` has a depth-dependent denominator; a fixed bag normalises
   sequencing depth away. Worth remembering as a cheap trick for the baseline itself.
3. **Abundance-weighted sampling is worth ~0.05 oracle AUC at bag 8192** (0.921 vs
   0.869). Expanded clones really do carry the CMV signal, so abundance belongs in the
   draw. This contradicts the design of the `dedupfix` arm launched earlier today,
   which moved abundance out of the sampler and into the likelihoods -- see below.

**Effective sample size of the three pooling-weight schemes**, median over the 152
test repertoires -- the count^2 number is the one the shipped code actually applies:

| pooling weight | median ESS | min | max |
|---|---|---|---|
| uniform | 182,241 | -- | -- |
| count (abundance once) | 5,500 | -- | -- |
| **count^2 (sampler + attention, i.e. current code)** | **6.9** | -- | -- |

Theta is currently built from a median of **seven** clonotypes per repertoire. Correction
3 above is confirmed with a direct measurement, and it is worse than the five-repertoire
spot check suggested.

### Consequence: a second, better arm of experiment 1 (`poolfix`)

Given (3), the right minimal fix is to remove the **second** application of abundance
and keep the first. `config_poolfix_seed{239,1,2}.yaml` is a verified **one-line diff**
from the 0.7401 `warmstart_full_medium` recipe -- `theta_pooling_weights: false`,
nothing else -- so it isolates the count^2 defect exactly. Queued to launch on GPUs
3/4/5 as each `budget_t60` run finishes.

`dedupfix_seed{239,1,2}` (abundance moved to the likelihoods, uniform sampling) is
left running as the contrasting arm: the oracle says its sampling scheme is ~0.05
worse, so if it nevertheless matches `poolfix`, the pooling fix is doing the work; if
it lands ~0.05 below, the oracle's ranking transfers to the real model too.

### Experiment 3, first budget-matched point: topic count buys 1.2x, budget buys more

`budget_t60_reps64` finished its full 46-epoch / 600-steps-per-repertoire budget.
Compared against `overfit_free_64` (T=30, same N, same batch) read at **its own epoch
46, which is the same 600 steps per repertoire** -- so this is a like-for-like
topic-count contrast, the one the original sweep never ran:

| N=64, 600 steps/repertoire | best `tm_gain` (x100) | nats |
|---|---|---|
| T=30 (`overfit_free_64` @ep46) | 4.148 | 0.0415 |
| T=60 (`budget_t60_reps64`, full run) | 5.044 | **0.0504** |
| T=30 at **1300** steps/rep (same run, full) | 6.177 | 0.0618 |

**Doubling the topic count buys ~1.22x; roughly doubling the optimisation budget buys
~1.49x.** Budget is the larger axis, consistent with correction 2. This also kills the
withdrawn rank law's quantitative form for good -- that predicted ~2x from doubling T.

**But "flat in N at matched budget" is not holding either.** At 600 steps/repertoire,
T=30 gives 0.0415 at N=64 and 0.0262 at N=128 -- a 1.58x drop for a 2x in N, against
epoch-to-epoch noise of ~15%. So the original sweep's 1/N decay was *partly* a budget
artifact and *partly* real; neither "it's all budget" nor "it's all N" survives. The
remaining two points (N=256, N=597 at T=60) will show whether the residual N-dependence
keeps halving or flattens out.

### Watch item: both experiment-1 arms are collapsing deeper than the baseline did

At epoch 19 the 0.7401 baseline (`warmstart_full_medium`) had recovered to
H_theta 0.19 / H_usage 2.21, having bounced through its own collapse at epochs 1-4.
All three `dedupfix` seeds sit at H_theta 0.0007-0.013 / H_usage 0.006-0.014 at the
same epoch, and `poolfix_seed239` looks the same at epoch 4. That is 4/4 runs failing
to follow the baseline's recovery. Flagged, not acted on: this document's own
2026-09-11/12 finding is that **collapse does not predict AUC** (collapsed runs scored
0.56-0.72, healthy ones 0.57-0.74, fully overlapping), and the decisive metric is
`evaluate-model` on the held-out 152, not the training-loop entropy trace.

### Experiment 3, second and third budget-matched points: doubling topics buys ~1.2-1.5x

`budget_t60_reps128` also finished its full 600-steps-per-repertoire budget. Reading
each T=30 run at the epoch where it has the *same* steps-per-repertoire gives three
like-for-like topic-count contrasts:

| N | steps/rep | T=30 gain (nats) | T=60 gain (nats) | T=60 / T=30 |
|---|---|---|---|---|
| 64 | 600 | 0.0415 | 0.0504 | 1.22x |
| 128 | 600 | 0.0262 | 0.0365 | 1.39x |
| 256 | 300 | 0.0132 | 0.0195 | 1.47x |

Doubling the topic count is worth ~1.2-1.5x, **sub-linear** -- and the ratio *grows*
with N, which is the one thing the withdrawn rank law got directionally right: topic
count matters more when topics are scarcer relative to repertoires. It is nowhere near
the ~2x per doubling that law predicted, so "T ~= 287 topics at N=606" stays retracted.

And the repertoire-count decay at matched budget, now on both topic counts:

| | N=64 -> N=128 at 600 steps/rep |
|---|---|
| T=30 | 0.0415 -> 0.0262 (1.58x) |
| T=60 | 0.0504 -> 0.0365 (1.38x) |

versus the ~2.4x per doubling the original, budget-unmatched sweep reported. **Roughly
half of the observed 1/N decay was an optimisation-budget artifact and roughly half is
a real repertoire-count effect.** Neither "it's all budget" (my prediction) nor "it's
all N" (the original conclusion) survives contact with the matched runs.

### First held-out result: the `dedupfix` arm is a clear regression, as the oracle predicted

`dedupfix_seed1` (early-stopped epoch 33 of 60) evaluated on the unchanged 152-repertoire
test split:

| | `dedupfix_seed1` | its paired baseline `warmstart_full_medium_seed1` |
|---|---|---|
| `theta_features.roc_auc` | **0.6159** | 0.734 |
| `theta_features_cv.roc_auc_mean` | 0.6029 +/- 0.035 | 0.704 +/- 0.010 |
| `quantile_features.roc_auc` | **0.6003** | -- |
| train `theta_features` ROC-AUC | 0.8021 | -- |

A 0.12 drop against the paired seed -- ~3 SE, so a real regression rather than noise.
This is the arm that moved abundance *out of the sampler* and into the likelihoods, and
`analysis/ceiling_sampling.py` had already predicted it would lose ground (uniform
sampling is worth ~0.05 less oracle AUC at bag 8192 than abundance-weighted). The
measured drop is larger than the oracle's, so the sampling scheme is not the whole
story, but the direction was called in advance. **Abundance-weighted sampling stays.**

One thing worth flagging rather than burying: `quantile_features` = **0.6003**, at the
top of the range this document has ever recorded for the per-sequence pathway (0.44-0.60
across every prior checkpoint, mostly 0.46-0.55). The per-sequence readout improved while
Theta got worse. If the sequence-attribution objective is ever picked back up, uniform
sampling over distinct clonotypes is worth re-testing there specifically.

The arm that actually tests correction 3 is `poolfix` (abundance kept in the sampler,
removed only from Theta's pooling -- a one-line diff from the 0.7401 recipe). Three
seeds running, results pending; nothing about the count^2 defect is settled by the
`dedupfix` result above.

### `dedupfix` is dead: all three seeds, paired, land 0.16 below baseline

| seed | `dedupfix` | paired `warmstart_full_medium` | delta |
|---|---|---|---|
| 239 | 0.5815 | 0.740 | -0.159 |
| 1 | 0.6159 | 0.734 | -0.118 |
| 2 | **0.4934** (below chance) | 0.700 | -0.207 |
| **mean** | **0.564** | **0.725** | **-0.161** |

Three paired seeds, mean gap 0.161 against a per-run SE of ~0.041 -- decisive, not the
kind of difference this document has been chasing in noise. **Moving clonal abundance
out of the sampler and into the likelihood terms is a large regression.**
`analysis/ceiling_sampling.py` predicted the direction (uniform sampling costs ~0.05
oracle AUC at bag 8192); the realised cost is 3x that, so the model loses more from the
sampling change than an oracle does.

Incidental pattern worth recording: the three seeds rank inversely with training length
(seed1 stopped at epoch 33 -> 0.616; seed239 at 51 -> 0.582; seed2 ran the full 60 ->
0.493), consistent with this arm degrading as it trains rather than converging.

### Experiment 3: the N-decay flattens at matched budget

`budget_t60_reps256` finished its full 600-steps-per-repertoire budget. The T=60 curve
at a *constant* 600 steps per repertoire:

| N | tm gain (nats) | ratio to previous |
|---|---|---|
| 64 | 0.0504 | -- |
| 128 | 0.0365 | 1.38x |
| 256 | 0.0316 | **1.15x** |
| 597 | pending (0.0194 at epoch 68 of 150) | -- |

The decay is **flattening**, not halving: 1.38x then 1.15x. The original,
budget-unmatched sweep reported ~2.4x per doubling all the way down. If the trend holds,
full scale at a matched 600-step budget should land near 0.028-0.030 nats -- roughly
**4x the 0.0078 this document recorded for `overfit_amortized_full`**, and close to
`overfit_free_full`'s 0.0324 after 270k steps. The picture that is emerging: the TM
branch is not defeated by repertoire count, it was starved of optimisation budget, and
what genuine N-dependence exists is concentrated at small N.

---

## An untrained model scores 0.673. Read every number in this document against that.

`analysis/eval_checkpoints.py --random_init` builds the flagship architecture from
`config_warmstart_full_medium.yaml`, loads **no weights at all**, and runs the exact
`topic_proportion_features` -> `classify_repertoires` path `evaluate-model` uses:

| model | `theta_features` test ROC-AUC |
|---|---|
| Fisher burden baseline | 0.874 |
| **best AIRRTM run ever** (`warmstart_full_medium`) | **0.740** |
| **untrained, randomly initialised network** | **0.6733** |
| majority-class accuracy (not AUC) | 0.536 |

**Everything ~50 epochs of training buys over a random network is +0.067 AUC**, against
a per-run standard error of +/-0.041. The best result this project has produced is
~1.6 SE above an untrained model.

The per-checkpoint curves say the same thing from the other direction:

| checkpoint | `warmstart_full_medium` | `warmstart_full_medium_seed1` | `poolfix_seed239` | `dedupfix_seed2` |
|---|---|---|---|---|
| epoch 0 (after one epoch, 604 steps) | **0.7115** | **0.7301** | 0.6406 | 0.6730 |
| epoch 5 | 0.5906 | **0.7368** | 0.5571 | 0.6147 |
| epoch 10 | 0.6362 | -- | 0.5572 | -- |
| epoch 15 | 0.6513 | -- | -- | -- |
| restored "best" (by `total_loss`) | 0.740 (ep 51) | 0.734 (ep 56) | 0.5131 (ep 46) | 0.4934 (ep 57) |

`warmstart_full_medium` is at 0.7115 after a **single epoch**, collapses to 0.59 by
epoch 5, and spends ~46 more epochs returning to 0.740. `warmstart_full_medium_seed1`
has already exceeded its own final score by epoch 5 (0.7368 vs 0.734). Both `poolfix`
and `dedupfix` peak at epoch 0 and decline from there -- so the arm comparison earlier
today was measuring *how fast each arm degrades*, not whether the pooling change helps.

### What this explains

Nearly every open puzzle in this document dissolves into it:

- **Why ~40 configs all land in 0.63-0.74**: that is the initialisation band plus noise.
- **Why seed variance (0.572-0.740) swamps every effect**: it is initialisation variance,
  which is most of the signal.
- **Why collapse does not predict AUC** (2026-09-11/12): `theta_features` was never
  reading the learned topic structure.
- **Why encoder capacity, head count, pooling mode, topic count and warm-start schedules
  all came out null**: there is very little learned signal for them to affect.

The mechanism is straightforward once stated: attention-pooled Theta is a 30-dimensional
projection of a repertoire's sequence-composition distribution, weighted (see correction
3) towards its largest clones. A *random* projection of "what do this person's biggest
clones look like" already separates CMV at 0.67. Training adds 0.067 on top.

### Consequences

1. **The baseline for any future claim is ~0.67, not the 0.536 majority-class accuracy.**
   Every comparison in this document above needs re-reading on that scale; the
   `theta_features` leaderboard's whole dynamic range is roughly 0.67-0.74.
2. **Early stopping on `total_loss` is not fit for purpose.** Restored epochs across one
   batch of runs span 5 to 57, and the selected checkpoint is frequently worse than the
   epoch-0 one. Any future run must select on a held-out AUC-relevant criterion, or
   evaluate the whole checkpoint sequence.
3. A 10-draw random-init control is running to put an error bar on the 0.673 itself.

### Loss-scale audit, and a `normalize_loss_scales` flag

The four likelihood terms are on incomparable scales, so no coefficient in this
document's ~40 configs means what it looks like. Chance values, in nats, before any
coefficient:

| term | chance | floor | natural range |
|---|---|---|---|
| reconstruction CE | ln(21) = **3.0445** | 0 | 3.045 |
| tm (batch softmax, 4x8192) | ln(32768) = **10.3972** | ln(8192) = 9.0109 | 1.386 (realistically ~0.05) |
| label (weighted BCE, w=0.5) | 0.5*ln2 = **0.3466** | 0 | 0.347 |
| kl (divided by latent_dim) | -- | 0 | unbounded |

Reconstruction's range is 9x the label's; tm's *usable* range is ~30x smaller than the
label's despite carrying the largest coefficient (0.665 vs 0.285 in the flagship).

**What actually moved in `warmstart_full_medium`, the best run this project has:**

| component | chance | epoch 0 | best over 60 epochs | improvement | share of its range |
|---|---|---|---|---|---|
| reconstruction | 3.0445 | 0.6657 | 0.6657 | **0.0000** | 0% (worse after epoch 0) |
| kl | -- | 1.9753 | 1.8567 | 0.1186 | -- |
| tm | 10.3972 | 10.3919 | 10.3919 | **0.0000** | **0%** |
| label | 0.3466 | 0.3452 | 0.3365 | 0.0087 | **2.5%** |

Everything training contributes to the objective after epoch 0 totals ~0.003. This is
the loss-side view of the random-init result above: the same fact, read off the
objective instead of the metric.

**Implemented** `CompositeLoss(normalize_loss_scales=False)` (default off, so every
existing config is bit-identical): divides each term by its own chance value before the
weighted sum, so every term is ~1 at chance and 0 at perfect. Scales are derived from
the batch in hand -- alphabet size from `pad_value`, the TM pool from the actual
negative pool and family, the label's from the batch's realised class weights, the two
entropy terms by `log n_topics` -- so nothing is hard-coded. The **reported** per-term
metrics stay unnormalised, so `tm`/`rec`/`label` in the logs remain directly comparable
to every run above; only `total_loss`, and therefore the gradient balance and early
stopping, change. 4 new tests (98/98 passing).

**Expect this to need re-tuning, not to be a drop-in.** Normalising divides tm's
gradient by 10.4 and multiplies the label's by 2.9, moving the label:tm gradient ratio
from 0.43 to ~12.8 -- a ~30x shift toward the label term. Given FINDINGS' own measured
encoder gradient norms (tm 4.95e-2 vs label 1.25e-2), that flips which branch drives the
shared encoder.

### Random-init control, first 3 of 10 draws

`theta_features` ROC-AUC of untrained networks: mean **0.6827**, sd 0.0151, range
0.6655-0.6937 (n=3 so far). The spread is small, which makes the comparison sharper, not
softer: the best trained run in this document's history (0.740) is about 4 initialisation
SDs above an untrained model, and *most* configs ever run land inside the untrained
range.

### Checkpoint curves: no run improves on its own epoch-0 checkpoint

`theta_features` ROC-AUC every 5 epochs:

| run | ep0 | ep5 | ep10 | ep15 | ep20 | ep25 | ep30 | ep35 |
|---|---|---|---|---|---|---|---|---|
| `warmstart_full_medium` | **0.7115** | 0.5906 | 0.6362 | 0.6513 | 0.6467 | 0.6508 | 0.6468 | 0.7034 |
| `warmstart_full_medium_seed1` | 0.7301 | **0.7368** | 0.6678 | 0.7027 | 0.6934 | 0.6907 | -- | -- |
| `poolfix_seed239` | **0.6406** | 0.5571 | 0.5572 | 0.6482 | 0.5462 | 0.6132 | 0.5157 | 0.5003 |
| `dedupfix_seed2` | 0.6730 | 0.6147 | 0.5634 | 0.6289 | 0.5641 | **0.7041** | -- | -- |

Every curve oscillates inside roughly the untrained band (0.67 +/- 0.015 x a few) with no
trend. `warmstart_full_medium_seed1` peaks at epoch 5 and never returns. The 0.740 that
this whole investigation has been trying to beat is the top of an oscillation, not a
converged value.

### Random-init control complete (n=12), and what the finished curves now say

**Untrained `theta_features` ROC-AUC: mean 0.6842, sd 0.0247, range 0.6505-0.7120.**

Against that, three things fall out of the four completed checkpoint curves:

| run | ep0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | restored |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `warmstart_full_medium` | .712 | .591 | .636 | .651 | .647 | .651 | .647 | .703 | .703 | **.712** | .666 | .707 | **.740** (ep51) |
| `warmstart_full_medium_seed1` | .730 | **.737** | .668 | .703 | .693 | .691 | .687 | .701 | .723 | .704 | .728 | .722 | .734 (ep56) |
| `poolfix_seed239` | **.641** | .557 | .557 | .648 | .546 | .613 | .516 | .500 | .549 | .544 | .535 | .560 | .513 (ep46) |
| `dedupfix_seed2` | .673 | .615 | .563 | .629 | .564 | **.704** | .506 | .585 | .525 | .516 | .555 | .566 | .493 (ep57) |

1. **The 0.740 headline is a lucky epoch.** Adjacent sampled checkpoints around it read
   0.666 (ep50) and 0.707 (ep55), and the every-5 maximum is only 0.7124. Epoch-to-epoch
   swing is ~0.04 -- the same size as the test-set standard error -- so 0.740 is a
   maximum taken over 60 epochs of a noisy oscillation, not a converged value. The
   flagship's honest level is ~0.70, i.e. **0.5-0.7 sd above an untrained network**.
2. **`warmstart_full_medium_seed1` is the better-behaved run** (0.687-0.737 throughout,
   no collapse), yet scores the same. Its whole trajectory sits just above the untrained
   band.
3. **`poolfix` and `dedupfix` fall *below* the untrained floor** and stay there --
   0.50-0.56 for most of training against 0.684 untrained. So reducing Theta's abundance
   weighting does have a real effect after all; it is just that the effect is *training
   actively destroying the signal present at initialisation*, which only became legible
   once the right baseline existed. The count^2 weighting protects that signal.

**Corrected framing for the whole project**: the useful dynamic range of `theta_features`
is roughly **0.68 (untrained) to 0.874 (Fisher)**. Everything AIRRTM has ever produced
occupies the bottom ~10% of it, and several arms sit below the floor.

`budget_t60_reps597`'s per-sequence pathway (`quantile_features`, the only held-out metric
a `theta_mode: free` run supports) reads 0.544 / 0.485 / 0.491 at epochs 0/5/10 -- at or
below chance, and declining. The unopposed-TM diagnostic's rising `tm_gain` is not buying
any held-out sequence-ranking signal.

### Four normalised-loss arms launched

All on `normalize_loss_scales: true`, `checkpoint_every: 2` (the early window is where
everything happens), seed 239:

| run | `tm_likelihood_coef` | warm-start | rationale |
|---|---|---|---|
| `normloss_drop` | 0.70 | on | flagship coefficients, normalisation the only change |
| `normloss_tm03` | 0.30 | on | label-heavy -- starve the term that never moves |
| `normloss_tm09` | 0.90 | on | tm-heavy -- roughly restores the pre-normalisation balance |
| `normloss_nowarm` | 0.70 | **off** | see below |

`normloss_nowarm` is the arm the curves motivated. Every curve peaks at epoch 0 and
troughs around epoch 5, and the flagship warm-start (`tm_likelihood_coef_start: 1.0`,
`coef_anneal_epochs: 8`) runs the model on the TM term *alone*, with the label loss and
both entropy regularisers at zero, across exactly that window. The schedule this document
credits with the 0.740 record is a candidate for what destroys the initialisation signal.

### Fold-in Theta implemented, and its first result is the sharpest control yet

`theta_mode="free"` runs could never be scored on held-out repertoires -- the free
embedding has no row for them -- so `evaluate-model` skipped `theta_features` entirely
and those runs fell back on 12 quantiles of a per-sequence score. That fallback is weak
twice over: it summarises a scalar projection of phi rather than the latent the model
defines, and its quantiles are rank statistics over the whole repertoire, so with
Emerson depths spanning 50k-590k clonotypes the 0.99999 quantile is the top 0.5
sequences in the smallest repertoire and the top 5.9 in the largest -- the feature
itself carries sequencing depth.

**Implemented the standard LDA fold-in** as a single EM E-step
(`airrtm/evaluation/scoring.py::fold_in_topic_proportions`): freeze phi, and note that
under a uniform prior a sequence's topic posterior is `p(t|s) = softmax_t(phi_ts)`
(because `exp(phi_ts)` is the model's unnormalised `p(s|t)`), so

    theta_t  ∝  sum_s softmax_t(phi_ts)

No learning rate, no iteration, and it returns the model's own latent in the model's own
units. `topic_proportion_features(theta_method="auto")` now routes free-Theta models
through it, so `evaluate-model` no longer skips them. Also added
`AIRRTM_Model.predict_topic_logits` (the raw phi; `predict_topic_probabilities` squashes
with a sigmoid, which is the wrong scale for topic-posterior arithmetic) and
`--features theta|quantile` to `analysis/eval_checkpoints.py`. 3 new tests, 101/101
passing. This also removes the stated reason for the guard rejecting
`theta_mode="free"` + `label_input="repertoire"`, unblocking the v1 replication.

**First result, on `budget_t60_reps597`** -- the unopposed-TM diagnostic, stopped at
epoch 100. This run has `tm_likelihood_coef: 1.0`, hence `label_likelihood_coef = 0`:
**it has never seen a label.**

| epoch | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fold-in `theta_features` | **0.7106** | 0.6521 | 0.7033 | 0.6632 | 0.6732 | 0.6871 | 0.6810 | 0.6690 | 0.6754 | 0.6847 | 0.6937 | 0.6855 |
| (train) | 0.857 | 0.856 | 0.869 | 0.859 | 0.856 | 0.865 | 0.853 | 0.869 | 0.851 | 0.864 | 0.857 | 0.857 |

**A model trained with zero label supervision scores 0.65-0.71 -- statistically
indistinguishable from the 0.6842 +/- 0.0247 untrained control, and from the ~0.70 the
best supervised run holds.** Its own epoch 0 is again the peak.

Taken with the random-init control, this closes the question the checkpoint curves
opened: `theta_features` is not measuring learned CMV discrimination. It reads a
projection of repertoire sequence composition that is present at initialisation,
survives 100 epochs of an unrelated objective, and is not improved by label supervision.
Any future claim on this metric has to clear **0.684 (untrained)**, and clear it by more
than the ~0.04 epoch-to-epoch noise.

---

## 2026-09-17: the first genuine learning signal, and a free lever nobody had pulled

Two positive results, from different directions.

### 1. `normloss_tm03` is the first run whose checkpoint curve shows a trend

Normalised loss scales (`normalize_loss_scales: true`) plus `tm_likelihood_coef: 0.3`
-- i.e. fix the term scales, then starve the term that provably never moves. Checkpoint
curve every 2 epochs, against the untrained control of 0.6842 +/- 0.0247:

| epoch | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | **18** | **20** | **22** | **24** | 26 | 28 | 30 | 32 | 34 | 36 | 38 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | .695 | .662 | .662 | .675 | .673 | .704 | .655 | .704 | .700 | **.746** | **.751** | **.762** | **.758** | .725 | .723 | .716 | .700 | .733 | .704 | .737 |

Epochs 0-16 sit inside the untrained band. Then **four consecutive checkpoints at
0.746-0.762**, above the untrained maximum (0.7120) and above this project's previous
record (0.7401), peaking at **0.7623** -- 3.2 initialisation sds above the untrained
mean. It decays to ~0.72 afterwards.

This is the first curve in this document that **rises out of the initialisation band and
stays out for several checkpoints**. Every other run oscillates inside it with no trend.
Caveat: n=1, and epoch-to-epoch noise is ~0.04 -- but four consecutive points above 0.74
is not that noise. Seeds 1 and 2 launched.

Note what it costs: `tm` is completely dead in this arm (10.3966-10.3980, gain
oscillating +/-0.0008 nats, sometimes worse than chance), and reconstruction collapses in
the first three epochs exactly as the baseline's does (rec_acc 0.770 -> 0.362 -> 0.48).
The label term is the only one doing anything, and that is the point of the arm.

### 2. An untrained network with 300 topics scores 0.734

Ablations of the random-init control, 5-7 draws each, no training at any point:

| untrained variant | mean | sd |
|---|---|---|
| T=30, attention pooling, count^2 (the flagship architecture) | 0.6842 | 0.0247 |
| T=100 | 0.6845 | 0.0097 |
| **T=300** | **0.7336** | 0.0231 |
| T=30, mean pooling | 0.6762 | 0.0100 |
| T=30, no count^2 pooling (`theta_pooling_weight_power: 0.0`) | 0.6682 | 0.0515 |

**Widening the topic read-out raises the floor for free**, and it is a threshold effect
rather than gradual -- T=100 is indistinguishable from T=30, T=300 is +0.05. An untrained
T=300 network beats every trained run this project has produced except the single 0.7401,
and beats `normloss_tm03`'s settled level.

Also: pooling mode and the count^2 weighting barely matter at initialisation
(0.676 / 0.668 vs 0.684), so neither is where the initialisation signal comes from --
it is the projection width.

**Consequence for every number in this document**: the bar is not a single 0.684, it is
0.684 at T=30 and **0.734 at T=300**. Any claim about a trained model has to clear the
untrained control *at its own topic count*.

### Launched

Six training runs plus two curves, all 8 GPUs:

| run | change | why |
|---|---|---|
| `normloss_tm03_t300_seed{239,1}` | T=30 -> 300 | combines both findings; T has never been swept above 60 in a trained config |
| `normloss_tm03_seed{1,2}` | replicate | the winner is n=1 against a +/-0.041 SE |
| `normloss_tm03_t100_seed239` | T=30 -> 100 | fills the T curve where the untrained control is flat |
| `normloss_tm01_seed239` | `tm_likelihood_coef` 0.3 -> 0.1 | pushes the axis that produced the result |
| curves on `normloss_tm09`, `normloss_nowarm` | -- | do the other normalised arms rise too, or only the label-heavy one |

### 2026-09-17: the T=300 gate failed, but the T=30 result replicates

**Stage-0 gate: failed.** Combining the two positive findings (label-heavy normalised
loss + the wider read-out) does not work. Both seeds of `normloss_tm03_t300` finish
*below* the untrained control at their own topic count:

| run | best on curve | untrained bar at its T | margin |
|---|---|---|---|
| `normloss_tm03_t300_seed239` | 0.7221 | 0.7336 (T=300) | **-0.012** |
| `normloss_tm03_t300_seed1` | 0.7148 | 0.7336 (T=300) | **-0.019** |

T=300 raises the floor by +0.05 for free and then training cannot reach it. This is the
only configuration tested where **training is worse than not training**, and it is worth
stating that way rather than as "T=300 underperformed": the comparison that makes it
damning is against its own random-init control, which did not exist until yesterday.

**The T=30 result does replicate.** Margin over each run's own untrained bar:

| run | best | bar | margin |
|---|---|---|---|
| `normloss_nowarm` | **0.7666** | 0.684 | **+0.083** |
| `normloss_tm03_seed239` | 0.7623 | 0.684 | +0.078 |
| `normloss_tm03_seed1` | 0.7545 | 0.684 | +0.071 |
| `normloss_tm03_t100_seed239` | 0.7531 | 0.6845 | +0.069 |
| `normloss_tm01_seed239` | 0.7309 | 0.684 | +0.047 |
| `normloss_tm03_seed2` | 0.7101 | 0.684 | +0.026 |

`normloss_tm03` across three seeds is **0.762 / 0.754 / 0.710**, every one above the bar,
mean +0.058. **This is the first result in this document that survives replication.**

Two open threads from the table. `normloss_nowarm` (no warm-start) still leads and had
not plateaued at its 60-epoch budget -- a 200-epoch rerun is training. And
`normloss_t100` is at +0.069 with 10 of 22 checkpoints scored, which would make the
topic-count story **non-monotone**: T=100 trains best while T=300 initialises best. If
that holds it is a real finding, not noise, because the two effects are measured against
different baselines.

### Correction: the loss scales should be normalised by reachable range, not chance

`normalize_loss_scales` divided each term by its value at chance. The defensible divisor
is how far the term can actually *fall*, which differs only for tm:

| term | chance | reachable floor | range |
|---|---|---|---|
| reconstruction | ln(21) = 3.0445 | ~0 (observed 0.0029, `rec_acc` 0.9991) | 3.0445 |
| label | 0.5*ln2 = 0.3466 | 0 | 0.3466 |
| **tm** | ln(S) = 10.3972 | **ln(S/R) = 9.0109** | **ln(R) = 1.3863** |

Even a perfect model only concentrates `p(s|r)` onto that repertoire's own `S/R`
sequences in the pool, so tm bottoms out at `log(S/R)`, and dividing by the chance value
**under-weights its gradient by 7.5x**. Effective weights at the `tm03` recipe:

| | rec | tm | label | label:tm |
|---|---|---|---|---|
| un-normalised flagship (tm 0.7) | 0.0475 | 0.6650 | 0.2850 | 0.4 |
| chance-normalised (what ran) | 0.0156 | 0.0274 | 1.9188 | 70.0 |
| **range-normalised** | 0.0156 | **0.2056** | 1.9188 | **9.3** |

Implemented as `CompositeLoss(loss_scale_mode="chance"|"range")`, default `"chance"` so
every existing config is bit-identical; handles `opposite_class_pair` (range `log 2`) and
`raw_bce` (chance == range). 2 new tests, 105/105 passing.

Note the *empirically* reachable tm range at full scale is far smaller than the
structural one -- best ever 0.0324 nats, unopposed. Normalising by a measured range would
multiply tm's gradient another ~46x and amplify a term that is mostly noise, which is why
the structural range is used.

**Also corrected:** an earlier scan of "best tm ever" flagged `overfit_bigenc_full` at
8.31 nats, apparently 2.09 below chance. That run uses a **4 x 1024** batch, so its chance
is `ln(4096) = 8.3178`, not `ln(32768)`. It is at chance like everything else. Any
cross-run comparison of raw `tm` has to be against that run's own pool size.

### Launched

| GPU | run | change |
|---|---|---|
| 0 | `normloss_nowarm_zmean_seed239` | `topic_input_from_mean: true` on the leader, 200 epochs |
| 1 | `normloss_tm03_zmean_seed239` | same flag on `tm03`, 120 epochs |
| 3 | `rangeloss_nowarm_depth1_seed239` | `loss_scale_mode: "range"` **and** encoder/decoder `depth: 4 -> 1` |
| 6 | `normloss_nowarm_long_seed239` | the leader, extended to 200 epochs |
| 7 | `normloss_tm03_nowarm_seed239` | the untested cross: no warm-start **and** `tm_coef: 0.3` |

`topic_input_from_mean` (implemented today) also closes a standing train/eval mismatch:
every inference path already read `z_mean` while training read the reparameterised
sample, so **every checkpoint in this project has been evaluated with a different phi
than it was fit with**. The depth-1 arm deliberately changes two things at once and is
recorded as exploratory; note `x_transformers` sets `unet_skips=(depth>1)`, so depth 1
silently disables those too.

## 2026-09-18: a new best classifier, a sequence-attribution result on the pathway this document had been reading wrong, and the constant learning rate in every config

### 1. `normloss_tm03_nowarm` -- 0.7904, the best repertoire AUC in the project

The untested cross: **no warm-start** *and* **`tm_likelihood_coef: 0.3`**. Neither
original arm had both.

| epoch | 10 | 15 | 20 | 25 | 30 | 35 | **40** | 45 | 50 | 55 | 60 | 65 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | .754 | .771 | .762 | .751 | .755 | .748 | **.7904** | .771 | .770 | .758 | .755 | .763 |

Twelve consecutive checkpoints above 0.74, peak **0.7904** at epoch 40, **margin +0.106**
over its 0.684 bar -- the widest in the project by a third, and the first result here to
clear 0.78. It decays to 0.66-0.73 after epoch 80. Fisher is 0.874, so the gap is now
0.084, down from 0.134 at the start of this work.

**What the winning checkpoint looks like inside:**

| epoch | 0 | 19 | 29 | **39** | 49 | 69 | 89 | 109 |
|---|---|---|---|---|---|---|---|---|
| `tm` | 10.3993 | 10.3954 | 10.3954 | **10.3953** | 10.3956 | 10.3957 | 10.3963 | 10.3959 |
| `rec_acc` | 0.357 | 0.363 | 0.369 | **0.468** | 0.474 | 0.584 | 0.639 | 0.662 |
| `label` | 0.3444 | 0.3265 | 0.3369 | **0.3555** | 0.3897 | 0.4529 | 0.5225 | 0.5929 |

At the AUC peak the TM is **at chance** (gain 0.0019 nats) and reconstruction is a
mediocre 0.468 -- and `rec_acc` then climbs to 0.66 *while held-out AUC decays from 0.79
to 0.70*. The best result this project has produced is achieved with both generative
components inert. That is the clearest statement yet of the outcome PLAN_v5 flagged as
possible: on Emerson this behaves as a supervised MIL classifier, and the machinery it is
named for is along for the ride.

Three replicates (`seed{1,2,3}`, 80 epochs, `checkpoint_every: 2`) were launched and have
now finished; their curves are pending (see section 7).

### 2. Four curves that closed threads, all negative

| run | best | bar | margin | verdict |
|---|---|---|---|---|
| `normloss_nowarm_long` (200ep) | 0.7583 | 0.684 | +0.074 | **worse than its own 60-epoch run (0.7666)** -- "still rising at ep58" was noise |
| `normloss_nowarm_zmean` | 0.7519 | 0.684 | +0.068 | no help |
| `normloss_tm03_zmean` | 0.7620 | 0.684 | +0.078 | vs 0.7623 without -- identical |
| `tm10x` (200ep) | 0.6756 | 0.6633 | +0.012 | at its untrained floor for the whole run |
| `rangeloss_depth1_seed1` | 0.7269 | 0.6633 | +0.064 | replicates seed239's +0.091; mean **+0.078** |
| `rangeloss_depth1_wide` | 0.7223 | 0.6659 | +0.056 | **peak is epoch 0** -- the widening failed |

Two conclusions worth keeping:

- **`topic_input_from_mean` (Stage 3) is a null on peak AUC**, twice over. It still fixes
  a real train/eval mismatch and is worth keeping on principle, but it buys nothing
  measurable. Its variance-reduction claim is untested against these curves.
- **Long budgets were the wrong call.** Extending the leader from 60 to 200 epochs made it
  worse. Subsequent runs use 80 epochs with `checkpoint_every: 2`.

`tm10x` is also a warning about reading training metrics: it had the best label loss of
anything here (0.3418, below chance at epoch 200) and the best reconstruction ever
recorded (0.95), and neither translated into held-out AUC.

### 3. The sequence-attribution comparison was being run through the wrong layer

The three PR-AUC measurements made earlier in the day (on `tm03_nowarm` ep40, `tm10x`
ep195 and `rangeloss_depth1` ep45) all used `label_head` and `topic_weights` -- both of
which read **phi**. The 166x result in this document was measured on a **different
pathway**, `attention x value` from `TopicAttentionPooling`, which the new script did not
implement. Any comparison against 166x that does not name the pathway is meaningless.

`analysis/sequence_attribution.py` (committed today; the previous version of this check
lived in an uncommitted scratch file) now scores all three readouts. The
`attention_value` reconstruction follows `TopicAttentionPooling.forward` exactly:

```
values_ST = pooling.value(h)
gated     = tanh(pooling.attend(h)) * sigmoid(pooling.gate(h))
attention = _grouped_softmax(pooling.score(gated), ..., log_weights)
score_s   = sum_{t < n_signal} attention[s,t] * values[s,t]
```

Two details that matter: the softmax is **over sequences within a repertoire**, so the
whole repertoire is normalised together (batching covers only the encoder forward), and
**clonal abundance enters the attention logits** as in training, including
`theta_pooling_weight_power`.

**Reference validation.** `warmstart_full_medium`'s `model.pt` -- the exact checkpoint
that scored 16.9x -- re-scored through this implementation:

| fraction | this implementation | the original figure in this document |
|---|---|---|
| 1e-5 | 61.7x | 166x |
| 1e-4 | 30.8x | 66x |
| 1e-3 | 20.3x | 43x |
| 1e-2 | 4.0x | 6.1x |
| 1e-1 | 1.31x | 1.4x |
| `pr_auc_over_chance` | **3.564x** | 16.9x |

The **shape reproduces exactly** -- smooth monotone decay from the extreme top to ~1.3 at
10%, the signature this document called genuine ranking signal rather than a lucky spike
-- but the magnitudes are 2-3x lower, converging at the coarse end. The likely cause is a
different ground-truth set: this scoring has **1,625 signal sequences at witness rate
1.18e-4**, where the historical check reports **603** over the same ~13.7M candidates,
probably the CDR3-only 482-clonotype list versus the V/J-aware 528. Enrichment is
normalised by witness rate, so a 2.7x difference in the denominator moves everything.

**Consequence: today's absolute PR-AUC numbers are not comparable to the historical ones,
but they are internally consistent** -- every scoring below used the identical ground
truth and the same 70 positive test repertoires.

### 4. `usage_lower1` scores 14.13x on phi -- the best sequence attribution here

All scorings on the same ground truth, `pr_auc_over_chance`:

| checkpoint | test AUC | `tm_gain` (nats) | `rec_acc` | `label_head` (phi) | `topic_weights` (phi) | `attention x value` |
|---|---|---|---|---|---|---|
| **`usage_lower1`** `model.pt` | 0.704 | **0.0130** | -- | **14.13** | **6.35** | 0.94 |
| `warmstart_full_medium` `model.pt` | 0.740 | -0.0088 | 0.444 | 0.86 | 0.90 | **3.56** |
| `rangeloss_depth1` ep45 | 0.754 | 0.0001 | **0.689** | 1.04 | 1.09 | 2.58 |
| `warmstart_full_medium` ep50 | 0.666 | -0.0088 | 0.444 | 1.00 | 1.08 | 2.01 |
| `normloss_tm03_nowarm` ep40 | **0.790** | 0.0023 | 0.468 | 0.99 | 0.89 | 1.43 |
| `warmstart_usage3_seed2` `model.pt` | 0.708 | 0.0119 | -- | 1.08 | 1.17 | 0.79 |

`usage_lower1`'s **`enrichment@1e-5` is 308x on both phi readouts: 5 true signal
clonotypes in the top 137 of ~13.7M sequences**, against 0.016 expected by chance
(Poisson p ~ 1e-9). That is the largest sequence-attribution effect recorded in this
project, ~4x the `warmstart_full_medium` reference on identical ground truth.

This **falsifies a claim made repeatedly in this document and earlier today**: that phi
has never beaten ~1.1x on any checkpoint. It was true of everything tested until now.
And on this checkpoint the *attention* pathway is the dead one (0.94x) -- exactly inverted
from `warmstart_full_medium`, where phi is dead and attention carries the signal.

Three caveats, all load-bearing:

1. **`warmstart_usage3_seed2` has near-identical TM gain (0.0119) and classifier AUC
   (0.708) and shows nothing on any pathway.** So TM gain alone does not predict
   attribution; n=2 and the two disagree. The same pattern held historically -- 166x
   reproduced on 2 of 3 seeds.
2. **`usage_lower1` is a fully collapsed run** (`H_theta` ~ 0.0001), which sits awkwardly
   with "a functioning Theta helps". A collapsed topic model produced the best attribution.
3. The Fisher list is a p-value threshold on exact clonotype identity, not ground truth.

**Withdrawn on the same day it was proposed:** "per-sequence PR-AUC tracks reconstruction
quality". It was a monotone ordering across three phi-readout points, and both the
pathway correction and the reference validation killed it -- the best sequence-ranker
(`warmstart_full_medium`, 3.56x) has one of the worst VAEs (0.444). Nothing currently
explains the ordering; every checkpoint differs in several ways at once.

What does survive: **good repertoire classification and good sequence attribution still do
not co-occur in any single checkpoint.** The best classifier (0.790) is the worst
attributor among scorable checkpoints (1.43x on its best pathway).

### 5. Every run in this project has trained at a constant 3e-4

Asked whether `rangeloss_depth1`'s loss curve indicated a step-size problem. It does not
-- it indicates overfitting:

| epoch | 0 | 12 | 24 | **36** | 48 | 60 | 84 | 102 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| val `total` | 5.234 | 2.211 | 1.722 | **1.404** | 1.514 | 1.546 | 1.548 | 1.676 | 1.827 |

Validation bottoms at epoch 36; the AUC peak was epoch 45. Train `total` goes 4.28 (ep1)
-> 1.72 (ep21) -> 1.23 (ep46) -> 1.18 (ep101) with within-epoch spread 0.02-0.03: no
oscillation, no spikes, so the LR is not too high, and the 4% move over 55 epochs says it
is not starved either. The **train-val gap widens 0.28 (ep46) -> 0.49 (ep101)**. Textbook
overfitting past a converged optimum. (The per-step figures come from the tqdm postfix, a
running epoch mean, so they rule out instability but do not measure gradient noise.)

**The LR-adjacent thing that is wrong:** `lr_decay_gamma: null` in *every* config in this
repository, ever. Given that training converges by ~epoch 46 while validation degrades
from ~36, a decay schedule is the standard response and has never been tried here. Also
never resolved: v1 used **1e-3**, 3.3x higher, and the 3e-4 in every config traces to an
early choice never validated at full scale on the current recipe.

Two arms launched on the `rangeloss_depth1` recipe, 80 epochs, one-line diffs:
`lr_decay_gamma: 0.97` (~0.09x LR by epoch 80) and `learning_rate: 1e-3`.

### 6. Free Theta + per-sequence label: the arm that makes phi do the work

With `theta_mode: "amortized"`, Theta is pooled from the same trunk that produces phi, so
the TM term can be satisfied by moving Theta instead of phi -- and the two are coupled, as
`AIRRTM_Model.forward`'s own docstring says: "the TM term ends up fighting itself".

**A free Theta breaks that.** Theta becomes 30 numbers per repertoire with no sequence
input, so the only way to fit `p(s|r) = sum_t theta_rt * exp(phi_ts)` is to make **phi**
sequence-discriminative -- the pathway that was stuck at 0.86-1.09x until `usage_lower1`.
Precedent: `budget_t60_reps597` was free-Theta and reached 0.023 nats, 10x any amortized
run, but had no label loss so phi never learned which topics were signal.

The existing guard rejects `label_input="repertoire"` + `theta_mode="free"` only, so
**`free` + `sequence` is already legal** -- and is the better version: with
`label_input: "sequence"` the label loss reads `sigmoid(phi)` and pushes on phi directly,
and the MIL smooth-max over the bag becomes live instead of an identity. Held-out
classification is not blocked because fold-in Theta scores unseen repertoires from frozen
phi.

Launched as a **pair**, because with `label_input: sequence` Theta's entire remaining job
is an auxiliary unsupervised objective on phi:

| arm | `theta_mode` | `label_input` | `tm_likelihood_coef` | what it is |
|---|---|---|---|---|
| A `freetheta_seqlabel_seed239` | `free` | `sequence` | 0.3 | Theta live, TM trains phi |
| B `freetheta_seqlabel_notm_seed239` | `free` | `sequence` | **0.0** | identical, Theta inert -- plain MIL through phi |

The intent was that arm B's Theta receives no gradient at all, so that **A - B is exactly
the price of Theta** -- see the correction below, which is that this did not hold. This cannot be done with `use_topic_model: False`, which
drops phi as well and forces `label_input: "repertoire"` -- the confound in every previous
no-TM comparison in this document (which found +0.048 / +0.044 / +0.014 across three
seeds, 3/3 directional against a +/-0.041 SE).

**Neither arm tested the premise, and both are broken in the same way: phi's scale is
unbounded.** `topic_l1_coef: 0.0` and `weight_decay: 0.0`, so nothing penalises the
magnitude of phi, and it grows close to linearly from epoch 0 in both arms:

| epoch | 0 | 10 | 20 | 30 | 40 | 50 | 53 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|---|
| A `phi_l2` | 0.20 | 1.67 | 8.89 | 20.96 | 34.63 | 50.81 | **57.16** | NaN | -- |
| B `phi_l2` | 0.35 | 0.83 | 2.14 | 4.52 | 11.06 | 25.58 | -- | 61.16 | 74.13 |

The TM term is `logsumexp_t(log theta + phi)`, so at `||phi|| ~ 57` the `exp(phi)`
overflows: **arm A died on the first step of epoch 54** (`RuntimeError: Non-finite loss`,
`train.py:493` -- `total_loss`, `tm_loss`, `label_loss`, `kl_divergence` all NaN,
`reconstruction_accuracy` 0.0438 at the failing step). Checkpoints survive to epoch 52.
Arm B survived 80 epochs only because `tm_likelihood_coef: 0` keeps that logsumexp out of
the gradient -- but its `tm_loss` still **rises from 10.41 to 26.3**, tracking `phi_l2`
exactly. A term carrying no weight drifting 16 nats *worse* than chance is the same
pathology, just harmless.

**Arm A, up to the crash:**

| epoch | 0 | 10 | 20 | 30 | 40 | 50 | 53 |
|---|---|---|---|---|---|---|---|
| `rec_acc` | 0.357 | 0.752 | 0.804 | 0.851 | 0.873 | 0.857 | 0.869 |
| `tm` | 10.401 | 10.403 | 10.401 | 10.398 | 10.398 | 10.398 | 10.399 |
| `label` | 0.3434 | 0.3432 | 0.3425 | 0.3421 | 0.3421 | 0.3422 | 0.3416 |
| `label_acc` | 0.556 | 0.556 | 0.561 | 0.576 | 0.581 | 0.586 | 0.588 |

Both arms reached `rec_acc` 0.75-0.80 by epoch 10 and arm A peaked at **0.873**, the
second-best reconstruction recorded here, where amortized runs sit at 0.357 for their
first ~20 epochs and reach ~0.5 by epoch 60. **Decoupling Theta from the shared trunk
frees the VAE immediately**, which is the one part of the design that worked as argued.

The other two terms did nothing. `tm` sat at chance (10.3972) for all 53 epochs despite
`coef: 0.3` -- **the free-Theta TM gradient inflated phi rather than making it
discriminative** -- and `label` moved 0.3434 -> 0.3416 against a chance of 0.3466, i.e.
the classifier barely trained at all under `label_input: "sequence"`. So the arm's premise
was never tested: phi did not become sequence-discriminative before the run diverged.

**Correction to the control's design.** The config comment for arm B claims "entropy
coefficients are 0 here too", and that is wrong: the file has `theta_entropy_coef: 0.5`
and `topic_usage_coef: 5.0`, so Theta receives gradient from both entropy terms
regardless of `tm_likelihood_coef`. `H_theta` falls 2.95 -> 1.99 across arm B's run,
confirming Theta moved. **A - B is therefore not the price of Theta**, and the pair would
have to be rerun with both entropy coefficients at 0 -- in addition to bounding phi -- for
that comparison to mean anything.

Arm B also overfits the label in the ordinary way: val `label` bottoms at 0.3391 (epoch
45) then climbs to 0.4308, while train reaches 0.2426.

**Before rerunning either arm: bound phi.** `topic_l1_coef` already exists and is 0 in
every config in this repository; `weight_decay` is 0 everywhere too. This is the first
configuration in the project where that has mattered, because it is the first where phi is
the only object the TM term can move.

### 7. State at the end of 2026-09-18, and what is running now

All seven runs finished training. **No AUC curve had been scored for any of them** -- and
in this project the training metrics have repeatedly failed to predict the curve, so none
of these is decided. Best validation `label` loss (chance = 0.3466):

| run | best `label` | @ep | best `label_acc` | note |
|---|---|---|---|---|
| `rangeloss_depth1_lrdecay_seed239` | **0.3245** | **78** | 0.619 | still improving at the budget |
| `normloss_tm03_nowarm_seed3` | 0.3219 | 40 | 0.652 | |
| `normloss_tm03_nowarm_seed1` | 0.3317 | 21 | 0.639 | |
| `normloss_tm03_nowarm_seed2` | 0.3358 | 24 | 0.623 | |
| `freetheta_seqlabel_notm_seed239` | 0.3391 | 45 | 0.606 | arm B |
| `rangeloss_depth1_lr1e3_seed239` | 0.3402 | 12 | 0.589 | ends at 0.702 -- degrades hard |
| `freetheta_seqlabel_seed239` | -- | -- | -- | **NaN crash at ~ep53** |

The loss shape separates them more sharply than the best value does. Train `label` at
epoch 79 against each run's val bottom:

| run | val bottom | train at ep79 | train-val gap |
|---|---|---|---|
| `rangeloss_depth1_lrdecay` | 0.3259 (ep78) | 0.3124 | **0.013** |
| `tm03_nowarm_seed3` | 0.3219 | 0.1300 | 0.306 |
| `tm03_nowarm_seed1` | 0.3317 | 0.1401 | 0.294 |
| `tm03_nowarm_seed2` | 0.3358 | 0.1220 | 0.404 |
| `rangeloss_depth1_lr1e3` | 0.3402 | 0.1362 | 0.566 |

The decay arm holds train and val 0.013 apart for 80 epochs where every other run opens a
0.29-0.57 gap: it is not fitting harder, it is the only one not memorising. Its `rec_acc`
still climbs to 0.655 and its `tm` stays pinned at 10.395 throughout, so the difference is
confined to the label term.

On the LR question the ordering at epoch 76 was **decay (0.3267) < constant (~0.43 by
ep80) < 1e-3 (0.6716)**, and `lrdecay` is the only mature run in this document whose best
epoch is its last -- consistent with the overfitting diagnosis. `lr_decay_gamma: null`
looks like a real default bug rather than a neutral choice, pending the curves.

**2026-09-19: all six completed runs launched for checkpoint-AUC curves** (40 checkpoints
each, GPUs 0-5). Nothing else is running.

### 8. Two things scoped out, and one thing that is 10 numbers wide

**Scoped out by decision:** (D) restricting the TM to a public-clonotype vocabulary, and
(E) a free per-clonotype phi table over that vocabulary. Both were proposed as the only
mechanisms with a plausible route to per-sequence identification -- phi is
`Linear(encoder(CDR3))`, a smooth function with **no per-clonotype parameter anywhere**,
where a classical topic model has a free `beta (T x V)` table and Fisher has 3.5M
independent per-clonotype tests. The consequence, recorded at the time: coefficient
rebalancing and encoder capacity have both been swept hard and neither touches that
limitation. `usage_lower1`'s 14.13x complicates that framing -- phi *can* carry
attribution in at least one run -- but does not remove it, since nothing yet explains why
that checkpoint and not its near-twin.

Also worth noting against "increase model capacity": every capacity *increase* tested has
been neutral or harmful (width 64->128 at both depths, heads 8->2, T 30->300, bigger
encoder), and the only *decrease* -- depth 4 -> 1 -- produced the best margin in the
project.

**Never swept: `vae_coef`.** Every run in this session used 0.05. It is PLAN_v5 Stage 2,
written into the plan and never executed, and it is the direct lever on `rec_acc`. Also
never built: Stage 1, gradient-norm logging -- every rebalance so far has been done on
loss-value arithmetic, which was wrong once already (chance vs range, 7.5x on tm).

**The signal/non-signal split is 10/30 and has never been swept.** Only the total was
varied (15/30/60/100/300), always at roughly 1:2. The classifier reads only the 10 signal
topics (`airrtm_model.py:215`), the MIL input is `label_features_ST[:, :n_topics_signal]`
(`:326`), both per-sequence readouts and `_compute_signal_topic_weights` use the same
slice (`:569`, `:604`, `:626`), and the new `attention x value` pathway sums over those
same 10 columns. So the entire label-relevant capacity of the model is 10 numbers per
repertoire. v1 used 4 signal + 4 non-signal. A run with, say, 25 signal + 5 non-signal at
the same total would test it directly for one GPU.

**On generation:** `generation.py` generates *sequences*, not repertoires -- it takes the
top-k by signal intensity, fits a diagonal Gaussian to their z, samples and decodes. Theta
never appears, so generation is bottlenecked on phi. Repertoire generation needs
`p(s|r) = sum_t theta_rt * p(s|t)`, and while theta_r is the only repertoire-level object
in the model, **`p(s|t)` does not exist**: phi is discriminative and there is no
`topic -> z` map. The missing piece is ~30 lines generalising `signal_latent_distribution`
from "signal" to per-topic (`mu_t, sigma_t` over the z of sequences with high `phi_t`;
then `t ~ theta_r`, `z ~ N(mu_t, sigma_t)`, decode). It needs both halves working: a
collapsed Theta generates identical repertoires, and an uninformative phi makes every
`mu_t` the same point.

## 2026-09-19: LR decay fails its own test, the 0.7904 does not fully replicate, and the arm with no topic model is the best thing on the board

Six curves, 40 checkpoints each (`checkpoint_every: 2`, epochs 0-78), all scored against
the same held-out split. Raw curves are committed at
`analysis/results/curves_2026-09-19_batch1.md`.

### 1. LR decay: the loss shape was right and the AUC was worse anyway

`rangeloss_depth1_lrdecay` (gamma 0.97) against its constant-LR parent, same seed, same
architecture, same epoch window, a one-line diff:

| epoch | 30 | 35 | 40 | **45** | 50 | 55 | 60 | 65 |
|---|---|---|---|---|---|---|---|---|
| parent, constant 3e-4 | .687 | .733 | .712 | **.754** | .725 | .734 | .739 | .712 |
| gamma 0.97 | .684 | .667 | .667 | .669 | .649 | .681 | .684 | .686 |

| run | best | @ep | bar | margin |
|---|---|---|---|---|
| parent `rangeloss_nowarm_depth1_seed239` | **0.7544** | 45 | 0.6633 | **+0.091** |
| `rangeloss_depth1_lrdecay` (gamma 0.97) | 0.6935 | 66 | 0.6633 | +0.031 |
| `rangeloss_depth1_lr1e3` | 0.7052 | 22 | 0.6633 | +0.042 |

The decay arm's curve has **exactly the shape the schedule promises** -- a late rise from
0.649 (ep50) to 0.694 (ep66) and then a flat plateau to epoch 78 with no decay tail. It
is simply 0.06 below the parent the whole way. The schedule settles the model into a
worse basin rather than holding it at a better one.

**This is the `tm10x` trap a third time.** The decay arm had the best label loss on the
board (0.3259, still falling at epoch 78) and the tightest train-val gap ever recorded
here (0.013, against 0.29-0.57 for every other run), and it cost 0.06 of held-out AUC.
`lr1e3` is the mirror image: train AUC 0.95 against test 0.64, blatant memorisation, and
its *peak* still beats the decay arm's. **Stability of the label loss is not the
objective and does not predict the curve.** Four interventions have now been judged on
training metrics and reversed by the curve.

Retest at **gamma 0.99** (0.67x LR by epoch 40, 0.45x by 80) launched on the
`normloss_tm03_nowarm` recipe, two seeds. Not because the depth-1 result is ambiguous --
it is not -- but because 0.97 may simply be too aggressive, and the gentler end of the
lever has never been seen.

### 2. The 0.7904 record is the top of a seed spread, not a level

`normloss_tm03_nowarm`, four seeds now, bar 0.684:

| seed | best | @ep | margin | curve shape |
|---|---|---|---|---|
| 239 (original) | **0.7904** | 40 | +0.106 | plateau ep10-70, then decays |
| 1 | 0.7690 | 12 | +0.085 | peaks at ep12, decays to 0.63-0.69 after ep40 |
| 2 | 0.7542 | 62 | +0.070 | dips to 0.66 mid-run, recovers late |
| 3 | 0.7411 | 46 | +0.057 | flat 0.68-0.74 throughout |

Mean 0.7637, and **no seed reproduces the 0.79**. The peak epochs are 40 / 12 / 62 / 46 --
i.e. the peak is wherever the noise happens to put it, not a property of the recipe. The
recipe is a genuine +0.08 over its bar on average, which is still the best replicated
result here, but the record itself was the top of the spread.

**A methodological problem this exposes, which applies to every number in this document.**
The "best on curve" statistic is a **maximum over 40 checkpoints** whose epoch-to-epoch
noise is ~0.04, while the untrained bar (0.6842) is a **mean over 5-7 independent draws**.
Maximum-of-40 against mean-of-7 is not a fair comparison and is biased upward. The
untrained control's own *maximum* was 0.7120. Read against that instead, the margins above
become +0.078 / +0.057 / +0.042 / +0.029 -- still positive, considerably less impressive.
Nothing in this document has been scored the fair way.

### 3. The arm with no topic model at all is the best-behaved curve on the board

`freetheta_seqlabel_notm_seed239` -- free Theta, `label_input: "sequence"`, and
**`tm_likelihood_coef: 0`**, i.e. plain supervised MIL through phi with the topic model
switched off:

| epoch | 0 | 10 | 20 | 26 | 30 | 36 | 42 | 48 | **54** | 60 | 66 | 72 | 76 | 78 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | .677 | .693 | .704 | .732 | .745 | .742 | .747 | .719 | **.771** | .695 | .664 | .705 | .758 | .764 |

**Best 0.7713 at epoch 54, ending at 0.7644 with no decay tail** -- it rises steadily from
epoch 20 and stays up. Every topic-model run on this board spikes and then decays. This
arm beats all three replicates of `normloss_tm03_nowarm` (0.769 / 0.754 / 0.741) and sits
0.02 below the unreplicated 0.7904.

Two things keep this from being a verdict:

1. **n=1**, and the seed spread on the neighbouring recipe is 0.741-0.790.
2. **This family has no untrained bar.** Its epoch-0 checkpoint scored 0.677, one draw.
   The 0.684 bar belongs to a different architecture (amortized Theta, attention pooling,
   `label_input: repertoire`), so the margin is not known.

But the direction matters: **the one arm here that removes the topic model entirely is
not worse than the recipe built around it, and its curve is better behaved.** That is the
`use_topic_model` comparison this document has wanted since the beginning, and this time
without the confound that sank the earlier one -- the previous no-TM runs had to switch
to `label_input: "repertoire"`, which drops phi from the label path altogether. Here phi
is the shared object and the only difference is the coefficient.

### 4. Launched, 2026-09-19 23:15 -- six arms on GPUs 0-5

| GPU | run | change | what it decides |
|---|---|---|---|
| 0, 1 | `freetheta_seqlabel_notm_seed{1,2}` | seed only | does the 0.7713 replicate |
| 2 | `freetheta_phibound_tm03_seed239` | `phi_l2_coef: 0.01`, both entropy coefs -> 0 | the TM arm that went NaN, with phi bounded |
| 3 | `freetheta_phibound_notm_seed239` | same | its matched control, so A - B is finally the price of Theta |
| 4, 5 | `tm03_nowarm_vae{015,03}_seed239` | `vae_coef` 0.05 -> 0.15 / 0.3 | PLAN_v5 Stage 2, the only never-swept lever |

`phi_l2 = (seq_topic_logits**2).mean()` and `phi_l2_coef` already exists in
`CompositeLoss`; it is 0 in every config in this repository. At 0.01 it costs ~0.5 of loss
at the `phi_l2 = 57` that killed the original arm -- enough to bound phi near O(10)
without dominating a label term that sits at ~0.34.

Also running: `tm03_nowarm_lrdecay_seed{239,1}` at gamma 0.99 on GPUs 6/7, curving now.

All eight arms are watched by detached chains that curve them and push the results to
`analysis/results/` without supervision (see PLAN_v6's Continuity section).
