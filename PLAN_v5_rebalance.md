# Getting reconstruction and TM back, without giving up the label gain

**Status: Stage 0 resolved 2026-09-17 — the T=300 gate FAILED, the T=30 result PASSED.**

Both seeds of `normloss_tm03_t300` finished *below* the untrained control at their own
topic count (0.7221 and 0.7148 against a 0.7336 bar). T=300 raises the floor by +0.05 for
free and training then cannot reach it — the only configuration tested where training is
worse than not training. **The T=300 line is struck from this plan.**

What did survive is the narrower result. `normloss_tm03` at T=30 across three seeds:
0.7623 / 0.7545 / 0.7101, every one above its 0.684 bar, mean margin +0.058 — the first
result in this project to replicate. `normloss_nowarm` leads outright at 0.7666 and had
not plateaued at 60 epochs. `normloss_t100` is at +0.069 with half its curve scored,
which if it holds makes the topic-count story **non-monotone**: T=100 trains best while
T=300 initialises best. T=100 replaces T=300 wherever a width is needed below.

---

## The number this plan is built on

Effective gradient weight on the shared encoder is `coefficient / chance_value`. Writing
that out for the two regimes:

| regime | rec | kl | tm | label | label:tm | label:rec | tm:rec |
|---|---|---|---|---|---|---|---|
| flagship (un-normalised, tm 0.7) | 0.0475 | 0.0025 | 0.6650 | 0.2850 | 0.4 | 6.0 | 14.0 |
| `normloss_tm03` (normalised, tm 0.3) | 0.0156 | 0.0025 | 0.0274 | **1.9188** | **70.0** | **123.0** | 1.8 |

**Correction (2026-09-17): the divisor should be reachable range, not chance.** Only tm
differs — reconstruction and the label can reach 0, but the batch-softmax tm bottoms out
at `log(S/R)`, so its reachable range is `log(R) = log(n_repertoires_in_batch) = 1.3863`,
not `log(S) = 10.3972`. Dividing by chance **under-weights tm's gradient by 7.5×**:

| regime | rec | tm | label | label:tm |
|---|---|---|---|---|
| chance-normalised (what the arms above ran) | 0.0156 | 0.0274 | 1.9188 | 70.0 |
| **range-normalised** | 0.0156 | **0.2056** | 1.9188 | **9.3** |

Implemented as `loss_scale_mode="chance"|"range"`, default `"chance"`. The structural
range is used rather than a measured one: the *empirically* reachable tm range at full
scale is ~0.03 nats, and normalising by that would amplify a term that is mostly noise.

**Normalising did not balance the terms — it inverted the imbalance.** Dividing by chance
amplifies whichever term has the smallest chance value, and the label's is 0.347 against
reconstruction's 3.045. `normloss_tm03` works *because* the label term now outweighs
reconstruction 123:1 and the TM term 70:1.

So the honest statement of where we are: the label branch finally learned something
because everything else was switched off by accident, the same way everything else has
been switched off by accident for the whole project — just in the other direction. **No
run in this repository's history has ever used a deliberately chosen gradient ratio.**
That is the gap this plan closes.

It also sets the success criterion. "Getting reconstruction and TM back" means finding a
ratio where they train *and* the label gain survives. If no such ratio exists, that is a
real scientific result about this architecture on this data, and we report it rather than
hiding it.

---

## Standing measurement rules

Carried over from PLAN_v4, and non-negotiable for everything below:

- The bar is the **untrained control at the same topic count**: 0.6842 ± 0.0247 at T=30,
  0.6845 ± 0.0097 at T=100, **0.7336 ± 0.0231 at T=300**. Every new topic count needs its
  own control (5+ draws, no training, `eval_checkpoints.py --random_init`). Stage 0 is
  what this rule was for: T=300 looked like the best trained result until it was compared
  against its own bar, at which point it became the worst.
- Judge on the **checkpoint curve**, never on the restored "best epoch" — early stopping
  tracks `total_loss` and has selected epochs from 5 to 57 on identical budgets.
- Training-loop `label_acc` does not predict held-out AUC. It has failed to, repeatedly.
- Single-run differences under ~0.10 AUC are noise (n_test = 152, SE ≈ 0.041). Three
  seeds minimum.

---

## Stage 1 — measure the gradients, then choose the ratio

Everything downstream needs this, and nothing downstream is interpretable without it.

**Build:** log the per-term gradient norm on the shared encoder every `log_every` steps.
Backward each term separately against `model.encoder.parameters()` with `retain_graph`,
record the norms, then step on the combined loss as now. Cost is a few extra backward
passes per logged step, not per step.

**Then:** add `gradient_balance_target` to `CompositeLoss` — a dict of desired relative
encoder-gradient norms (e.g. `{label: 1.0, reconstruction: 0.3, tm: 0.1, kl: 0.05}`) with
coefficients rescaled by an EMA of the measured norms, GradNorm-style. Default off.

**Why first:** FINDINGS measured these norms exactly once (rec 1.56e-2, kl 9.67e-3, tm
4.95e-2, label 1.25e-2) and every coefficient decision since has been made blind. With
this, "bring reconstruction back" becomes a number we set rather than a coefficient we
guess.

**Gate:** the logged ratios at the `normloss_tm03` setting should reproduce the
123:70:1 table above. If they don't, the loss-value arithmetic is not predicting the
gradient reality and Stage 2's design changes.

---

## Stage 2 — reconstruction back

**The observation to fix:** `rec_acc` is 0.77 after one epoch and 0.36 by epoch 3, in
*every* run ever logged, and never returns above ~0.51. The autoencoder is destroyed in
the first three epochs and the rest of training is spent in a degraded state.

**Sweep** `vae_coef` upward at the winning recipe (0.05 → 0.15, 0.3, 0.5), three seeds at
the best value. Under normalisation these are interpretable for the first time: at
`vae_coef=0.5` reconstruction's effective weight is 0.156 against the label's 1.01, a
ratio of 6.5 rather than 123.

**Do not re-run the old VAE warm-start as-is.** FINDINGS already tried `vae_coef_start:
1.0` annealed down, got `rec_acc` to 0.98 during the pure-VAE phase, and then watched it
collapse back to 0.37 *at the anneal transition*. The lesson is not "warm-start doesn't
work", it is "the steady-state `vae_coef` was too low to hold what the warm-start built".
Re-run it only with a steady-state value Stage 2 has shown can hold `rec_acc` up.

**Gate:** `rec_acc` ≥ 0.70 sustained past epoch 20 **and** held-out AUC still clears the
untrained control at that T by ≥0.05. Both, or the arm fails.

---

## Stage 3 — feed the topic branch `z_mean`, not the sampled `z`

**Implemented** as `AIRRTM_Model(topic_input_from_mean=False)` (default off, so every
existing config is unchanged). With it on, `phi` and `Theta` read the posterior mean
`z_mean_SL`; the decoder and the KL still read the reparameterised sample, so the VAE
itself is untouched. 2 new tests, 103/103 passing.

Two reasons, one of which was not on the original list:

1. **It removes noise that is pure variance.** Theta is pooled over a bag of 8192
   sampled sequences and then read by a `Linear(n_signal -> 1)` head. Reparameterisation
   noise there does not regularise anything useful -- it just widens the estimator.
2. **It closes a standing train/eval mismatch.** Every inference path already reads
   `z_mean`: `predict_topic_logits`, `infer_repertoire_topic_proportions`, and the
   disjoint theta-context branch of `forward` (whose own docstring says "the posterior
   mean, not a sample"). Only the scored path in training reads a sample. So every
   checkpoint in this project has been *evaluated* with a different phi than it was
   *fit* with. That has been true since the amortized head was introduced, and it is
   invisible in any training-loop metric.

**Running** (2026-09-17): `normloss_nowarm_zmean_seed239` (200 epochs) and
`normloss_tm03_zmean_seed239` (120 epochs) — the flag on the two best recipes. The
T=300 companion is dropped with the rest of the T=300 line; use T=100 if a width
variant is wanted.

**Gate:** held-out AUC at least matches the flag-off arm, with lower seed-to-seed spread.
The variance reduction is the primary claim; a mean improvement would be a bonus.

### Excluded from this stage (user decision, 2026-09-17)

- **Separating the topic branch from the VAE bottleneck** -- giving phi/Theta their own
  projection off the encoder output so they stop competing with the decoder and KL for
  the same 64 dimensions. This is an architecture change and the plan stays on
  coefficients, sampling and read-out wiring. **What that commits us to:** if Stage 2
  shows reconstruction and the label gain genuinely trade off, we accept the trade-off
  and pick a point on it rather than re-architecting to escape it.
- **Restricting the TM term to public clonotypes**, and **giving TM the non-signal topics
  only.** Both dropped. Consequence worth stating plainly: these were the only two
  proposals with a *mechanism* for making TM useful rather than merely fittable. Without
  them the only remaining lever on TM is its coefficient -- and we have measured that
  raising it crowds the label out (the un-normalised flagship ran tm at 14x
  reconstruction and 2.5x the label, and never moved `tm` by more than 0.000 nats over
  60 epochs). **So this plan no longer contains a route to a working TM branch.** If TM
  is wanted later, 3a/3b are where to restart.

## Stage 4 — the thing TM was for

 TM's payoff was never repertoire AUC, it was **sequence
attribution**: ranking the actual CMV-associated CDR3s. Every checkpoint tested so far is
at chance by the metric that applies at this witness rate (`pr_auc_over_chance`
0.94–1.11× on both readouts, precision@100 exactly 0).

Score the best available checkpoints with `signal_enrichment` against the Fisher-selected list
on the 70 positive test repertoires. This is also the axis where the discarded `dedupfix`
arm was unexpectedly strong (`quantile_features` 0.600, and `poolfix` 0.627 — the highest
this project has recorded), so uniform-over-distinct sampling is worth re-testing here
specifically even though it lost badly on repertoire AUC.

---

## The outcome that would make this plan stop

If Stage 2 shows that reconstruction and TM can only be revived by surrendering the
label gain, the correct conclusion is that **AIRRTM on Emerson reduces to a supervised
MIL classifier over a wide random projection of repertoire composition**, and the
generative machinery it is named for does not earn its place on this dataset. That is a
publishable negative with three independent supports already in hand — the untrained
control, the label-free fold-in control, and the loss audit showing no term but the label
ever moves. It should be written up as the result, not treated as a failure to avoid.

---

## Order and cost

| stage | GPUs | wall clock | blocking? |
|---|---|---|---|
| 0 — confirm across seeds | **done** | — | T=300 failed; T=30 passed |
| 1 — gradient-norm logging + balance mode | 1 | ~2 h build, then free | yes for 2 |
| 2 — `vae_coef` sweep, 4 values × 3 seeds | 6–8 | ~1 day | no |
| 3 — `topic_input_from_mean`, 2 recipes | 2 | **running** | no |
| 4 — sequence attribution | eval only | hours | no |
