# Next Steps — CS5344 Tabular Data Augmentation for Anomaly Detection

Status as of the first Kaggle submission (partial public leaderboard: ~36% of NSL-KDD test
data, ~39% of UNSW-NB15 test data — final standings may shift):

| Dataset | Ours | Starter kit | Gap | Local proxy `S` (ours) | Local proxy `S` (baseline) | Local gap |
|---|---|---|---|---|---|---|
| NSL-KDD | 0.92664 | 0.93155 | starter +0.00491 | 0.9507 | 0.9521 | starter +0.0014 |
| UNSW-NB15 | 0.41918 | 0.43384 | starter +0.01466 | 0.4744 | 0.4606 | **copula +0.0138 (reversed on real LB)** |

**Headline finding**: NSL-KDD is directionally consistent between local proxy and real leaderboard.
UNSW-NB15 is not — our conditional Gaussian-copula generator looked clearly better locally but
lost on the real (partially novel-attack-inclusive) leaderboard. This reversal, not just the
starter kit's narrow win on NSL-KDD, is what the steps below are prioritized around.

Leading hypotheses for the UNSW reversal (not yet disambiguated):
1. **Boundary too tight.** Our copula strictly bounds every value to the *observed, per-protocol-group*
   training range (no extrapolation). The starter kit's bootstrap+jitter uses a *global* (not
   per-protocol) noise scale, which is a known flaw (it can push protocol-specific fields like
   `stcpb` off their true value) but incidentally adds more slack to the normal region everywhere.
   If UNSW's held-out set leans more on genuinely novel attack shapes than NSL-KDD's does, the
   looser boundary may simply generalize better.
2. **Preprocessing/encoding mismatch.** Our local proxy harness (one-hot categoricals + standard-scaled
   numerics) is a guess at the organizers' fixed pipeline, not a known match. Some of the local-to-real
   divergence could be measurement mismatch rather than a real generalization gap.

---

## Ordered by expected impact

### 1. Fix the `tcprtt` inflation bug + audit all derived/recomputed columns
- **Status**: identified, not yet fixed.
- **What**: `tcprtt = synack + ackdat` is recomputed post-generation to satisfy the exact identity
  the EDA verified, but `synack`/`ackdat` are each bounded to their own historical range while their
  *sum* isn't bounded to the historical `tcprtt` range. Confirmed empirically: 4/70,000 UNSW rows
  exceed the real training max (5.04 vs. 3.18).
- **Why first**: cheap, mechanical, closes an easy-to-spot correctness gap regardless of which
  hypothesis above turns out to be right.

### 2. Build a pseudo-novelty stress test (unsupervised clustering of abnormal rows → per-cluster AUPRC)
- **Status**: proposed, not built.
- **What**: `attack_cat`/`class` labels are not released, so we can't measure generalization to
  unseen attack types directly. Cluster the labeled abnormal training rows (k-means/DBSCAN) and
  report AUPRC separately for majority vs. smallest/most-atypical clusters, using our own harness
  consistently across candidates (self-consistent, not confounded by guessing the organizers'
  encoding).
- **Why this rank**: this is the only way to test hypothesis 1 without spending another rate-limited
  Kaggle submission, and it validates or kills the direction of steps 3–4 before more effort goes in.

### 3. Loosen the copula's per-protocol-group conditioning
- **Status**: proposed, not built.
- **What**: quick, cheap experiment — raise `min_group_size` so more protocol groups fall back to
  global (wider) marginals/correlation, and/or blend in some global std as extra slack on top of the
  per-group empirical interpolation.
- **Why this rank**: directly tests hypothesis 1 with a minimal code change; if it closes the gap in
  both the local proxy and the step-2 stress test, it's likely also worth a real Kaggle submission.

### 4. Principled boundary-widening (controlled extrapolation beyond the observed range)
- **Status**: proposed, not built.
- **What**: fit a smooth tail (e.g. lognormal/Gamma) beyond the empirical max for unbounded
  continuous columns instead of a hard clamp, or explicitly inflate the copula's implicit variance.
  A deeper, more principled version of step 3.
- **Why this rank**: higher potential payoff than step 3 but more work and harder to explain cleanly
  in the report — pursue after step 3's cheaper experiment confirms the direction is right.

### 5. Keep submitting to Kaggle to calibrate local proxy vs. real leaderboard
- **Status**: one submission done per dataset.
- **What**: submissions are rate-limited per day, so don't wait for every improvement — submit the
  step-1 bug fix and the step-3 loosened variant as soon as each is ready, to get real signal on
  whether the local proxy/stress-test conclusions actually transfer.
- **Why this rank**: necessary reality check, but each submission should be spent on a specific,
  falsifiable question rather than every minor tweak.

### 6. Ablate hand-picked generator hyperparameters
- **Status**: not started.
- **What**: `contamination=0.02` (IsolationForest scrub), `rare_category_min_freq=0.01`,
  `min_group_size=50`, and the `n/(n+100)` shrinkage constant were reasonable defaults, never tuned.
  Small grid search scored against the local proxy + stress test.
- **Why this rank**: likely free but incremental AUPRC gains; do after the structural questions
  (1–4) are resolved so tuning isn't wasted on a boundary-tightness bug.

### 7. Diversity/ensemble strategies
- **Status**: not started.
- **What**: blend copula-generated rows with a fraction of scrubbed real rows, or combine multiple
  independent copula fits, to increase coverage of the submitted synthetic set.
- **Why this rank**: worth trying once steps 2–4 clarify whether coverage or fidelity is the actual
  bottleneck; also a good "method novelty" point for the report.

### 8. Generator seed-robustness check
- **Status**: not started.
- **What**: we've stress-tested the *detector's* seed sensitivity (10 IsolationForest seeds) but
  never the *generator's* own RNG seed. Run generation under several seeds, report variance in local
  proxy `S`.
- **Why this rank**: cheap, doesn't move the score itself, but directly feeds the "robustness"
  grading criterion and is good due diligence before finalizing a method.

### 9. Deep-learning generator (CTGAN/VAE) — only if 1–8 plateau
- **Status**: not planned unless needed.
- **What**: the brief allows deep learning but requires clearly explaining the generated data's
  distributional properties — complexity isn't rewarded for its own sake.
- **Why last**: expensive, harder to explain, and the current copula approach hasn't been fully
  exhausted yet.

---

## Parallel, non-technical track (not ranked — runs alongside, not after)

None of the above touches the 60% of the grade that's proposal (15%) + presentation (20%) +
report (25%). Draft these using the EDA/Augmentation notebooks' own markdown as the backbone
(the "why" is already written there) rather than as an afterthought once the model is "done":
- Keep `requirements.txt` / `.venv` reproducibility current as the pipeline changes.
- Draft the method-novelty and explainability narrative early — it's easier to write while the
  reasoning (e.g. the UNSW reversal investigation) is fresh than to reconstruct later.
