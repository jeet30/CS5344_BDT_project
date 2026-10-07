# Experiment 3: Integer Clamping & Contrastive Likelihood-Ratio (CLR) Purification

**Date:** 5 October 2026  
**Authors:** Aayush Majumdar (A0352046L) & Vadiraj Rao (A0353463A) — Group 42  

## 1. Objectives & Motivation
Following Experiment 2 where the hybrid approach scored `0.92492` on NSL-KDD and `0.51789` on UNSW-NB15:
1. **NSL-KDD Goal:** Recover the loss caused by micro-float corruption on discrete administrative flags (`num_compromised`, `su_attempted`, `root_shell`, `num_root`, `num_file_creations`) by enforcing strict integer clamping and zero-dominance.
2. **UNSW-NB15 Goal:** Purify normal traffic by weighting records via Contrastive Likelihood-Ratio (CLR):
   $$\mathcal{L}(x) = \sum_{j \in \text{sentinels}} \log \frac{P(x_j | y=0)}{P(x_j | y=1) + \epsilon}$$
   aiming to remove normal records that overlap with attack modes.

---

## 2. Experimental Results (Official Kaggle Leaderboard)
- **NSL-KDD:** **`0.92885`** (+0.00393 recovery from Expt 2's `0.92492`, near Vadiraj baseline `0.92993`).
- **UNSW-NB15:** **`0.49966`** (-0.01823 regression from Expt 2's `0.51789`, and down from Phase 7 `0.53307`).

---

## 3. Post-Mortem & Diagnostic Findings
1. **NSL-KDD Success:**
   - Enforcing exact integer types and zero-clamping on administrative flags succeeded in restoring discrete boundaries.
   - ECOD's empirical cumulative distribution on zero-valued counters correctly matched the test set distribution.
2. **UNSW-NB15 Regression ("The WAN Wipeout"):**
   - CLR purification heavily penalized WAN traffic (`sttl=254`, `dttl=252`) because $88\%$ of attack records in the training set are in the WAN mode.
   - As a result, synthetic WAN representation fell from the true normal level of **$27.0\%$** down to **$0.007\%$**.
   - When downstream ECOD and Isolation Forest were trained on this synthetic set, they treated `sttl=254` as an unseen extremity.
   - Consequently, all legitimate normal WAN flows in the private test set were falsely flagged as anomalies, collapsing AUPRC to `0.49966`.

---

## 4. Key Takeaways
- Pure contrastive likelihood filtering without mode preservation causes **mode extinction**.
- Any valid generator on UNSW-NB15 must preserve both Campus LAN ($73\%$) and WAN ($27\%$) normal modes.
- Proceed to Experiment 4 with full mode restoration and literature-grounded deep generative/utility methods.

