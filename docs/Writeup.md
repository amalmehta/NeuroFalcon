# Neuro Falcon: Most Day-to-Day Decoder Drift Is Firing-Rate Drift

**Question.** A decoder is trained on one week of motor-cortex recordings. How much does it lose on later days, and what's the cheapest way to win that back?

**Data.** FALCON M2 (DANDI 000953): monkey N, a 96-channel Utah array in motor cortex, decoding 2-D finger velocity from 20 ms spike counts. Training uses 7 held-in runs (19–28 Oct 2020). Testing uses 6 held-out runs from 4 later days (30 Oct; 18, 19 and 24 Nov). All loading, binning, streaming and the R² metric (variance-weighted, per day, averaged across days) come from the FALCON harness itself.

**Decoders.**
- Wiener filter: ridge regression on 10 lagged bins (200 ms).
- GRU: z-scored counts → linear 96→32 → GRU with 64 units → linear 64→2, trained on random 2 s windows.
- Both were sized on a within-run 80/20 split of held-in data. Held-out days were never used to choose a setting.

**Zero-shot result.** On the held-in validation split, the GRU roughly doubles the Wiener filter (R² 0.56 against 0.30). On held-out days with no adaptation, the Wiener filter collapses to 0.01, while the GRU keeps 0.16 ± 0.02 (5 seeds). The drop isn't uniform. Two days stay decodable (30 Oct: 0.44, 18 Nov: 0.36, on held-back halves) and two fail (19 Nov: −0.05, 24 Nov: 0.00).

**Recalibration protocol.** Each held-out run is cut in half. The decoder may adapt on the first B seconds of the first halves (B = 2.5, 5, 10, 20 s; each day only has 50–106 s in total). It's scored on the second halves, which it never sees during adaptation. The ablation varies what adaptation may change.

| Method (5 seeds, mean of 4 days) | 2.5 s | 5 s | 10 s | 20 s |
|---|---|---|---|---|
| No adaptation | 0.18 | 0.18 | 0.18 | 0.18 |
| **Renorm: re-z-score inputs with the new day's spikes (no labels)** | **0.28** | **0.27** | **0.29** | **0.31** |
| Renorm, then fine-tune the readout | 0.22 | 0.20 | 0.29 | 0.31 |
| Fine-tune readout only | 0.14 | 0.20 | 0.23 | 0.26 |
| Fine-tune input layer only | 0.14 | 0.22 | 0.23 | 0.25 |
| Fine-tune everything | 0.10 | 0.22 | 0.27 | 0.23 |
| New GRU trained only on the budget | −0.21 | −0.10 | −0.04 | 0.05 |
| New Wiener filter on the budget (ridge chosen in-budget) | −0.13 | −0.02 | −0.06 | 0.13 |

Seed-to-seed std is 0.01–0.04 for the GRU rows ([full table](../results/Ablation.md), [curve](../results/Recalibration%20Curve.png)).

**Findings.**
1. **The failure is mostly a shift in per-channel firing statistics.** Re-z-scoring with 2.5 s of unlabelled spikes lifts the two collapsed days from −0.05 to 0.21 and from 0.00 to 0.12, and leaves the healthy days unchanged (0.44 → 0.45, 0.36 → 0.35). Neither the readout nor the dynamics need to change for that.
2. **Labels don't help at these budgets.** No labelled method beats renorm at 2.5–20 s. Fine-tuning on 2.5 s does worse than no adaptation at all (0.10–0.14 against 0.18), which is overfitting to a couple of trials. Renorm + readout catches up with renorm only at 10–20 s.
3. **Pretraining is worth more than 20 s of new labels.** A GRU or Wiener filter trained only on the new day's 20 s reaches 0.05–0.13, far below any adapted pretrained GRU.
4. **Late days stay hard.** Even the best method reaches only 0.21 and 0.15 on 19 and 24 Nov, against 0.42–0.45 on the earlier days. Something beyond firing statistics changes over three to four weeks.

**Limits.**
- Only 4 held-out days, and seeds vary only the training, not the data.
- The local held-out scores use calibration runs, not FALCON's hidden eval split.
- Fine-tuning used one fixed setting (300 steps at learning rate 1e-3) for every budget. Tuning it per budget might help the labelled methods, but tuning on held-out days would leak.
- Minival files are about 3 s each, so the minival R² (GRU 0.33 ± 0.05) is noisy.
- The EvalAI image is written and its runfile checked locally, but it isn't built or submitted yet.

**Next.** Submit the renorm decoder to the leaderboard. Then test whether per-channel statistics alone explain the late-day gap: plot each channel's mean-rate change against the per-day R² loss, and try a continual renorm that updates the running statistics during the session.
