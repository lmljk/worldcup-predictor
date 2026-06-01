# Backtest Findings (living document)

Walk-forward, look-ahead free. Test set = major-tournament matches (WC / Euro / Copa /
AFCON / Asian Cup). Metrics: mean RPS / Brier / log-loss; lower is better. Top-pick accuracy
= % where argmax matched outcome.

## Run 1 — `xi` time-decay sweep (2018-2023 majors, n=388, refit 120d)

| xi (daily decay) | DC RPS | DC top-pick | ELO RPS | ELO top-pick |
|------------------|--------|-------------|---------|--------------|
| 0.0008 | **0.19042** | 0.528 | 0.18992 | 0.575 |
| 0.0019 | 0.19198 | 0.500 | 0.18992 | 0.575 |
| 0.0030 | 0.19448 | 0.510 | 0.18992 | 0.575 |
| 0.0050 | 0.19984 | 0.505 | 0.18992 | 0.575 |

**Findings:**
1. **Lower xi is better for international football.** Club-football literature favours
   xi≈0.003; that is too aggressive here. National sides play ~10 matches/year (vs ~50 for
   clubs), so heavy decay discards signal. Default set to **xi=0.0010**.
2. **Pure ELO is a strong baseline that Dixon-Coles does not beat on 1X2 yet** (ELO RPS
   0.1899 vs best DC 0.1904; ELO top-pick 57.5% vs 52.8%). ELO replays the full match
   history (friendlies + qualifiers), giving richer strength estimates than DC's windowed fit.
3. Both sit in the competitive ~0.19 RPS band (good models / bookmakers ≈ 0.18-0.19).
4. **DC's unique value is the scoreline distribution** (over/under, BTTS, exact scores, and
   the goal-level inputs the Monte Carlo needs) — things ELO cannot produce.

**Implication for the model:** neither single model dominates → the planned **market-anchored
ensemble** is the right call. Next: blend ELO (match-result strength) + DC (scoreline) + market
de-vig, then calibrate, and re-measure against both baselines.

## Open items
- Add a **raw-market baseline** (needs historical odds → The Odds API Business, or free scrape).
  This is the real bar; if the ensemble can't beat de-vigged market, it has no edge.
- Per-factor ablation (altitude/rest/travel/injury) once context layer is wired.
- Calibration curves + isotonic fit.
