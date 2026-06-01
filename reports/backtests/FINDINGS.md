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

## Run 2 — Polymarket title market anchor (free, live)

Source: Polymarket "World Cup Winner" market (60 sub-markets, Yes-price = implied title prob),
de-vigged by normalising (overround ~3.1% — prediction markets are tight). `cli market` /
dashboard "model vs market" panel. Biggest model-vs-market divergences (model − market):

| Team | Model | Market | Edge | Read |
|------|-------|--------|------|------|
| France | 7.0% | 16.5% | **−9.5%** | rating models underrate France's squad talent the market prices in |
| Argentina | 16.2% | 8.7% | **+7.5%** | our model overrates the holders / top-ELO side |
| Colombia | 4.8% | 1.7% | +3.1% | model bullish |
| Portugal | 6.6% | 9.1% | −2.5% | market bullish |

**Read:** the France gap is the textbook reason for market-anchoring — a pure ELO/DC model
can't see roster quality, the market can. Blending pulls our estimate toward market and the
edge column flags where to investigate (injuries, draw difficulty, squad news).

## Run 3 — squad-talent prior (clubelo, fixes the France gap)

Transfermarkt market value is bot-protected (not free-scrapable). Substitute: **clubelo.com**
club Elo (free, no key) → each squad's talent = mean club Elo of its players (top-club players
≈ high market value). Captures roster quality the results-based model misses.

Squad-talent ranking (mean club Elo): England 1915, **France 1878 (#2)**, Spain 1866,
Germany 1858, Brazil 1854 — France is a top-talent squad its recent *results* don't reflect.

Applied as a modest log-space nudge to attack/defence (weight 0.10). Effect on title odds:

| Team | before | after | market | note |
|------|--------|-------|--------|------|
| France | 7.0% | **8.7%** | 16.6% | gap −9.5% → −7.9% (narrowed, not erased) |
| England | 8.7% | 11.5% | 10.8% | now ≈ market |
| Spain | 15.4% | 18.0% | 16.3% | now ≈ market |
| Argentina | 16.2% | 16.2% | 8.7% | lower talent (z+1.1), still model-overrated |

**Findings:** talent correctly lifts high-roster-quality sides (France/England/Spain). The
residual France gap is intentional — fully closing it would just be curve-fitting to the
market and defeat having an independent model. **Caveat:** this is a current-snapshot prior;
it can't be walk-forward validated without historical club-Elo snapshots, so it's a transparent
adjustment (shown as a factor), not a backtest-proven weight. Coverage ~60% of players (clubelo
is Europe-centric; Saudi/MLS/African-domestic players fall back to a baseline).

## Open items
- **Historical raw-market 1X2 baseline** for internationals isn't available free at scale
  (The Odds API soccer/historical is paid; football-data.co.uk is club leagues only). Two
  honest paths: (a) buy The Odds API Business for the historical archive, or (b) **forward
  scoring** — the `review` loop records predicted-vs-actual once matches resolve, so during the
  tournament we get an empirical model-vs-market RPS comparison for free. Use (b) by default.
- **Per-match 1X2 market anchor**: those Polymarket markets aren't up yet (~9 days out). When
  they appear, wire `fetch_polymarket` match markets → de-vig → blend into per-match predict.
- Per-factor ablation (altitude/rest/travel/injury) once context layer is wired.
- Calibration curves + isotonic fit.
