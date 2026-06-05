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

## Run 4 — situational context layer (free: altitude / rest / travel)

Deterministic nudges to match λ from the schedule + a static 16-venue table (altitude,
coords). No paid data. Modest, transparent priors (not walk-forward validated — WC sample
too small). Effect: only the 3 Mexico City (2240m) games trigger notable notes — Mexico is
acclimatised (no penalty), opponents −6% goals, giving Mexico a real home-altitude edge.
Rest-day differential rarely fires in groups (FIFA spaces rest evenly); travel penalties are
small. Weather/heat deliberately deferred — Open-Meteo only forecasts ~16 days out, so it's
wired for the in-tournament review loop, not pre-tournament.

Applied to per-match prediction and the group-stage Monte Carlo; knockout matches stay neutral
(venues TBD until the bracket resolves).

## Run 5 — player recent form + penalty takers (free, goalscorers dataset)

martj42 `goalscorers.csv` (47.6k goals with scorer + penalty flag, free) → per player:
recent international goals (last 2y) and penalty-taker flag (≥3 career penalties). Folded
into goal share as `career_rate × (1 + 0.06·recent_goals, capped 1.5) × (1.15 if PK taker)`.
Effect on the Golden Boot: Kane 6.5%→**13.1%** (14 recent goals + PK taker + England's focal
point), and recent form correctly elevates Lautaro near Messi. Sharpens the player layer from
"career rate only" to "career + current form + set-piece duty". Names matched by normalised
string within national team.

## Run 6 — official knockout bracket (free, pure engineering)

Replaced the random per-sim bracket with the **official 2026 R32 slot map** (group winners vs
third-placed; runners-up vs runners-up; same-group separation until QF+). Official A–L groups
from the final draw; the 8 best thirds are assigned to their 8 eligible slots by a
backtracking matcher (memoised by qualifying-set → ~instant; 30k sims in ~0.4s). Now title
odds are **draw-aware** (a team's path depends on its real group/position, not an average
field): France 8.7%→9.7%, England→12.6%. Bracket tree uses the published adjacent-pair order.
Approximation vs FIFA Annex C: the *exact* third-to-slot row isn't replicated, but eligibility
+ same-group separation are enforced (legal, realistic paths).

## Run 7 — player age: TESTED, NOT adopted (negative result)

Question: does player age improve goal prediction (e.g. down-weight older players like Neymar)?
Data: scraped DOB/age for 1247 squad players; joined to goalscorers history.

1. **Naive scoring-by-age curve is survivorship-biased** — the 33+ bucket shows a *higher*
   rate (3.10 g/yr) than mid-20s (1.89), because the only 33+ players in a 2026 squad are
   elite survivors (Ronaldo, Messi, Modrić), not average decliners.
2. **Predictive test (no leakage):** predict a player's goals in the next 2 years from prior
   3-year goals, with vs without age. R² = 0.358 (form only) → 0.360 (form + age + age²),
   **Δ = +0.002 — negligible**. Among in-form players (≥5 prior goals), the ≥32 group actually
   scored *more* in the window (5.19) than ≤28 (4.57).

**Decision: do NOT fit an age multiplier.** Recent form already encodes decline — a player who
has slowed (e.g. Neymar) shows up as low recent-goals, so an explicit age penalty is redundant
and would wrongly punish productive veterans. Age is kept as **displayed info only** (player age
+ squad average age), not a prediction factor. (Same discipline as the time-decay / context
findings: a factor must beat baseline to be adopted.)

## Run 8 — cold-form down-weight: TESTED and adopted (the right lever vs age)

Follow-up to Run 7 (age rejected). Question: should a once-strong scorer who has gone cold
be down-weighted? Test (cutoff 2024-06, no leakage): recent-2y form predicts next-2y goals
better than older form (R² 0.303 vs 0.278). Decisive split among players strong 2–4y ago:
- went **cold** (≤1 recent goal): avg **3.46** future goals
- stayed **hot** (≥4 recent): avg **6.30**

→ **Adopted.** In `players._weight`, a player with career rate ≥ 0.18 (a real scorer) but
<3 recent goals is scaled to 0.55 / 0.70 / 0.85 (recent 0/1/2); hot scorers boosted as before.
Effect: **Neymar share 0.153 → 0.117** (cold), in-form **Raphinha becomes Brazil's #1**; Messi
(age 38 but recent 6) stays high. Confirms form — not age — is the lever, and it handles the
"injured vs not-scoring" ambiguity correctly because cold players score ~45% less *regardless
of cause*. The exact "is he in today's XI" gate is applied match-day via confirmed lineups
(`fd_lineup_absences` → `match_scorers(absent=...)`).

## Run 9 — match-importance + steeper-recency weighting: TESTED, NOT adopted

Two user hypotheses, backtested (2016-2023 majors, n≈500, lower RPS better):

| config | DC RPS |
|--------|--------|
| baseline (xi=0.0010, importance=0) | **0.1920** |
| + importance 0.5 / 1.0 (WC/major weighted > friendlies) | 0.1929 / 0.1932 |
| steeper recency xi=0.003 | 0.1970 |
| recent-heavy xi=0.005 + importance | 0.2018 |

**Both hurt.** Why: national teams play only ~10 matches/year, so (a) down-weighting
friendlies/qualifiers throws away real strength signal the model needs, and (b) over-weighting
the last ~3 years starves an already-small sample. Confirms Run 1 (lower xi is better for
international football). **Decision: keep importance=0, xi=0.0010.** The `importance` knob is
kept in `dc.fit` (default off) as a tested, documented option.

**On club football (UCL/leagues):** club *matches* can't enter the national-team match model
(different entities), but club *strength* already does — `clubelo` ratings are computed from
those very UCL/league matches and feed the squad-talent factor. Leaning harder on it = raising
the talent weight, but that's a current-snapshot prior (no historical club-Elo → not
walk-forward validatable), so it's left modest (0.10).

## Run 10 — EA FC25 ratings, 3-year window: TESTED, both ADOPTED

(EA Sports **FC 26** releases ~Sept 2026, *after* the World Cup → FC 25 is the latest game.)

**FC25 squad ratings (adopted).** Free dataset (16k players, OVR + 6 categories + 30 sub-attrs
+ position/league). Per team we build a projected best XI (4-3-3 by OVR) and compute attack
(forwards weighted on attacking sub-attrs) / defence (defenders+GK on defensive) / overall.
Sanity: France #1 (85.5), Brazil best defence (80.4) — sensible. Two uses:
  * **Team prior**: blended with clubelo into the attack/defence nudge, using FC's attack/
    defence *split* (attacking squads boost goals, defensive squads concede less). Pushes
    France title 9.7%→10.9% (gap −9.5%→−5.7% across talent+FC).
  * **Player awards**: FC OVR **passed a predictive test** — adding it to prior-goals lifts
    player-goal R² 0.375→0.388 (Δ+0.013, ~6× the age effect). Folded into goal share with a
    league-tier multiplier (top-5 leagues ×1.06). Mbappé (OVR 91) → 28% of France's goals.
  * Still a current-snapshot prior (no historical FC ratings → not walk-forward validatable),
    so weights kept modest.

**3-year training window (adopted).** Backtest: 8-year history RPS 0.1923 → **3-year 0.1903**
(better). Removing stale squad/manager-era data helps; all 48 teams still have ≥8 matches.
Default `train_years` 8→3. (Note: this is a hard window with gentle decay — distinct from the
*steeper-decay* test in Run 9, which hurt.)

**Rejected / infeasible:**
  * **Exclude friendlies (B-squad proxy)**: hurts (RPS 0.1923→0.1991); friendlies carry real
    signal. True B-squad exclusion needs historical lineups (martj42 has none) — not possible.
  * **Coach ability + tactical-style counter**: no free coach-rating dataset exists; tactical
    "克制" would be fabricated — deliberately not implemented.

## Run 11 — BRACKET BUG FIX (knockout tree adjacency)

Bug (user-spotted): the structured bracket paired R32 matches *sequentially* into the R16
(matches 1&2 → R16, 3&4 → R16…), which is NOT the official tree. It wrongly put Group-J winner
(Argentina) and Group-K winner (Portugal) in the *same Round of 16*.

Fix: encoded the real 2026 bracket from the official match tree (R32 = matches 73-88; R16
89-96 with the published non-sequential feeders, e.g. M89 = W74 vs W77; QF 97-100; SF; Final).
Verified: Argentina (W-J, match 86 → R16 match 95) and Portugal (W-K, match 87 → R16 match 96)
now meet only at QF match 100 — correct. Rebuilt the Monte Carlo knockout to walk this explicit
tree (`_R16_PAIRS/_QF_PAIRS/_SF_PAIRS`) instead of adjacent pairs. Sanity sums hold (title 1.0,
R32 32, QF 8, final 2). Title odds shift to reflect real draw quadrants (e.g. Argentina's weak
Group J + soft quadrant lifts its model prob; market edge still flags any overrating).

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

## Run 12 — rest-day differential: TESTED and VALIDATED (kept modest)

User asked which of {injuries, motivation, venue, style-counter, yellow-cards, rotation}
can become free, backtest-validated factors. Most were already settled (venue=Run 4 adopted;
motivation/importance=Run 9 rejected; style-counter/coach=Run 10 rejected as un-free;
injuries+rotation+suspensions = match-day only, via confirmed-lineup absences). The one
genuinely new, free, walk-forward-testable factor was **rest-day differential** — until now
only an *assumed* prior in `context.py`, never validated.

Walk-forward, look-ahead free (majors 2014-2023, n=673; each team's rest = days since their
previous played match strictly before the date; shorter-rested side's λ nudged down). 8 param
configs:

| per_day | cap | fired | baseline RPS | rest RPS | Δ |
|---------|-----|-------|--------------|----------|------|
| 0.010 | 0.06 | 123 | 0.19114 | 0.19107 | −0.00007 |
| 0.015 | 0.06 | 123 | 0.19114 | 0.19106 | −0.00008 |
| 0.025 | 0.10 | 123 | 0.19114 | 0.19103 | −0.00011 |
| 0.040 | 0.10 | 123 | 0.19114 | 0.19099 | −0.00015 |

**Findings:** rest-adjustment beats baseline in **all 8 configs**, monotonically improving
with penalty size — a consistent, real (if small) signal, the mirror image of Run 9's
importance (8/8 *worse*). But the magnitude is ~0.0001 RPS, ~10× smaller than adopted factors
(talent / FC / 3-yr window ≈ 0.001–0.002), and only 18% of major-tournament matches fire
(FIFA spaces rest evenly). **Decision: promote rest-day from "assumed prior" to
"backtest-validated factor", but keep the existing conservative magnitude (per_day=0.015,
cap=0.06).** Not cranked to the strongest config: monotonic-better on a single sample is the
overfitting trap the doctrine warns against; the conservative setting is inside the tested-good
range and never hurts.

**Six-factor verdict (final):**
| factor | free data | status |
|--------|-----------|--------|
| venue (altitude/travel) | yes | ADOPTED (Run 4) |
| rest/fatigue | yes | ADOPTED, now validated (Run 12) |
| motivation/importance | yes | REJECTED (Run 9, hurt) |
| style-counter / coach | no free rating | REJECTED (Run 10, would fabricate) |
| injuries | live only | match-day via confirmed-lineup absences |
| rotation | live only | match-day via confirmed lineups |
| yellow-cards / suspension | no historical | suspension folded into match-day absences |

Tool: `skill/backtest/ablation_rest.py` (reusable).

## Run 13 — referee factor: CONSIDERED, NOT adopted (no free historical data)

User asked whether referee tendencies are modeled. They are **not** — by design, same
discipline as age (Run 7) / importance (Run 9) / coach-tactics (Run 10).

**Data reality.** The football-data.org feed exposes a `referees` field, and the match-detail
loader (`data_loader.fetch_fd_match`) pulls the appointed referee match-day alongside lineups.
But the *training* set (martj42 results.csv) is **scores only** — no per-match referee, no
card/penalty event history. football-data populates `referees` only ~1-2 days pre-match. So
there is no historical referee+events series to fit or **walk-forward validate** against — it
fails the "a factor must beat baseline to be adopted" rule before it even starts.

**Even with the data, weak for 1X2.** Referee tendencies mostly move the *cards / penalties*
markets (a bookings sub-model), not who wins. The marginal effect on the match result is small
and noisy — isolating it would need paid event data (Opta/StatsBomb tier), and scraped
referee-bias tables are ToS-gray (same objection as Macau odds).

**Decision:** referee is **display-only context** (shown match-day on the dashboard next to
lineups), not a prediction factor. Folding it into λ would be fabricating a signal we cannot
verify. If ever pursued, it belongs in a separate cards/bookings model fed by a paid events
dataset, not the goals/result model.

## Run 14 — dead-rubber / 战意 (final-group-round motivation): TESTED, NOT adopted

Hypothesis (user, borrowed from a competing model's blurb): on matchday 3 a team already
mathematically **qualified or eliminated** has lower motivation (rests starters) and
underperforms its model win-probability. Distinct from Run 9 (which weighted match *importance*
in training) — this is a match-day status effect.

Method (look-ahead free): reconstructed 4-team groups from the match graph for clean top-2
editions (WC 1998-2022, Euro 1996-2012); computed standings from matchday 1-2 results only;
flagged a side "dead" if it was top-2-clinched or eliminated under **all** 3^2 remaining-result
combinations; refit DC `as_of` each MD3 date. n=144 MD3 matches, 142 dead sides.

| | result |
|--|--|
| model expected win-rate of dead sides | 33.5% |
| **actual** win-rate of dead sides | **38.7%** |
| penalty 0.0 / 0.08 / 0.15 / 0.25 → RPS | 0.2052 / 0.2060 / 0.2070 / 0.2091 |

**Findings: the opposite of the hypothesis.** Dead-rubber sides won *more* than the model
predicted, and a motivation penalty made RPS monotonically worse (mirror of Run 9). The
"qualified team slacks off and loses" narrative isn't extractable: teams that clinch early are
the strong teams, and that strength swamps any intensity drop — a strength-confounded
non-signal. **One real by-product:** MD3 dead-rubber matches are genuinely noisier (subset RPS
0.205 vs ~0.19 for majors overall), so these games carry more variance — but the variance is
*not* directional ("dead team loses" is false), so there is nothing to bet on. **Decision: no
motivation factor.** (Caveat: clinched and eliminated were pooled; a finer split is possible
future work, but the simple directional factor is dead either way.) Tool:
`skill/backtest/ablation_deadrubber.py`.

## Run 15 — calibration / overconfidence ("upset variance layer"): TESTED, NOT adopted

Hypothesis (#3): the ensemble is overconfident — favourites win less than predicted — so a
probability-calibration / variance layer (the "爆冷波动层" the blurb brags about) should help.
Method: 935 walk-forward major-match predictions (2010-2023); one-vs-rest reliability + ECE;
temperature scaling p ∝ p^(1/T) fit on the chronological first half, tested on the second.

| | result |
|--|--|
| ECE (10-bin, one-vs-rest) | **0.0230** (already well-calibrated) |
| reliability shape | mid-range (0.40-0.80) slightly **under**-confident; only the tiny 0.80+ bins (n=23/7) over |
| fitted T | **0.925** (T<1 = *sharpen*, not soften) |
| holdout RPS (uncal → T-scaled) | 0.18919 → 0.18884 (−0.00036, better) |
| holdout log-loss | 0.96513 → 0.96575 (+0.00062, worse) |

**Findings:** the model is **not** systematically overconfident on 1X2 — if anything mildly
*under*-confident mid-range, so the optimal temperature *sharpens* (T=0.925). The holdout effect
is marginal and metric-conflicting (RPS slightly better, log-loss slightly worse) → no clean
baseline beat. Crucially, the blurb's instinct (soften favourites for upsets, T>1) would push
calibration the **wrong** way and hurt. **Decision: no calibration layer.** The residual
Argentina-vs-market gap (Run 2, +7.5%) is therefore a *strength-estimate* issue (ELO overrates
Argentina), not a variance/calibration one — a different lever, not fixable by softening probs.
Tool: `skill/backtest/calibration.py`.

## Run 16 — pre-tournament injury (season-ending absence): NOT backtestable (free data)

Hypothesis (#2): a key player ruled out for the whole tournament should lower that team's
talent across all its matches (not just match-day, which the lineup-absence channel already
handles). **Verdict: cannot be walk-forward validated on free data** — there is no historical,
machine-readable "player X missed tournament Y with injury" series, and no counterfactual, so
the doctrine's "must beat baseline" gate can't even be applied. At best it would be a transparent
prior (haircut a team's `squad_talent` when a star is pre-ruled-out), shown as a factor, never a
backtest-proven weight — same status as the talent/context priors.

**Update — implemented as a *mechanical* prior (not a tuned penalty).** Rather than hand-pick a
haircut, a confirmed tournament-long absentee is **removed from the squad before** squad-talent /
FC25 strength is computed (`model/injuries.py`, fed by curated `data/injuries_wc2026.json` with a
public source per entry). The projected XI rebuilds with the next-best player, so attack/defence
fall by the absentee's **marginal** contribution — the magnitude is endogenous, nothing is fitted.
Verified mechanically: injecting "Mbappé out" drops France FC25 attack_z 1.848 → 1.505 (−0.343),
the XI's real marginal loss. Surfaced on the dashboard team card as "injury prior", explicitly
labelled a prior. The file ships **empty** (no fabricated injuries); it activates only when a real
ruling is added. Distinct from the match-day lineup-absence channel (single-match goal shares).

**Net of Runs 14-16:** all three "advanced layers" a competing blurb advertises
(upset-variance, dead-rubber motivation, and — from Run 10 — crude coach-tactics) either fail
validation or can't be validated. Restraint is the finding: two of three intuitions were
falsified outright on data.

## Run 17 — defending-champion "curse" (卫冕魔咒): TESTED, NOT adopted

User question: "everyone says the holder won't repeat — Argentina won 2022, Messi has his title,
so 2026 won't be Argentina. Would you mark them down?" Tested it instead of asserting.

Look-ahead free: the defending champion of edition (tour, year) is the winner of the *previous*
edition — known before kickoff. Hardcoded holder map for WC/Euro/Copa 1994-2024 (22 editions,
100 holder matches). DC refit `as_of` each match; compared holder model-expected vs actual
win-rate; swept a penalty on the holder's λ.

| | result |
|--|--|
| holder model-expected win-rate (100 matches) | 47.5% |
| holder **actual** win-rate | **47.0%** — essentially identical, no aggregate curse |
| penalty 0.0 / 0.05 / 0.10 / 0.20 → RPS | 0.19096 / 0.19043 / 0.19032 / 0.19150 |

Per-edition fates (the vivid part of the narrative IS real, but split):
  * **Group-stage flops** (curse confirmed): WC 2002 France (0% vs 51%), WC 2010 Italy (0/47),
    WC 2014 Spain (33/52), WC 2018 Germany (33/50); Euro 2000/2008 holders also crashed.
  * **Over-performers** (curse reversed): WC 2006 Brazil (80/57), **WC 2022 France reached the
    final (71/40)**, WC 1994/1998 holders, and every recent **South-American Copa holder** —
    **Argentina 2024 won 83% vs 52% expected**, Chile 2016, Brazil 2021.

**Findings: the curse is a survivorship narrative, not a stable factor.** Aggregate holder
win-rate matches the model almost exactly (47.0% vs 47.5%); the penalty's best RPS gain is
0.0006 at 10% and *reverses* by 20% (non-monotonic = noise/overfit on n=100). Edition-level there
is only a weak tilt (≈10 under / 7 over / 4 neutral), concentrated in **European** holders and the
**2002-2018 WC window**. **Decision: no defending-champion factor** (same fate as importance /
dead-rubber / calibration). **Specifically for Argentina 2026 the precedent runs the *opposite*
way:** South-American Copa holders and the most recent WC holder (France 2022) over-performed, so
the data gives the "won't repeat" claim no support for *this* team. Argentina's model 22.5% vs
market 8.6% gap is an aging/ELO strength-estimate issue, to be pulled down by live market-anchoring
once books open — not by hand-editing on a falsified narrative. Tool:
`skill/backtest/ablation_holder.py`.

## Run 18 — title-level market anchor with fade-out (architecture, not a backtest)

Follow-up to the Argentina question (Runs 2/17): the headline title number was the *raw*
Monte Carlo, which overrates top-ELO sides the market discounts for aging (Argentina model
22.5% vs market 8.5%, the board's biggest divergence). Per-match market anchoring (MARKET_WEIGHT
0.60) only activates once per-match books open, but the **title book is already live** — it was
only being shown for comparison, not folded into the headline.

Fix (publish step): the displayed title probability is now
`title_final = w·market + (1−w)·model`, renormalised. The **raw model + edge stay untouched** in
the model-vs-market panel, so the independent signal is fully preserved (Argentina still shows
22.5% / +14.0% there). **Fade-out:** `w = W_TITLE_MAX·(1−coverage)` where coverage = fraction of
fixtures carrying a live per-match market (`market_1x2`). Pre-tournament coverage=0 → w=0.60; as
books open the Monte Carlo itself drifts to market, so the direct title anchor decays to 0,
avoiding double-anchoring. W_TITLE_MAX=0.60 (matches MARKET_WEIGHT).

Effect (coverage 0, w=0.60): Argentina 22.5% → **14.1%** (toward, not to, market 8.5%); France
7.6% → 12.4% (anchor lifts *up* where the model under-rates — works both directions); distribution
still sums to 1. Deliberately does **not** close the gap (Run 3 principle: erasing it = reproducing
the market). Dashboard headline + team cards show the blended figure with a "60% market-anchored"
note; raw model remains one panel away.

## Run 19 — weather (aggregate effect on goals): TESTED, NOT adopted

User challenge: deferring weather to the in-tournament review loop "because forecasts only reach
~16 days out" conflates *forecasting* with *validation*. For any past match we can fetch the
weather that ACTUALLY occurred (Open-Meteo ERA5 archive, free, no key, look-ahead free) and test
whether it moved the result — the backtest that was missing.

Method: geocoded host cities, joined the actual match-day weather (max temp / precip / wind) to
**1642 major-tournament matches** (1990+).

| max temp | n | avg goals | | precip | n | avg goals |
|----------|---|-----------|-|--------|---|-----------|
| 10-18°C | 265 | 2.555 | | dry | 925 | 2.484 |
| 18-24°C | 454 | 2.579 | | light | 400 | 2.487 |
| 24-28°C | 382 | 2.421 | | moderate | 199 | 2.658 |
| 28-32°C | 347 | 2.499 | | heavy | 118 | 2.492 |
| 32°C+ | 178 | 2.506 | | | | |

Pearson r (vs total goals): **tmax −0.003, precip −0.002, wind −0.001** — all ≈ 0. Hot (≥30°C)
2.490 vs mild (14-24°C) 2.573, Δ −0.083 goals (~3%, trivial). **Findings: weather has no usable
effect on tournament-match goals.** Two honest reasons it's a non-factor: (a) tournaments are
**scheduled to avoid extreme weather** (Qatar 2022 → winter; summer matches → evenings), so the
sample's weather variance is compressed by design; (b) even the residual hot-match dip is tiny and
symmetric (both sides), so it can't move 1X2. **Decision: weather stays display-only (the forecast
shown on a match card), not a prediction factor — in pre-match OR review.** Corrects the earlier
"16-day forecast" framing: the real reason is *it was tested on 1642 matches and doesn't move
results*. Altitude (Run 4) stays — different mechanism (physiological, ~6%, can't be scheduled
around). Tool: `skill/backtest/ablation_weather.py`.

## Run 20 — climate mismatch (cold-climate teams in heat): TESTED, NOT adopted

The steelman after Run 19's aggregate null: a *differential* — a cold-climate side stressed by
heat its opponent is acclimatised to (the heat analogue of altitude). Method: each national team
got a home-climate baseline = mean daily-max temp of its country (one ERA5 climatology year,
look-ahead free; e.g. Austria 10.5°C, Argentina 23.0, Algeria 30.6). For each match, heat
mismatch = match_tmax − baseline per side; tested the **427 matches** where the two sides differ
by ≥6°C, walk-forward.

| | result |
|--|--|
| more-heat-stressed side: model-expected win-rate | 32.4% |
| more-heat-stressed side: **actual** win-rate | **38.2%** |
| penalty 0 / 0.01 / 0.02 / 0.04 per °C → RPS | 0.1908 / 0.1930 / 0.1956 / 0.1970 |

**Findings: the opposite of the hypothesis.** The heat-stressed side won *more* than the model
predicted, and penalising it monotonically worsened RPS (same shape as Runs 9/14/17). The
confound: cold-climate = European = systematically stronger, so "cold team in heat" is usually the
**favourite** (a Euro power vs a weaker hot-climate host), and strength swamps any heat
disadvantage. **Decision: no climate-mismatch factor.** Weather is now comprehensively dead — both
the average effect (Run 19) and the differential (Run 20). Tool:
`skill/backtest/ablation_climate_mismatch.py`.
