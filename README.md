# GTA VI launch-day Twitch concurrent-viewer forecast

Forecasts, for each hour h = 2..24 of the UTC release day (2026-11-19), five
quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95) of the concurrent viewers in
the "Grand Theft Auto VI" Twitch category, anchored on hour 1's print and
shaped by comparable launches.

## How it works

`main.py` (Python 3.13, standard library only) reads the case file and:

1. Takes hour 1's print `p1` from the case's `known` row.
2. Takes the case's `comparables` — past big game launches, each with its
   launch-day hourly Twitch curve — and keeps the ones that share GTA VI's
   unlock pattern: console-only (PlayStation / Xbox / Switch, no PC at
   launch) and not early-access contaminated.
3. For each hour h, computes the log-ratio to hour 1 across those launches,
   `r_h = log(viewers_h / viewers_1)`, and forecasts
   `q_tau(h) = p1 * exp(m_h + z_tau * s_h)`, where `m_h` is the median of
   `r_h`, `s_h` its standard deviation (floored at 0.30, capped at 0.60),
   and `z_tau` the standard normal quantile at each level.

Why console-only: GTA VI unlocks at midnight local time, rolling by
timezone, with no PC and no early access. Hour 1 (00:00-01:00 UTC) holds
only the regions already unlocked, while the US — the largest market —
unlocks at 05:00 UTC (East) and 08:00 UTC (West), so viewership dips in
the small hours and climbs steeply through the US day, peaking in the US
afternoon / Europe evening. Console-only launches show exactly that
shape (median peak ~2.6x hour 1 at hours 18-20); mixing in
PC-simultaneous launches flattens it and misses the surge.

Validation (leave-one-out over the kit's 37 comparables, mean pinball
loss on the log scale): this method scores 0.11 on console-only targets,
against 0.17 for the persistence baseline — about a third better on the
launches shaped like this one.

## Run

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Writes `<id>.out` per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50,
q75, q95]}, ...]}` — exactly 23 rows, hours 2..24, quantiles
non-decreasing, each above 0 and at most 1e9.

## Files

- `main.py` — the forecaster (the entrypoint the grader runs)
- `LICENSE` — MIT
- `.makefun/owner.json` — the wallet this entry belongs to
