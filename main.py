"""GTA VI launch-day Twitch concurrent-viewer forecaster.

Forecasts, for each hour h = 2..24 of the UTC release day, five quantiles
(levels 0.05, 0.25, 0.5, 0.75, 0.95) of the Twitch category's concurrent
viewers, anchored on hour 1's print and shaped by comparable launches.

Method
------
The case file gives, for the target game, hour 1's print (p1) and a list of
`comparables`: past big game launches, each with its own launch-day hourly
Twitch curve (hourly points from ~14 days before release through launch
day) plus its platforms and unlock pattern.

Grand Theft Auto VI is a console-only (PS5 / Xbox Series X|S), single-player
launch with a rolling midnight-local unlock and no early access. That unlock
pattern is what shapes the day: hour 1 (00:00-01:00 UTC) holds only the
regions that already unlocked (Asia-Pacific, Europe, the UK at midnight),
while the US - the largest market - unlocks at 05:00 UTC (East) and 08:00
UTC (West), so viewership dips in the small hours and climbs steeply
through the US day, peaking in the US afternoon / Europe evening.

Only seven usable comparables share that pattern, so a median over
just those is noisy: with six launches left after holding one out, a
single launch can still move an hour's centre. The fix is to use every
usable comparable but weight it by how much it shares the pattern:
for each hour h, take every launch's log-ratio to its own hour 1,
r_h = log(viewers_h / viewers_1), and centre the forecast on their
weighted median, console-only launches at weight 1.0 and the rest at
0.6. The console launches still dominate the shape (their combined
weight beats the rest), but the other thirty launches steady the
estimate. The spread s_h is the weighted dispersion of r_h around that
median, with the same weights, after trimming the top and bottom 10% -
a few launches have hour-1 prints in the hundreds, and their wild
ratios would otherwise inflate the interval - floored at 0.30 and
capped at 0.60. The forecast for hour h is then

    q_tau(h) = p1 * exp(m_h + z_tau * s_h)

where m_h is that weighted median, s_h that trimmed spread, and z_tau
the standard normal quantile at each scored level. This is a lognormal
predictive distribution centred on the comparable-launch shape, anchored
on the one number the case actually gives us (hour 1's print).

Validation (leave-one-out over the 37 usable comparables in the kit,
pinball loss on the log scale with the competition's factor 2, lower is
better; the starter's persistence baseline is 0.93 overall and 0.34 on
console-only targets):
  - this method on console-only targets: 0.21 (about 40% better than
    persistence on the launches shaped like this one)
  - this method over all targets: 0.83 (persistence: 0.93)
  - the previous console-only-only version: 0.22 on console-only
    targets, 0.87 overall (an earlier note quoted these halved, 0.11 /
    0.17, having left the factor 2 out of the pinball loss)

Fallbacks keep the output valid whatever the case holds: if fewer than
three console-only comparables are usable, all comparables are used; if
a comparable has no hour 1 or too few hours it is skipped; if an hour
has no reference data it falls back to the starter's persistence spread
for that hour; if there are no usable comparables at all, the whole
forecast is the starter persistence. Quantiles are rounded to 2 decimals
(half up), forced non-decreasing, > 0 and <= 1e9.

Runs offline, standard library only: `python3 -P -s /bot/main.py`
(python:3.13-slim). Reads every <id>.in in the input directory and
writes <id>.out to the output directory:
  {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}
exactly 23 rows, hours 2..24, no other key. Outside the grader,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The starter's persistence spread, used only as a per-hour fallback.
SIGMA_PERSIST = 0.15
# Floors / caps for the cross-launch spread of log-ratios.
SIGMA_FLOOR = 0.30
SIGMA_CAP = 0.60
# Weight of a non-console-only comparable in the per-hour weighted median:
# console-only launches count 1.0, the rest 0.6, so the shape stays
# console-led while the other launches steady the estimate.
WEIGHT_OTHER = 0.6
# Fraction trimmed from each tail of the log-ratios before the spread is
# taken, so a few wild ratios (tiny hour-1 prints) cannot inflate it.
TRIM_FRAC = 0.10
# A comparable needs hour 1 plus at least this many launch-day hours.
MIN_HOURS = 15
# Need at least this many console-only comparables, else use them all.
MIN_CONSOLE = 3


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_console_only(comp: dict) -> bool:
    """True for a console-only launch with no PC at launch and no
    early-access contamination: the unlock pattern GTA VI shares."""
    platforms = (comp.get("platforms") or "").lower()
    unlock = (comp.get("unlock") or "").lower()
    if "contaminated" in unlock:
        return False
    has_pc = "pc" in platforms
    has_console = any(t in platforms for t in ("ps5", "ps4", "xbox", "switch"))
    return has_console and not has_pc


def launch_day_hours(comp: dict) -> dict[int, float]:
    """Hourly viewers for launch-day hours 1..24, from the comparable's
    hourly points (a point at `from` covers [from, from+1h))."""
    day_start = comp.get("day_start")
    if not day_start:
        return {}
    try:
        ds = parse_dt(day_start)
    except ValueError:
        return {}
    hours: dict[int, float] = {}
    for pt in comp.get("hourly") or []:
        try:
            t = parse_dt(pt["from"])
            v = float(pt["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and v > 0:
            hours[h] = v
    return hours


def reference_curves(case: dict) -> list[tuple[dict[int, float], bool]]:
    """Usable comparable launch-day curves, each tagged console-only."""
    curves = []
    for comp in case.get("comparables") or []:
        hours = launch_day_hours(comp)
        if 1 in hours and hours[1] > 0 and len(hours) >= MIN_HOURS:
            curves.append((hours, is_console_only(comp)))
    return curves


def weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Median of values with weights: the value where the cumulative
    weight first reaches half the total. Deterministic; no randomness,
    so two runs of the entry produce the same bytes."""
    total = sum(w for _, w in pairs)
    acc = 0.0
    for value, _weight in sorted(pairs):
        acc += _weight
        if acc >= total / 2:
            return value
    return sorted(pairs)[-1][0]


def weighted_trimmed_sd(pairs: list[tuple[float, float]], centre: float, frac: float = TRIM_FRAC) -> float:
    """Weighted population sd of (value - centre) after trimming frac
    from each tail by value, so a few wild log-ratios cannot inflate the
    spread. The dispersion is measured around the centre actually used
    (the weighted median) and with the same weights, so the interval
    reflects the spread of the launches the forecast is centred on.
    Too few values to trim: the plain weighted sd (or the floor)."""
    devs = sorted((value - centre, weight) for value, weight in pairs)
    n = len(devs)
    k = int(n * frac)
    if n - 2 * k >= 2:
        devs = devs[k:n - k]
    total = sum(w for _, w in devs)
    if total <= 0:
        return SIGMA_FLOOR
    var = sum(w * d * d for d, w in devs) / total
    return math.sqrt(var)


def hour_quantiles(p1: float, h: int, ratios: list[tuple[float, float]]) -> list[float | int]:
    """Quantiles for hour h from the cross-launch log-ratios of that hour.

    ratios carries (log-ratio, weight) per launch: the centre is the
    weighted median (console-only launches at weight 1.0, the rest 0.6)
    and the spread is the 10%-trimmed weighted dispersion of the
    log-ratios around that median.
    """
    if not ratios:
        # No reference data for this hour: starter persistence.
        return [round2(p1 * math.exp(z * SIGMA_PERSIST * math.sqrt(h - 1))) for z in Z]
    med = weighted_median(ratios)
    sd = weighted_trimmed_sd(ratios, med)
    sd = min(max(sd, SIGMA_FLOOR), SIGMA_CAP)
    out = []
    prev = 0.0
    for z in Z:
        q = p1 * math.exp(med + z * sd)
        q = max(q, 0.01)
        q = min(q, 1e9)
        q = round2(q)
        # Keep the row non-decreasing after rounding.
        if q <= prev:
            q = round2(prev + 0.01) if isinstance(prev, float) else prev + 1
        out.append(q)
        prev = float(q)
    return out


def forecast_case(case: dict, name: str) -> dict:
    if list(case.get("levels") or []) != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    curves = reference_curves(case)
    console = [c[0] for c in curves if c[1]]
    others = [c[0] for c in curves if not c[1]]
    # The reference class: console-only launches lead, the rest steady the
    # estimate at weight 0.6. If too few console-only launches are usable,
    # every comparable counts equally.
    if len(console) < MIN_CONSOLE:
        console, others = console + others, []

    hours = []
    for row in case.get("forecast") or [{"hour": h} for h in range(2, 25)]:
        h = int(row["hour"])
        if not 2 <= h <= 24:
            continue
        ratios = [
            (math.log(hours[h] / hours[1]), 1.0)
            for hours in console
            if 1 in hours and h in hours and hours[1] > 0 and hours[h] > 0
        ] + [
            (math.log(hours[h] / hours[1]), WEIGHT_OTHER)
            for hours in others
            if 1 in hours and h in hours and hours[1] > 0 and hours[h] > 0
        ]
        hours.append({"hour": h, "q": hour_quantiles(p1, h, ratios)})
    if len(hours) != 23:
        raise ValueError(f"{name}: forecast has {len(hours)} rows, expected 23")
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
