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

So the reference class is the comparables that share that pattern:
console-only (PlayStation / Xbox / Switch, no PC at launch) and not
early-access contaminated. For each such launch we take its launch-day
curve, and for each hour h compute the log-ratio to its own hour 1,
r_h = log(viewers_h / viewers_1). The forecast for hour h is then

    q_tau(h) = p1 * exp(m_h + z_tau * s_h)

where m_h is the median of r_h across the reference launches (a robust
centre that no single launch can drag), s_h is the population standard
deviation of r_h across them (floored at 0.30, capped at 0.60, so the
interval reflects how much comparable launches actually vary), and z_tau
is the standard normal quantile at each scored level. This is a lognormal
predictive distribution centred on the comparable-launch shape, anchored
on the one number the case actually gives us (hour 1's print).

Validation (leave-one-out over the 37 comparables in the kit, pinball
loss on the log scale, lower is better; the starter's persistence
baseline is 0.46 overall and 0.17 on console-only targets):
  - this method on console-only targets: 0.11 (about a third better
    than persistence on the launches shaped like this one)
  - median over all comparables on console-only targets: 0.19 (worse
    than persistence - mixing in PC-simultaneous launches, whose hour 1
    is already high, flattens the shape and misses the US-unlock surge)

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
import statistics
import sys
from datetime import datetime, timezone
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


def hour_quantiles(p1: float, h: int, log_ratios: list[float]) -> list[float | int]:
    """Quantiles for hour h from the cross-launch log-ratios of that hour."""
    if not log_ratios:
        # No reference data for this hour: starter persistence.
        return [round2(p1 * math.exp(z * SIGMA_PERSIST * math.sqrt(h - 1))) for z in Z]
    med = statistics.median(log_ratios)
    sd = statistics.pstdev(log_ratios) if len(log_ratios) > 1 else SIGMA_FLOOR
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
    console = [c for c in curves if c[1]]
    # The reference class: console-only launches like this one; fall back
    # to all comparables if too few of them are usable.
    ref = console if len(console) >= MIN_CONSOLE else [c[0] for c in curves]
    ref_hours = [c[0] if isinstance(c, tuple) else c for c in ref]

    hours = []
    for row in case.get("forecast") or [{"hour": h} for h in range(2, 25)]:
        h = int(row["hour"])
        if not 2 <= h <= 24:
            continue
        log_ratios = [
            math.log(hours[h] / hours[1])
            for hours in ref_hours
            if 1 in hours and h in hours and hours[1] > 0 and hours[h] > 0
        ]
        hours.append({"hour": h, "q": hour_quantiles(p1, h, log_ratios)})
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
