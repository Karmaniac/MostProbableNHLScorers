"""Compare tims.py (v1) and tims_v2.py on past days: rank everyone who played each day as of that morning with
both models, then score the predictions against who actually scored.

Usage:
    py backtest.py START [END]          dates YYYY-MM-DD; END defaults to yesterday
    py backtest.py --season 20252026    a whole regular season (any season the NHL API has; several allowed)
    Add --csv PATH to also save every player-game's inputs and result (training data for train.py).

How to read it (lower log loss and Brier score are better):
    - Log loss / Brier: how close each player's chance was to what happened, over every player.
    - Top N: how many of each day's N highest-ranked players scored. This is what matters for Tims picks.
    - Calibration: in each band, the average predicted chance should match the share who actually scored.

Players come from each game's box score, so everyone is counted on the team he played for that night, and
scratched players are left out (both models would have had to guess about them the same way).
Regular-season games only.
"""

import argparse
import csv
import datetime as dt
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import features
import tims
import tims_v2

MODELS = {"v1": tims, "v2": tims_v2}
TOP_NS = (10, 25, 50)
BUCKETS = [(0, 0.2, "Under 20%"), (0.2, 0.3, "20-30%"), (0.3, 0.4, "30-40%"), (0.4, 1.01, "40%+")]

# Both models take a player's team from his profile's *current* team, which is wrong for past seasons.
# Each analyze() call sets the team he played for that night here, and player profiles are patched to match.
# Both models also share one cache, so each page is fetched once.
_night = threading.local()
_fetch = tims.get_json


def get_json(url, retries=3):
    data = _fetch(url, retries)
    team = getattr(_night, "team", None)
    if team and url.endswith("/landing"):
        data = {**data, "currentTeamAbbrev": team}
    return data


tims.get_json = tims_v2.get_json = get_json


def day_games(date):
    """(skaters, goals) for the date's regular-season games, or None if any aren't final yet.
    skaters = (name, team, player id) for everyone who dressed, from each game's box score;
    goals = {player id: goals that count for Tims}."""
    games = [g for g in tims.get_json(f"{tims.API}/score/{date}").get("games", []) if g.get("gameType") == 2]
    if any(g.get("gameScheduleState") == "OK" and g.get("gameState") not in ("OFF", "FINAL") for g in games):
        return None
    scorers = {}
    for g in games:
        for goal in tims.counted_goals(g):
            scorers[goal["playerId"]] = scorers.get(goal["playerId"], 0) + 1

    def box(game):
        b = tims.get_json(f"{tims.API}/gamecenter/{game['id']}/boxscore")
        return [(p["name"]["default"], b[side]["abbrev"], p["playerId"])
                for side in ("awayTeam", "homeTeam")
                for p in b["playerByGameStats"][side]["forwards"] + b["playerByGameStats"][side]["defense"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        skaters = [p for players in pool.map(box, games) for p in players]
    return skaters, scorers


# Training data: one row per player-game, every input known that morning, plus what happened.
CSV_FIELDS = [
    "date", "season", "player_id", "name", "pos", "team", "opp", "home", "team_b2b", "opp_b2b",
    "goals", "scored",
    # v1's inputs and prediction
    "v1_rank", "v1_prob", "v1_xg", "v1_shots_pg", "v1_sh_pct", "v1_opp", "gp", "missed",
    # v2's inputs (blended) and prediction
    "v2_rank", "v2_prob", "v2_xg", "v2_pp_xg", "ev_min", "pp_min", "ev_rate", "pp_rate", "ev_sh", "pp_sh",
    "opp_ev", "opp_pen", "opp_pk",
    # raw ice time before blending: this season so far and the last two weeks, per game
    "gp_season", "ev_min_season", "pp_min_season", "gp_recent", "ev_min_recent", "pp_min_recent",
]


def training_row(date, season, by_model, pid, ctx2, goals, played_yesterday):
    (r1_rank, r1), (r2_rank, r2) = by_model["v1"][pid], by_model["v2"][pid]
    return {
        "date": date, "season": season, "player_id": pid, "name": r1["name"], "team": r1["team"],
        "opp": r1["opp"].split()[-1], "goals": goals.get(pid, 0), "scored": int(pid in goals),
        "v1_rank": r1_rank, "v1_prob": round(r1["prob"], 5), "v1_xg": round(r1["xg"], 5),
        "v2_rank": r2_rank, "v2_prob": round(r2["prob"], 5), "v2_xg": round(r2["xg"], 5),
        "v2_pp_xg": round(r2["pp_xg"], 5),
        **features.inputs(r1, r2, ctx2, played_yesterday),
    }


def rank(model, ctx, skaters):
    def one(p):
        _night.team = p[1]
        try:
            return model.analyze(p[0], p[1], ctx, p[2])
        finally:
            _night.team = None
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(one, skaters))
    rows = [r for r in rows if "opp" in r and r.get("id")]  # skip lookups that failed
    return sorted(rows, key=model.sort_key, reverse=True)


def report(preds, days):
    """preds: {model: [(date, rank, prob, scored)]}, every model over the same players."""
    def line(label, values):
        print(f"  {label:<24}" + "".join(f"{v:>22}" for v in values))

    names = list(preds)
    first = preds[names[0]]
    n, hits = len(first), sum(y for *_, y in first)
    print(f"\n{days} days, {n} player-games, {hits} scored ({hits / n:.1%})\n")
    line("", names)
    eps = 1e-6
    line("Log loss (lower=better)", [f"{-sum(math.log(max(eps, p if y else 1 - p)) for _, _, p, y in preds[m]) / n:.4f}" for m in names])
    line("Brier (lower=better)", [f"{sum((p - y) ** 2 for _, _, p, y in preds[m]) / n:.4f}" for m in names])
    base = hits / n
    print(f"  {'(guess everyone ' + format(base, '.0%') + ')':<24}"
          f"{-(base * math.log(base) + (1 - base) * math.log(1 - base)):>22.4f}  log loss with no model at all")
    for top in TOP_NS:
        cells = []
        for m in names:
            picks = [y for _, r, _, y in preds[m] if r <= top]
            cells.append(f"{sum(picks)}/{len(picks)} ({sum(picks) / len(picks):.0%})" if picks else "-")
        line(f"Top {top} scored", cells)
    print("\n  Calibration: predicted -> actual (players)")
    for lo, hi, label in BUCKETS:
        cells = []
        for m in names:
            band = [(p, y) for _, _, p, y in preds[m] if lo <= p < hi]
            cells.append(f"{sum(p for p, _ in band) / len(band):.0%} -> {sum(y for _, y in band) / len(band):.0%} ({len(band)})"
                         if band else "-")
        line(label, cells)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Compare tims.py and tims_v2.py on past days.")
    ap.add_argument("start", nargs="?", help="first date, YYYY-MM-DD")
    ap.add_argument("end", nargs="?", help="last date, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--season", type=int, nargs="+", help="whole regular seasons instead, e.g. 20242025 20252026")
    ap.add_argument("--csv", metavar="PATH",
                    help="also save every player-game's inputs and result, one row each (training data for train.py)")
    args = ap.parse_args()
    yesterday = dt.date.today() - dt.timedelta(days=1)
    if args.season:
        known = {s["id"]: s for s in tims.get_json(f"{tims.API}/standings-season")["seasons"]}
        ranges = []
        for sid in args.season:
            if sid not in known:
                sys.exit(f"Season {sid} not found.")
            ranges.append((f"Season {sid}", dt.date.fromisoformat(known[sid]["standingsStart"]),
                           min(dt.date.fromisoformat(known[sid]["standingsEnd"]), yesterday)))
    elif args.start:
        ranges = [(f"{args.start} to {args.end or yesterday}", dt.date.fromisoformat(args.start),
                   dt.date.fromisoformat(args.end) if args.end else yesterday)]
    else:
        sys.exit(__doc__)

    out = open(args.csv, "w", newline="", encoding="utf-8") if args.csv else None
    writer = csv.DictWriter(out, CSV_FIELDS) if out else None
    if writer:
        writer.writeheader()
    results = []
    for label, start, end in ranges:
        preds, days, d = {m: [] for m in MODELS}, 0, start
        while d <= end:
            date, d = d.isoformat(), d + dt.timedelta(days=1)
            day = day_games(date)
            if day is None:
                print(f"{date}: games not final yet, skipped")
                continue
            skaters, scorers = day
            if not skaters:
                continue
            games, season = tims.load_schedule(date)
            defense, _ = tims.load_defense(date)
            ctx2, _ = tims_v2.load_context(date)
            ranked = {"v1": rank(tims, {"date": date, "games": games, "season": season, "defense": defense}, skaters),
                      "v2": rank(tims_v2, ctx2, skaters)}
            common = set.intersection(*({r["id"] for r in rows} for rows in ranked.values()))
            by_model = {}
            for m, rows in ranked.items():
                rows = [r for r in rows if r["id"] in common]
                preds[m] += [(date, i, r["prob"], r["id"] in scorers) for i, r in enumerate(rows, 1)]
                by_model[m] = {r["id"]: (i, r) for i, r in enumerate(rows, 1)}
            if writer:
                played_yesterday = set(tims.load_schedule((dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat())[0])
                for pid in common:
                    writer.writerow(training_row(date, season, by_model, pid, ctx2, scorers, played_yesterday))
                out.flush()
            days += 1
            print(f"{date}: {len(common)} players, {len(scorers)} scorers", flush=True)
        results.append((label, preds, days))
    if out:
        out.close()
        print(f"\nSaved training data to {args.csv}")
    for label, preds, days in results:
        if days:
            print(f"\n=== {label} ===")
            report(preds, days)
    if not any(days for _, _, days in results):
        sys.exit("No finished game days in that range.")


if __name__ == "__main__":
    main()
