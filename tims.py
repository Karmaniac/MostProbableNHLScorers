"""Rank NHL players by their chance of scoring tonight, for the Tim Hortons Hockey Challenge.

Usage:
    py tims.py                  rank every skater playing today
    py tims.py players.txt      rank only the players in the file
    Options: --date YYYY-MM-DD (default today), --top N (show only the top N per group),
             --json PATH (also save the rankings for the website in site/),
             --live PATH (only save that day's live scores and goal scorers),
             --backfill START --history CSV [--results JSON] (re-rank and log past days)

players.txt: one player per line. Optional extras:
    - Team hint to disambiguate:  Elias Pettersson (VAN)   or   Sebastian Aho, CAR
    - Group headers (Tims has you pick one player per group):  Group 1   /   Group 2:
    - Lines starting with # are ignored.

Model (expected goals tonight -> probability of at least one goal):
    1. Shot volume: shots per game over this season + the previous two, weighted toward recent seasons.
    2. Finishing: shooting % regressed toward a positional average so small samples don't fool us.
    3. Recent form: shots per game over the last 10 games nudges the volume estimate (line/PP changes).
    4. Matchup: opponent goals-against per game vs league average (shrunk early in the season),
       plus a small home-ice adjustment.
    5. P(score) = 1 - exp(-xG), treating goals as a Poisson process.

Data comes from the NHL's public API (api-web.nhle.com). Only the standard library is used.
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

API = "https://api-web.nhle.com/v1"
SEARCH = "https://search.d3.nhle.com/api/v1/search/player"

# --- Model tuning ---------------------------------------------------------------
SEASON_WEIGHTS = [1.0, 0.6, 0.3]          # per-game weight for current, last, and 2-seasons-ago
PRIOR_SH_PCT = {"F": 0.105, "D": 0.045}  # league-typical shooting % by position
PRIOR_SHOTS = 100                        # how many "average" shots to blend into each player's sh%
FALLBACK_SHOTS_PG = {"F": 1.4, "D": 1.0} # used when a player has no NHL history
RECENT_GAMES = 10
RECENT_WEIGHT = 0.3                      # max weight of last-10 shot rate vs the long-run rate
OPP_PRIOR_GAMES = 10                     # shrink opponent GA/GP toward league avg by this many games
OPP_FACTOR_RANGE = (0.8, 1.25)
HOME_FACTOR, AWAY_FACTOR = 1.03, 0.97
MISSED_GAMES_WARNING = 2                 # flag players who sat out this many straight team games
HISTORY_TOP = 50                         # how many of each day's picks to keep in the history CSV
HISTORY_FIELDS = ["date", "rank", "player_id", "name", "pos", "team", "opp", "gp", "goals", "shots_pg",
                  "sh_pct", "l10", "opp_factor", "xg", "prob", "flagged", "scored"]


@lru_cache(maxsize=None)  # one run asks for the same player pages once per day it ranks (--backfill)
def get_json(url, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "tims-picker/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)


def normalize(name):
    """Lowercase, strip accents and punctuation so 'Stützle' == 'stutzle'."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def parse_player_file(path):
    """Return a list of (group_name, [(player_name, team_hint), ...])."""
    header = re.compile(r"^\[?\s*(group|tier|list)\b.*$|^.+:$", re.IGNORECASE)
    hint = re.compile(r"^(.*?)\s*(?:\(([A-Za-z]{3})\)|,\s*([A-Za-z]{3}))\s*$")
    groups = [["All players", []]]
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if header.match(line):
                groups.append([line.strip("[]: "), []])
                continue
            m = hint.match(line)
            if m:
                groups[-1][1].append((m.group(1), (m.group(2) or m.group(3)).upper()))
            else:
                groups[-1][1].append((line, None))
    return [(g, players) for g, players in groups if players]


# --- Cached league-wide lookups --------------------------------------------------

def load_schedule(date):
    """Map team abbrev -> game info for every game on `date`. Returns (games, season_id)."""
    data = get_json(f"{API}/schedule/{date}")
    day = next((d for d in data.get("gameWeek", []) if d["date"] == date), None)
    games, season = {}, None
    for g in (day or {}).get("games", []):
        if g.get("gameType") not in (2, 3):  # regular season / playoffs only
            continue
        season = g["season"]
        home, away = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        games[home] = {"opp": away, "home": True, "start": g["startTimeUTC"]}
        games[away] = {"opp": home, "home": False, "start": g["startTimeUTC"]}
    return games, season


def load_defense(date):
    """Return ({team: opponent-strength factor}, league avg GA/GP) from standings before `date`."""
    # Standings for a date include that day's games, so use the day before.
    before = (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()
    rows = get_json(f"{API}/standings/{before}").get("standings", [])
    total_gp = sum(r["gamesPlayed"] for r in rows)
    if not total_gp:
        return {}, None
    league_avg = sum(r["goalAgainst"] for r in rows) / total_gp
    factors = {}
    for r in rows:
        shrunk = (r["goalAgainst"] + league_avg * OPP_PRIOR_GAMES) / (r["gamesPlayed"] + OPP_PRIOR_GAMES)
        factors[r["teamAbbrev"]["default"]] = min(max(shrunk / league_avg, OPP_FACTOR_RANGE[0]), OPP_FACTOR_RANGE[1])
    return factors, league_avg


@lru_cache(maxsize=None)
def team_completed_game_dates(team, season, before_date):
    data = get_json(f"{API}/club-schedule-season/{team}/{season}")
    return sorted(
        g["gameDate"] for g in data.get("games", [])
        if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL") and g["gameDate"] < before_date
    )


# --- Per-player analysis ---------------------------------------------------------

def find_player(name, team_hint):
    """Resolve a name to a search result. Returns (result or None, note)."""
    q = urllib.parse.quote(name)
    results = get_json(f"{SEARCH}?culture=en-us&limit=20&q={q}&active=true") or []
    results = [r for r in results if r.get("positionCode") != "G"]
    if not results:
        return None, "player not found"
    exact = [r for r in results if normalize(r["name"]) == normalize(name)]
    cands = exact or results
    if team_hint:
        cands = [r for r in cands if (r.get("teamAbbrev") or "").upper() == team_hint] or cands
    note = ""
    if not exact:
        note = f"closest match: {cands[0]['name']}"
    elif len(cands) > 1:
        cands.sort(key=lambda r: r.get("positionCode") == "D")  # prefer forwards
        note = f"{len(cands)} players named this; picked {cands[0]['positionCode']} {cands[0].get('teamAbbrev')} (add a team hint)"
    return cands[0], note


def season_totals(landing, season):
    """Sum NHL regular-season rows for one season (traded players have several rows)."""
    gp = goals = shots = 0
    for s in landing.get("seasonTotals", []):
        if s.get("leagueAbbrev") == "NHL" and s.get("gameTypeId") == 2 and s.get("season") == season:
            gp += s.get("gamesPlayed", 0)
            goals += s.get("goals", 0)
            shots += s.get("shots", 0)
    return gp, goals, shots


def load_skaters(teams):
    """Every forward and defenceman on the given teams' current rosters, as (name, team, player_id)."""
    def roster(team):
        data = get_json(f"{API}/roster/{team}/current")
        return [(f"{p['firstName']['default']} {p['lastName']['default']}", team, p["id"])
                for p in data.get("forwards", []) + data.get("defensemen", [])]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return [p for players in pool.map(roster, sorted(teams)) for p in players]


def analyze(name, team_hint, ctx, pid=None):
    """Score one player. Pass `pid` to skip the name search (all-players mode)."""
    row = {"input": name, "name": name, "team": "?", "pos": "?", "prob": 0.0, "xg": 0.0, "notes": []}
    try:
        match = {"playerId": pid, "teamAbbrev": team_hint}
        if pid is None:
            match, note = find_player(name, team_hint)
            if note:
                row["notes"].append(note)
            if not match:
                return row
            pid = match["playerId"]
        landing = get_json(f"{API}/player/{pid}/landing")
        row["name"] = f"{landing['firstName']['default']} {landing['lastName']['default']}"
        team = landing.get("currentTeamAbbrev") or match.get("teamAbbrev") or "?"
        pos = "D" if landing.get("position") == "D" else "F"
        row.update(id=pid, team=team, pos=landing.get("position", "?"))

        season = ctx["season"]
        # This season's games before the date (so past dates can be re-ranked), newest first.
        log = sorted((g for g in get_json(f"{API}/player/{pid}/game-log/{season}/2").get("gameLog", [])
                      if g["gameDate"] < ctx["date"]), key=lambda g: g["gameDate"], reverse=True)

        # Weighted multi-season shot rate and finishing.
        wgp = wgoals = wshots = 0.0
        for i, w in enumerate(SEASON_WEIGHTS):
            if i == 0:
                gp, goals, shots = len(log), sum(g.get("goals", 0) for g in log), sum(g.get("shots", 0) for g in log)
                row.update(gp=gp, goals=goals)
            else:
                gp, goals, shots = season_totals(landing, season - 10001 * i)
            wgp += w * gp
            wgoals += w * goals
            wshots += w * shots
        shots_pg = wshots / wgp if wgp else FALLBACK_SHOTS_PG[pos]
        if not wgp:
            row["notes"].append("no recent NHL history")
        sh_pct = (wgoals + PRIOR_SH_PCT[pos] * PRIOR_SHOTS) / (wshots + PRIOR_SHOTS)

        # Recent form from this season's game log.
        recent = log[:RECENT_GAMES]
        if recent:
            recent_spg = sum(g.get("shots", 0) for g in recent) / len(recent)
            w = RECENT_WEIGHT * len(recent) / RECENT_GAMES
            shots_pg = shots_pg * (1 - w) + recent_spg * w
            row["l10"] = f"{sum(g.get('goals', 0) for g in recent)}G/{sum(g.get('shots', 0) for g in recent)}S in {len(recent)}"

        # Has the player been missing team games (injury / healthy scratch)?
        team_dates = team_completed_game_dates(team, season, ctx["date"]) if team != "?" else []
        last_played = log[0]["gameDate"] if log else ""
        missed = sum(1 for d in team_dates if d > last_played)
        if missed >= MISSED_GAMES_WARNING:
            row["notes"].append(f"!! missed last {missed} team games (injured/scratched?)")

        row["shots_pg"], row["sh_pct"] = shots_pg, sh_pct
        game = ctx["games"].get(team)
        if not game:
            row["notes"].append(f"{team} not playing {ctx['date']}")
            return row

        opp_factor = ctx["defense"].get(game["opp"], 1.0)
        venue = HOME_FACTOR if game["home"] else AWAY_FACTOR
        xg = shots_pg * sh_pct * opp_factor * venue
        row.update(
            opp=("vs " if game["home"] else "@ ") + game["opp"],
            opp_factor=opp_factor,
            xg=xg,
            prob=1 - math.exp(-xg),
            missed=missed,
        )
    except Exception as e:  # keep going if one lookup fails
        row["notes"].append(f"error: {e}")
    return row


# --- Output ----------------------------------------------------------------------

def sort_key(r):
    # Players with an injury/scratch warning drop below everyone who's been dressing.
    return (r.get("missed", 0) < MISSED_GAMES_WARNING, r["prob"])


def print_table(rows):
    w = max(len(r["name"]) for r in rows) + 2
    print(f"  {'#':>3}  {'Player':<{w}}{'Pos':<4}{'Team':<5}{'Opp':<7}{'GP':>3}{'G':>3}"
          f"{'S/GP':>6}{'Sh%':>6}  {'Last 10':<13}{'OppD':>5}{'xG':>6}{'P(goal)':>9}")
    def num(key, width, fmt, scale=1, suffix=""):
        return f"{format(r[key] * scale, fmt) + suffix:>{width}}" if key in r else f"{'-':>{width}}"

    for i, r in enumerate(rows, 1):
        print(f"  {i:>3}  {r['name']:<{w}}{r['pos']:<4}{r['team']:<5}{r.get('opp', '-'):<7}"
              f"{r.get('gp', '-'):>3}{r.get('goals', '-'):>3}"
              f"{num('shots_pg', 6, '.2f')}{num('sh_pct', 6, '.1f', 100, '%')}  {r.get('l10', '-'):<13}"
              f"{num('opp_factor', 5, '.2f')}{r['xg']:>6.2f}{r['prob'] * 100:>8.1f}%")
        for n in r["notes"]:
            print(f"      ^ {n}")


def write_json(path, date, games, league_avg, ranked_groups):
    """Save the rankings for the website (site/index.html reads this file)."""
    def player(rank, r):
        return {
            "rank": rank, "id": r.get("id"), "name": r["name"], "pos": r["pos"], "team": r["team"], "opp": r.get("opp"),
            "gp": r.get("gp"), "goals": r.get("goals"),
            "shots_pg": round(r["shots_pg"], 2) if "shots_pg" in r else None,
            "sh_pct": round(r["sh_pct"], 4) if "sh_pct" in r else None,
            "l10": r.get("l10"),
            "opp_factor": round(r["opp_factor"], 3) if "opp_factor" in r else None,
            "xg": round(r["xg"], 3), "prob": round(r["prob"], 4),
            "flagged": r.get("missed", 0) >= MISSED_GAMES_WARNING,
            "notes": r["notes"],
        }
    payload = {
        "date": date,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "league_avg_ga": round(league_avg, 3) if league_avg else None,
        "games": sorted(
            ({"away": g["opp"], "home": team, "start": g["start"]} for team, g in games.items() if g["home"]),
            key=lambda g: (g["start"], g["home"]),
        ),
        "groups": [{"name": name, "players": [player(i, r) for i, r in enumerate(rows, 1)]}
                   for name, rows in ranked_groups],
    }
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def goal_key(goal):
    """TEAM|First Last, matching the website's key for a data.json row (names come from the same NHL API)."""
    return f"{goal['teamAbbrev']}|{goal['firstName']['default']} {goal['lastName']['default']}"


def counted_goals(game):
    """A game's goals that count for Tims (shootout goals don't)."""
    return [g for g in game.get("goals", []) if g["periodDescriptor"]["periodType"] != "SO"]


def write_live(path, date):
    """Save tonight's scores and goal scorers for the website's picks panel (site/live.json).
    scorers maps TEAM|Name -> one highlight clip URL (or null until the NHL posts it) per goal."""
    games, scorers = [], {}
    for g in get_json(f"{API}/score/{date}").get("games", []):
        away, home = g["awayTeam"], g["homeTeam"]
        games.append({"away": away["abbrev"], "home": home["abbrev"], "state": g["gameState"],
                      "away_score": away.get("score"), "home_score": home.get("score"),
                      "period": g.get("period"), "clock": (g.get("clock") or {}).get("timeRemaining"),
                      "intermission": (g.get("clock") or {}).get("inIntermission", False)})
        for goal in counted_goals(g):
            scorers.setdefault(goal_key(goal), []).append(goal.get("highlightClipSharingUrl"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": date, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "games": games, "scorers": scorers}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved live scores for {date} ({len(games)} games, {sum(map(len, scorers.values()))} goals) to {path}")


def record_history(json_path, csv_path, today, results_path=None):
    """Append the last saved day's top picks to the history CSV, with whether each one scored.
    Also saves that day's goal counts per TEAM|Name to results_path (the website's pick scorecard).
    Call before data.json is overwritten. Skips days already recorded or with games still to finish."""
    try:
        with open(json_path, encoding="utf-8") as f:
            day = json.load(f)
    except FileNotFoundError:
        return
    date = day["date"]
    if date >= today or not day["groups"]:
        return
    exists = os.path.exists(csv_path)
    if exists:
        with open(csv_path, newline="", encoding="utf-8") as f:
            if any(r["date"] == date for r in csv.DictReader(f)):
                return
    games = get_json(f"{API}/score/{date}").get("games", [])
    if any(g.get("gameScheduleState") == "OK" and g.get("gameState") not in ("OFF", "FINAL") for g in games):
        print(f"History: {date} games not final yet; will record on a later run.")
        return
    goals = [g for game in games for g in counted_goals(game)]
    if results_path:
        # ponytail: grows ~1.5 KB per game day (~300 KB a season); prune old dates if it gets slow to load.
        try:
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)
        except FileNotFoundError:
            results = {}
        day_goals = {}
        for g in goals:
            day_goals[goal_key(g)] = day_goals.get(goal_key(g), 0) + 1
        results[date] = day_goals
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    ids = {g["playerId"] for g in goals}
    names = {normalize(f"{g['firstName']['default']} {g['lastName']['default']}") for g in goals}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, HISTORY_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for p in day["groups"][0]["players"][:HISTORY_TOP]:
            # Older data.json files have no player ids, so fall back to the name.
            scored = p["id"] in ids if p.get("id") else normalize(p["name"]) in names
            w.writerow({**p, "date": date, "player_id": p.get("id"), "scored": int(scored)})
    print(f"History: recorded top {HISTORY_TOP} picks for {date} in {csv_path}.")


def rank_players(ctx, groups, league_avg):
    """Analyze every player in [(group name, [(name, team_hint, player_id)])]; returns [(group name, sorted rows)]."""
    all_players = [p for _, players in groups for p in players]
    print(f"Analyzing {len(all_players)} players for {ctx['date']} ({len(ctx['games']) // 2} games)", end="")
    print(f", league avg {league_avg:.2f} GA/GP" if league_avg else "", end="\n\n")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda p: analyze(p[0], p[1], ctx, p[2]), all_players))
    ranked, i = [], 0
    for group_name, players in groups:
        ranked.append((group_name, sorted(results[i:i + len(players)], key=sort_key, reverse=True)))
        i += len(players)
    return ranked


def backfill(start, today, csv_path, results_path):
    """Re-rank every game day from `start` to yesterday as if it were that morning, and log each one
    to the history CSV and results JSON (days already in the CSV are skipped).
    ponytail: uses today's rosters, so players since traded/sent down are on their current team or missing."""
    d = dt.date.fromisoformat(start)
    with tempfile.TemporaryDirectory() as tmp:
        day_json = os.path.join(tmp, "day.json")
        while d.isoformat() < today:
            date, d = d.isoformat(), d + dt.timedelta(days=1)
            games, season = load_schedule(date)
            if not games:
                continue
            defense, league_avg = load_defense(date)
            ctx = {"date": date, "games": games, "season": season, "defense": defense}
            ranked = rank_players(ctx, [(f"Everyone playing {date}", load_skaters(games))], league_avg)
            write_json(day_json, date, games, league_avg, ranked)
            record_history(day_json, csv_path, today, results_path)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Rank players by chance to score tonight (Tims Hockey Challenge).")
    ap.add_argument("file", nargs="?",
                    help="text file with one player name per line (omit to rank every skater playing that day)")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="game date, YYYY-MM-DD (default: today)")
    ap.add_argument("--top", type=int, help="only show the top N players in each group")
    ap.add_argument("--json", metavar="PATH", help="also save the full rankings as JSON (used by the website)")
    ap.add_argument("--history", metavar="CSV",
                    help="before overwriting --json, append its day's top picks and whether they scored to CSV")
    ap.add_argument("--results", metavar="PATH",
                    help="with --history, also save each finished day's goal scorers to this JSON (pick scorecard)")
    ap.add_argument("--backfill", metavar="START",
                    help="with --history, re-rank every game day from START (YYYY-MM-DD) to before --date and log them")
    ap.add_argument("--live", metavar="PATH", help="only save the date's live scores and goal scorers as JSON")
    args = ap.parse_args()

    if args.live:
        write_live(args.live, args.date)
        return

    if args.backfill:
        if not args.history:
            sys.exit("--backfill needs --history CSV (and optionally --results JSON).")
        backfill(args.backfill, args.date, args.history, args.results)
        return

    if args.history and args.json:
        try:
            record_history(args.json, args.history, args.date, args.results)
        except Exception as e:  # never let history break the day's picks
            print(f"History: failed ({e})")

    games, season = load_schedule(args.date)
    if not games:
        if args.json:  # no games is a normal day for the website, not an error
            write_json(args.json, args.date, {}, None, [])
            print(f"No NHL games on {args.date}; wrote an empty {args.json}.")
            return
        sys.exit(f"No NHL regular-season/playoff games on {args.date}.")
    defense, league_avg = load_defense(args.date)
    ctx = {"date": args.date, "games": games, "season": season, "defense": defense}

    if args.file:
        groups = parse_player_file(args.file)
        if not groups:
            sys.exit("No player names found in the file.")
        # (name, team_hint, player_id) — no id yet, so analyze() searches by name.
        groups = [(g, [(name, hint, None) for name, hint in players]) for g, players in groups]
    else:
        groups = [(f"Everyone playing {args.date}", load_skaters(games))]

    ranked = rank_players(ctx, groups, league_avg)
    picks = []
    for group_name, rows in ranked:
        print(f"== {group_name} ==")
        print_table(rows[:args.top])
        print()
        if rows[0]["prob"] > 0:
            picks.append((group_name, rows[0]))

    if args.json:
        write_json(args.json, args.date, games, league_avg, ranked)
        print(f"Saved rankings to {args.json}\n")

    print("Recommended picks:")
    for group_name, r in picks:
        label = f"{group_name}: " if len(groups) > 1 else ""
        print(f"  {label}{r['name']} ({r['team']} {r.get('opp', '')}) - {r['prob'] * 100:.1f}% to score")


if __name__ == "__main__":
    main()
