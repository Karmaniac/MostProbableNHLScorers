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

Model v2 (expected goals tonight -> probability of at least one goal), with even strength (EV) and the
power play (PP) scored separately, since players shoot and score far more per minute on the PP:
    1. Ice time: EV and PP minutes per game over this season + the previous two (weighted toward this one),
       pulled toward the last two weeks so line and PP-unit changes show up quickly.
    2. Shot rate: EV and PP shots per 60 minutes, shrunk toward the league rate for the position.
    3. Finishing: EV and PP shooting %, shrunk toward the league rate for the position.
    4. Matchup: opponent even-strength goals allowed (EV part), how often they're shorthanded (more PP
       minutes) and their penalty kill (PP part), each shrunk toward average early in the season,
       plus a small home-ice adjustment.
       xG = venue * (EV min/60 * EV shots/60 * EV sh% * opp EV
                     + PP min * opp penalties/60 * PP shots/60 * PP sh% * opp PK)
    5. P(score) = 1 - exp(-xG), treating goals as a Poisson process.

Data comes from the NHL's public APIs (api-web.nhle.com and api.nhle.com/stats). Only the standard library is used.
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
STATS = "https://api.nhle.com/stats/rest/en"

# --- Model tuning ---------------------------------------------------------------
SEASON_WEIGHTS = [1.0, 0.6, 0.3]          # weight of current, last, and 2-seasons-ago shots and goals
MINUTES_SEASON_WEIGHTS = [1.0, 0.4, 0.1]  # same for ice time, leaning harder on this season (roles change)
# League rates by position, only used if last season's league totals can't be loaded (normally computed).
FALLBACK_LEAGUE = {
    "F": {"ev_rate": 7.0, "pp_rate": 13.0, "ev_sh": 0.105, "pp_sh": 0.14},
    "D": {"ev_rate": 4.0, "pp_rate": 6.5, "ev_sh": 0.045, "pp_sh": 0.06},
}
EV_RATE_PRIOR_MIN = 120                   # minutes of league-average EV shooting blended into each shot rate
PP_RATE_PRIOR_MIN = 30                    # same for the PP (players get far fewer PP minutes)
EV_SH_PRIOR_SHOTS = 100                   # league-average shots blended into each player's EV sh%
PP_SH_PRIOR_SHOTS = 50                    # same for PP sh%
FALLBACK_MINUTES = {"F": (11.0, 0.3), "D": (15.0, 0.2)}  # (EV, PP) minutes a game with no NHL history
RECENT_DAYS = 14                          # window for recent ice time (line and PP-unit changes)
RECENT_TOI_GAMES = 5                      # games in that window needed for its full weight
RECENT_TOI_WEIGHT = 0.5                   # max weight of recent ice time vs the season average
RECENT_GAMES = 10                         # games in the "Last 10" summary
OPP_PRIOR_GAMES = 10                      # shrink opponent EV goals allowed/GP toward league avg by this many games
OPP_FACTOR_RANGE = (0.8, 1.25)
PEN_PRIOR_GAMES = 10                      # same for opponent times shorthanded/GP
PEN_FACTOR_RANGE = (0.75, 1.3)
PK_PRIOR_CHANCES = 30                     # shrink opponent PP goals allowed per time shorthanded by this many
PK_FACTOR_RANGE = (0.7, 1.4)
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


def stats_report(report, cond):
    """Rows of an NHL stats report (api.nhle.com/stats), summed over the regular-season games matching `cond`."""
    q = urllib.parse.quote(f"gameTypeId=2 and {cond}")
    return get_json(f"{STATS}/{report}?isAggregate=true&isGame=true&limit=-1&cayenneExp={q}").get("data", [])


def load_defense(date, season):
    """Return ({team: {"ev", "pen", "pk"}}, league avg GA/GP), from games before `date`. Each factor is the
    team vs league average, shrunk toward average early in the season: ev = even-strength goals allowed/GP,
    pen = times shorthanded/GP (more PP minutes for the other side), pk = PP goals allowed per time shorthanded."""
    # Standings for a date include that day's games, so use the day before.
    before = (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()
    rows = get_json(f"{API}/standings/{before}").get("standings", [])
    total_gp = sum(r["gamesPlayed"] for r in rows)
    if not total_gp:
        return {}, None
    league_avg = sum(r["goalAgainst"] for r in rows) / total_gp
    try:
        # Date-filtered team reports name teams by franchiseName (e.g. "Utah Mammoth"), the same as standings.
        kills = {r.get("teamFullName") or r["franchiseName"]: r
                 for r in stats_report("team/penaltykill", f'seasonId={season} and gameDate<"{date}"')}
    except Exception:  # without PK stats, fall back to all goals against and neutral PP factors
        kills = {}
    teams = []
    for r in rows:
        k = kills.get(r["teamName"]["default"], {})
        pp_ga, times_sh = k.get("ppGoalsAgainst") or 0, k.get("timesShorthanded") or 0
        teams.append((r["teamAbbrev"]["default"], r["gamesPlayed"], r["goalAgainst"] - pp_ga, times_sh, pp_ga))
    league_ev = sum(t[2] for t in teams) / total_gp
    league_pen = sum(t[3] for t in teams) / total_gp
    league_pk = sum(t[4] for t in teams) / max(sum(t[3] for t in teams), 1)

    def shrink(value, n, avg, prior, bounds):
        return min(max((value + avg * prior) / (n + prior) / avg, bounds[0]), bounds[1]) if avg else 1.0
    factors = {abbrev: {"ev": shrink(ev_ga, gp, league_ev, OPP_PRIOR_GAMES, OPP_FACTOR_RANGE),
                        "pen": shrink(times_sh, gp, league_pen, PEN_PRIOR_GAMES, PEN_FACTOR_RANGE),
                        "pk": shrink(pp_ga, times_sh, league_pk, PK_PRIOR_CHANCES, PK_FACTOR_RANGE)}
               for abbrev, gp, ev_ga, times_sh, pp_ga in teams}
    return factors, league_avg


def skater_window(cond):
    """{player id: totals} for one stretch of games, ice time in minutes. Shots and goals that weren't on the
    power play count as even strength (shorthanded ones are rare)."""
    rows = {}
    for r in stats_report("skater/summary", cond):
        rows[r["playerId"]] = {"pos": "D" if r["positionCode"] == "D" else "F", "gp": r["gamesPlayed"] or 0,
                               "goals": r["goals"] or 0, "shots": r["shots"] or 0, "pp_goals": r["ppGoals"] or 0,
                               "pp_shots": 0, "ev_min": 0.0, "pp_min": 0.0}
    for r in stats_report("skater/powerplay", cond):
        if r["playerId"] in rows:
            rows[r["playerId"]]["pp_shots"] = r["ppShots"] or 0
    for r in stats_report("skater/timeonice", cond):
        if r["playerId"] in rows:
            rows[r["playerId"]].update(ev_min=(r["evTimeOnIce"] or 0) / 60, pp_min=(r["ppTimeOnIce"] or 0) / 60)
    for t in rows.values():
        t["ev_shots"] = max(t["shots"] - t["pp_shots"], 0)
        t["ev_goals"] = max(t["goals"] - t["pp_goals"], 0)
    return rows


def league_rates(window):
    """League shot rates (per 60) and shooting % by position and strength, from one season's totals."""
    out = {}
    for pos, fallback in FALLBACK_LEAGUE.items():
        ts = [t for t in window.values() if t["pos"] == pos]
        s = {k: sum(t[k] for t in ts) for k in ("ev_shots", "ev_goals", "ev_min", "pp_shots", "pp_goals", "pp_min")}
        out[pos] = {
            "ev_rate": 60 * s["ev_shots"] / s["ev_min"] if s["ev_min"] else fallback["ev_rate"],
            "pp_rate": 60 * s["pp_shots"] / s["pp_min"] if s["pp_min"] else fallback["pp_rate"],
            "ev_sh": s["ev_goals"] / s["ev_shots"] if s["ev_shots"] else fallback["ev_sh"],
            "pp_sh": s["pp_goals"] / s["pp_shots"] if s["pp_shots"] else fallback["pp_sh"],
        }
    return out


def load_context(date):
    """Everything analyze() needs for one game day, using only games played before it. Returns (ctx, league avg
    GA/GP), or (None, None) on a day with no games."""
    games, season = load_schedule(date)
    if not games:
        return None, None
    defense, league_avg = load_defense(date, season)
    recent_start = (dt.date.fromisoformat(date) - dt.timedelta(days=RECENT_DAYS)).isoformat()
    seasons = [skater_window(f'seasonId={season} and gameDate<"{date}"'),
               skater_window(f"seasonId={season - 10001}"),
               skater_window(f"seasonId={season - 20002}")]
    recent = skater_window(f'seasonId={season} and gameDate>="{recent_start}" and gameDate<"{date}"')
    ctx = {"date": date, "games": games, "season": season, "defense": defense,
           "seasons": seasons, "recent": recent, "league": league_rates(seasons[1])}
    return ctx, league_avg


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

        row.update(gp=len(log), goals=sum(g.get("goals", 0) for g in log))
        recent = log[:RECENT_GAMES]
        if recent:
            row["l10"] = f"{sum(g.get('goals', 0) for g in recent)}G/{sum(g.get('shots', 0) for g in recent)}S in {len(recent)}"

        # Shot rates per 60 minutes and shooting %, by strength, over this season and the previous two,
        # each shrunk toward the league rate for the position.
        lg = ctx["league"][pos]
        hist = [s.get(pid) or {} for s in ctx["seasons"]]

        def wsum(key, weights=SEASON_WEIGHTS):
            return sum(w * h.get(key, 0) for w, h in zip(weights, hist))
        ev_rate = 60 * (wsum("ev_shots") + lg["ev_rate"] / 60 * EV_RATE_PRIOR_MIN) / (wsum("ev_min") + EV_RATE_PRIOR_MIN)
        pp_rate = 60 * (wsum("pp_shots") + lg["pp_rate"] / 60 * PP_RATE_PRIOR_MIN) / (wsum("pp_min") + PP_RATE_PRIOR_MIN)
        ev_sh = (wsum("ev_goals") + lg["ev_sh"] * EV_SH_PRIOR_SHOTS) / (wsum("ev_shots") + EV_SH_PRIOR_SHOTS)
        pp_sh = (wsum("pp_goals") + lg["pp_sh"] * PP_SH_PRIOR_SHOTS) / (wsum("pp_shots") + PP_SH_PRIOR_SHOTS)

        # Expected minutes tonight: weighted season average per game, pulled toward the last two weeks.
        gp_w = wsum("gp", MINUTES_SEASON_WEIGHTS)
        if gp_w:
            ev_min = wsum("ev_min", MINUTES_SEASON_WEIGHTS) / gp_w
            pp_min = wsum("pp_min", MINUTES_SEASON_WEIGHTS) / gp_w
        else:
            ev_min, pp_min = FALLBACK_MINUTES[pos]
            row["notes"].append("no recent NHL history")
        rec = ctx["recent"].get(pid)
        if rec and rec["gp"]:
            w = RECENT_TOI_WEIGHT * min(rec["gp"], RECENT_TOI_GAMES) / RECENT_TOI_GAMES
            ev_min = ev_min * (1 - w) + rec["ev_min"] / rec["gp"] * w
            pp_min = pp_min * (1 - w) + rec["pp_min"] / rec["gp"] * w
        ev_shots, pp_shots = ev_min / 60 * ev_rate, pp_min / 60 * pp_rate
        shots_pg = ev_shots + pp_shots
        sh_pct = (ev_shots * ev_sh + pp_shots * pp_sh) / shots_pg if shots_pg else lg["ev_sh"]
        row.update(ev_min=ev_min, pp_min=pp_min, ev_rate=ev_rate, pp_rate=pp_rate, ev_sh=ev_sh, pp_sh=pp_sh)

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

        opp = ctx["defense"].get(game["opp"], {})
        opp_ev, pen, pk = opp.get("ev", 1.0), opp.get("pen", 1.0), opp.get("pk", 1.0)
        venue = HOME_FACTOR if game["home"] else AWAY_FACTOR
        ev_xg = ev_shots * ev_sh * opp_ev * venue
        pp_xg = pp_shots * pen * pp_sh * pk * venue
        xg = ev_xg + pp_xg
        row.update(
            opp=("vs " if game["home"] else "@ ") + game["opp"],
            opp_factor=opp_ev,
            pen_factor=pen,
            pk_factor=pk,
            pp_xg=pp_xg,
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
          f"{'EVmin':>6}{'PPmin':>6}{'xS':>5}{'Sh%':>6}  {'Last 10':<13}{'OppD':>5}{'PK':>5}{'xG':>6}{'P(goal)':>9}")
    def num(key, width, fmt, scale=1, suffix=""):
        return f"{format(r[key] * scale, fmt) + suffix:>{width}}" if key in r else f"{'-':>{width}}"

    for i, r in enumerate(rows, 1):
        print(f"  {i:>3}  {r['name']:<{w}}{r['pos']:<4}{r['team']:<5}{r.get('opp', '-'):<7}"
              f"{r.get('gp', '-'):>3}{r.get('goals', '-'):>3}"
              f"{num('ev_min', 6, '.1f')}{num('pp_min', 6, '.1f')}{num('shots_pg', 5, '.2f')}"
              f"{num('sh_pct', 6, '.1f', 100, '%')}  {r.get('l10', '-'):<13}"
              f"{num('opp_factor', 5, '.2f')}{num('pk_factor', 5, '.2f')}{r['xg']:>6.2f}{r['prob'] * 100:>8.1f}%")
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
            "ev_min": round(r["ev_min"], 1) if "ev_min" in r else None,
            "pp_min": round(r["pp_min"], 1) if "pp_min" in r else None,
            "pk_factor": round(r["pk_factor"], 3) if "pk_factor" in r else None,
            "pp_xg": round(r["pp_xg"], 3) if "pp_xg" in r else None,
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
            ctx, league_avg = load_context(date)
            if not ctx:
                continue
            ranked = rank_players(ctx, [(f"Everyone playing {date}", load_skaters(ctx["games"]))], league_avg)
            write_json(day_json, date, ctx["games"], league_avg, ranked)
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

    ctx, league_avg = load_context(args.date)
    if not ctx:
        if args.json:  # no games is a normal day for the website, not an error
            write_json(args.json, args.date, {}, None, [])
            print(f"No NHL games on {args.date}; wrote an empty {args.json}.")
            return
        sys.exit(f"No NHL regular-season/playoff games on {args.date}.")
    games = ctx["games"]

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
