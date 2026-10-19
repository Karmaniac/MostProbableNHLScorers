# Who Scores Tonight

Ranks every NHL skater playing today by their chance to score, for the Tim Hortons Hockey Challenge.
The site is published to GitHub Pages and updates itself; the ranking script uses only the Python
standard library and the NHL's public APIs.

## The site

- **Rankings** (`site/index.html`): everyone playing today, with search and filters by game and position.
  Injured or scratched players (missed their team's last 2+ games) are hidden by default.
- **Your picks**: tap ☆ beside a player to add them to the picks panel. During the evening it shows
  each pick's game score, and a 🚨 Goal badge that links to the goal's highlight video.
- **Your record**: your past picks, how many scored, total goals, and your scoring streak.
  Picks are saved in your browser, so they're per device and never leave it.
- **Past picks & results** (`site/history.html`): the model's daily top 50 and who actually scored.

## How players are ranked

Each player gets an expected-goals number for tonight (xG), turned into a chance of scoring at least once:

```
log(xG) = b0 + b1·log(PP minutes) + b2·log(EV minutes) + … + b_home·home + …   (20 inputs)
chance  = 1 − e^(−xG)        (goals treated as a Poisson process)
```

The weights come from a **Poisson regression** (`model.json`) trained on two full seasons of real games:
2024–25 to learn the weights, 2025–26 to choose its settings, then refit on both. It was tested on the
2026–27 season, which it never saw, where it beat both hand-built formulas on accuracy. Its inputs
(`features.py`), all known on the morning of the game:

- **Ice time** (the two biggest weights): expected even-strength (EV) and power-play (PP) minutes, from
  this season and the previous two, pulled toward the last two weeks (`tims_v2.py`).
- **Shooting**: EV and PP shots per 60 and shooting %, plus the original formula's shots per game
  and shooting % (below), each blended with league averages so hot streaks on small samples don't dominate.
- **Matchup**: the opponent's goals against, even-strength defence, penalties taken and penalty kill.
- **Context**: home or away, forward or defenceman, back-to-backs, games played, missed games.
- **Availability**: players who missed their team's last 2+ games drop to the bottom.

If `model.json` or the extra stats can't be loaded, `tims.py` falls back to the original formula (v1):

```
xG = shots per game × shooting % × opponent factor × home/away factor
```

1. **Shots per game**: this season and the previous two, weighted 1.0 / 0.6 / 0.3, with the last 10 games
   getting up to 30% of the weight.
2. **Shooting %**: blended with 100 league-average shots (10.5% forwards, 4.5% defence).
3. **Opponent**: goals against per game vs league average, shrunk toward average early in the season
   and capped at 0.8–1.25×.
4. **Home ice**: +3% at home, −3% away.

v1's constants are at the top of `tims.py`; v2's at the top of `tims_v2.py`.

### Retraining the model

```
python backtest.py --season 20242025 20252026 20262027 --csv training_data.csv   # about an hour
python train.py training_data.csv     # fits the regression (and LightGBM to compare), writes model.json
```

`train.py` needs NumPy (and LightGBM for the comparison); the site itself only needs the standard library.
`Model Research Report.pdf` explains the research and results in detail.

## Automatic updates

| Workflow | When (Eastern) | What it does |
|---|---|---|
| `update.yml` | 10 AM and 5 PM | Ranks today's players into `site/data.json`, logs the previous day's results to `site/history.csv` and `site/results.json`, commits them and deploys |
| `live.yml` | Every 10 min, ~7 PM–3 AM | Writes tonight's scores, goal scorers and highlight links to `site/live.json` (not committed) and deploys |

The NHL API doesn't allow requests from web pages, so the live scores come from the workflow rather
than the browser. GitHub often starts scheduled runs late, so expect scores to be 10–20 minutes behind.
Both workflows can also be run by hand from the Actions tab.

## Running it locally

```
python tims.py                    # rank everyone playing today
python tims.py players.txt        # rank only the players in the file (supports Tims groups)
python tims.py --date 2026-10-16  # a different day
python tims.py --top 25 --json site/data.json   # also write the site's data
python tims.py --live site/live.json            # just tonight's scores and scorers
python tims.py --model v1         # rank with the original formula instead of the regression
python tims.py --backfill 2026-09-29 --history site/history.csv --results site/results.json
                                  # re-rank past days as of each morning and log who scored
```

Backfilled days use only games played before each date, but today's rosters, so players traded or
sent down since then show on their current team or are missing.

`players.txt` takes one name per line, with optional `Group 1` headers and team hints like
`Elias Pettersson (VAN)`. To preview the site, run `python -m http.server -d site` and open
http://localhost:8000.

## Files

- `tims.py`: the rankings (v1 formula + regression), the history logger and the live-score writer
- `tims_v2.py`: even-strength / power-play ice time and shot rates, an input to the regression
- `features.py` and `model.json`: the regression's inputs and fitted weights
- `backtest.py`, `train.py`: rebuild the training data from past seasons, and refit and compare models
- `site/`: the published site (`index.html`, `history.html` and their data files)
- `algorithm.html`: a longer explanation of the model, with a calculator
- `.github/workflows/`: the scheduled update and live-score jobs
