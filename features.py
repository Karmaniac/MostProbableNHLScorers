"""The regression model's inputs, shared by training (backtest.py, train.py) and the daily rankings (tims.py),
so the site always computes them exactly the way the model was trained. Only the standard library is used."""

import json
import math

# (name, function of an inputs() dict). Logs of the multiplicative inputs; 0/1 flags as they are.
# The order and names must match model.json, which train.py writes from this list.
FEATURES = [
    ("log EV minutes", lambda r: math.log(max(r["ev_min"], 1))),
    ("log PP minutes", lambda r: math.log(r["pp_min"] + 0.1)),
    ("log EV shots/60", lambda r: math.log(r["ev_rate"])),
    ("log PP shots/60", lambda r: math.log(r["pp_rate"])),
    ("log EV sh%", lambda r: math.log(r["ev_sh"])),
    ("log PP sh%", lambda r: math.log(r["pp_sh"])),
    ("log opp EV defence", lambda r: math.log(r["opp_ev"])),
    ("log opp penalties", lambda r: math.log(r["opp_pen"])),
    ("log opp penalty kill", lambda r: math.log(r["opp_pk"])),
    ("log v1 shots/GP", lambda r: math.log(max(r["v1_shots_pg"], 0.1))),
    ("log v1 sh%", lambda r: math.log(r["v1_sh_pct"])),
    ("log v1 opp defence", lambda r: math.log(r["v1_opp"])),
    ("EV minutes trend", lambda r: math.log((r["ev_min_recent"] + 1) / (r["ev_min_season"] + 1))
        if r["ev_min_recent"] != "" and r["ev_min_season"] != "" else 0.0),
    ("PP minutes trend", lambda r: math.log((r["pp_min_recent"] + 0.5) / (r["pp_min_season"] + 0.5))
        if r["pp_min_recent"] != "" and r["pp_min_season"] != "" else 0.0),
    ("log games this season", lambda r: math.log1p(r["gp_season"])),
    ("log missed games", lambda r: math.log1p(r["missed"])),
    ("home", lambda r: r["home"]),
    ("defenceman", lambda r: r["pos"] == "D"),
    ("team on back-to-back", lambda r: r["team_b2b"]),
    ("opp on back-to-back", lambda r: r["opp_b2b"]),
]


def inputs(r1, r2, ctx2, played_yesterday):
    """Everything the model reads for one player tonight, from his tims.py (v1) row, his tims_v2.py row,
    tims_v2's context for the day, and the teams that played yesterday. Rounded as in the training data."""
    pid = r1["id"]
    opp = r1["opp"].split()[-1]
    s0, rec = ctx2["seasons"][0].get(pid) or {}, ctx2["recent"].get(pid) or {}

    def per_game(t, key):
        return round(t[key] / t["gp"], 3) if t.get("gp") else ""
    return {
        "pos": "D" if r1["pos"] == "D" else "F", "home": int(r1["opp"].startswith("vs ")),
        "team_b2b": int(r1["team"] in played_yesterday), "opp_b2b": int(opp in played_yesterday),
        "v1_shots_pg": round(r1["shots_pg"], 4), "v1_sh_pct": round(r1["sh_pct"], 5),
        "v1_opp": round(r1["opp_factor"], 4), "gp": r1.get("gp", 0), "missed": r1.get("missed", 0),
        "ev_min": round(r2["ev_min"], 3), "pp_min": round(r2["pp_min"], 3),
        "ev_rate": round(r2["ev_rate"], 4), "pp_rate": round(r2["pp_rate"], 4),
        "ev_sh": round(r2["ev_sh"], 5), "pp_sh": round(r2["pp_sh"], 5),
        "opp_ev": round(r2["opp_factor"], 4), "opp_pen": round(r2["pen_factor"], 4), "opp_pk": round(r2["pk_factor"], 4),
        "gp_season": s0.get("gp", 0), "ev_min_season": per_game(s0, "ev_min"), "pp_min_season": per_game(s0, "pp_min"),
        "gp_recent": rec.get("gp", 0), "ev_min_recent": per_game(rec, "ev_min"), "pp_min_recent": per_game(rec, "pp_min"),
    }


def load_model(path):
    """The regression saved by train.py, checked against FEATURES."""
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    if model.get("features") != [name for name, _ in FEATURES]:
        raise ValueError(f"{path} was trained on different inputs than features.py; rerun train.py")
    return model


def score(model, inp):
    """(xG, chance to score) for one player's inputs()."""
    z = model["intercept"] + sum(w * (float(fn(inp)) - m) / s for (_, fn), m, s, w
                                 in zip(FEATURES, model["mean"], model["std"], model["weights"]))
    xg = math.exp(min(z, 5))
    return xg, 1 - math.exp(-xg)
