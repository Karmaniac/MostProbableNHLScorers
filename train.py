"""Fit a Poisson regression and a LightGBM model to backtest.py's training data, and compare both with
tims.py (v1) and tims_v2.py.

Usage:
    py train.py training_data.csv [--out model.json] [--no-lightgbm]

LightGBM (gradient-boosted decision trees, Poisson objective) uses the same 20 inputs but can learn
interactions and thresholds the regression can't, e.g. "PP minutes matter more against a weak penalty kill".
Its settings (tree size, leaf size) are chosen on the validation season, and the number of trees by early
stopping there. It's saved to model_lightgbm.txt; using it in tims.py would need the lightgbm package.

The model keeps the shape of the hand-built ones: expected goals is a product of factors,
    xG = exp(b0 + b1*log(EV minutes) + b2*log(PP minutes) + ... + b_home*home + ...)
so each weight says how strongly that input moves tonight's xG, learned from who actually scored.
P(score) = 1 - exp(-xG), as before.

Seasons are split by time, never at random (a random split lets the model peek at the future):
    train = every season but the last two, validate = second-last (picks the regularization strength),
    test = the last one (reported once, after everything is chosen).
The saved model is refit on train + validate; using it in tims.py needs only the standard library.

Needs NumPy, plus LightGBM for the trees (py -m pip install lightgbm). Everything runs on the CPU.
"""

import argparse
import csv
import json
import math
import sys
import time

import numpy as np

from backtest import report
from features import FEATURES

LAMBDAS = [0.1, 1, 10, 100, 1000]  # L2 strengths tried on the validation season
MAX_STEPS = 25
TEXT = {"date", "name", "pos", "team", "opp"}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in TEXT and v != "":
                r[k] = float(v)
    return rows


def design(rows, mean=None, std=None):
    """Feature matrix with an intercept column, standardized (with the training mean/std when given)."""
    X = np.array([[float(fn(r)) for _, fn in FEATURES] for r in rows])
    if mean is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
    return np.hstack([np.ones((len(X), 1)), (X - mean) / std]), mean, std


def fit(X, y, lam, label):
    """Poisson regression with an L2 penalty (not on the intercept), by Newton's method. Prints each step."""
    p = X.shape[1]
    beta = np.zeros(p)
    beta[0] = math.log(float(y.mean()))
    pen = np.full(p, float(lam))
    pen[0] = 0
    start = time.perf_counter()
    for step_no in range(1, MAX_STEPS + 1):
        mu = np.exp(np.minimum(X @ beta, 5))
        grad = X.T @ (y - mu) - pen * beta
        hess = (X * mu[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        change = float(np.abs(step).max())
        loss = float((mu - y * np.log(mu)).mean())
        print(f"    {label}  step {step_no:>2}  change {change:.1e}  train loss {loss:.5f}  {time.perf_counter() - start:6.2f}s",
              flush=True)
        if change < 1e-7:
            break
    return beta, step_no, time.perf_counter() - start


def predict(beta, X):
    return 1 - np.exp(-np.exp(np.minimum(X @ beta, 5)))


def log_loss(probs, rows):
    y = np.array([r["scored"] for r in rows])
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def predictions(rows, models):
    """{model: [(date, rank, prob, scored)]} for v1, v2 and each trained model ({name: probs}) on the same
    player-games. Every model is ranked by its own chances within each day."""
    probs = {"v1": [r["v1_prob"] for r in rows], "v2": [r["v2_prob"] for r in rows], **models}
    preds = {}
    for m, ps in probs.items():
        by_day = {}
        for r, p in zip(rows, ps):
            by_day.setdefault(r["date"], []).append((float(p), bool(r["scored"])))
        preds[m] = [(date, i, p, y) for date, day in by_day.items()
                    for i, (p, y) in enumerate(sorted(day, key=lambda py: -py[0]), 1)]
    return preds, len({r["date"] for r in rows})


def luck_test(preds, a, b, top, samples=2000, seed=1):
    """Extra scorers in model a's daily top N vs model b's, with a 95% range from resampling the days."""
    def per_day(m):
        hits = {}
        for date, rank, _, y in preds[m]:
            if rank <= top:
                hits[date] = hits.get(date, 0) + y
        return hits
    ha, hb = per_day(a), per_day(b)
    diffs = np.array([ha.get(d, 0) - hb.get(d, 0) for d in sorted(set(ha) | set(hb))])
    rng = np.random.default_rng(seed)
    totals = np.sort(rng.choice(diffs, size=(samples, len(diffs))).sum(axis=1))
    return int(diffs.sum()), int(totals[int(samples * 0.025)]), int(totals[int(samples * 0.975) - 1])


# --- LightGBM: gradient-boosted trees on the same inputs ---------------------------------------
LGB_BASE = dict(objective="poisson", learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=10.0, verbose=-1, seed=1)
LGB_GRID = [dict(num_leaves=leaves, min_data_in_leaf=leaf_size) for leaves in (7, 15, 31) for leaf_size in (100, 400)]
LGB_MAX_ROUNDS = 3000


def lgb_fit(X, y, params, rounds, valid=None, label=""):
    """Train boosted trees; with `valid` = (X, y), stop once the validation score stops improving."""
    import lightgbm as lgb
    start = time.perf_counter()
    train_set = lgb.Dataset(X, y, feature_name=[f"f{i}" for i in range(X.shape[1])])
    callbacks = [lgb.log_evaluation(period=250)]
    sets, names = [], []
    if valid is not None:
        sets, names = [lgb.Dataset(*valid, reference=train_set)], ["validation"]
        callbacks.append(lgb.early_stopping(150, verbose=False))
    if label:
        print(f"    {label}", flush=True)
    booster = lgb.train({**LGB_BASE, **params}, train_set, num_boost_round=rounds,
                        valid_sets=sets, valid_names=names, callbacks=callbacks)
    return booster, time.perf_counter() - start


def lgb_predict(booster, X):
    return 1 - np.exp(-booster.predict(X, num_iteration=booster.best_iteration or None))


def summary(preds):
    """Headline numbers per model: log loss, Brier, and the share of each day's top N who scored."""
    out = {}
    for m, ps in preds.items():
        p = np.clip(np.array([x[2] for x in ps]), 1e-6, 1 - 1e-6)
        y = np.array([x[3] for x in ps], dtype=float)
        ranks = np.array([x[1] for x in ps])
        out[m] = {"log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                  "brier": float(((p - y) ** 2).mean()),
                  **{f"top{n}": float(y[ranks <= n].mean()) for n in (10, 25, 50)}}
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Fit a Poisson regression and LightGBM to backtest.py's training data.")
    ap.add_argument("csv", help="training data from: py backtest.py --season ... --csv PATH")
    ap.add_argument("--out", default="model.json", help="where to save the fitted model (default: model.json)")
    ap.add_argument("--no-lightgbm", action="store_true", help="only fit the regression")
    args = ap.parse_args()

    device_name = f"CPU (NumPy {np.__version__})"
    t0 = time.perf_counter()
    rows = load(args.csv)
    seasons = sorted({int(r["season"]) for r in rows})
    if len(seasons) < 3:
        sys.exit(f"Need at least 3 seasons (train / validate / test); the file has {seasons}.")
    train_s, valid_s, test_s = seasons[:-2], seasons[-2], seasons[-1]
    train = [r for r in rows if int(r["season"]) in train_s]
    valid = [r for r in rows if int(r["season"]) == valid_s]
    test = [r for r in rows if int(r["season"]) == test_s]
    print(f"Loaded {len(rows)} player-games in {time.perf_counter() - t0:.1f}s: train {train_s} {len(train)}, "
          f"validate {valid_s} {len(valid)}, test {test_s} {len(test)}\n")

    # 1. Pick the penalty strength on the validation season.
    print(f"1. Choosing the penalty strength: train on {train_s}, score on {valid_s}")
    Xtr, mean, std = design(train)
    Xva, _, _ = design(valid, mean, std)
    ytr = np.array([r["goals"] for r in train])
    best = None
    for lam in LAMBDAS:
        beta, _, _ = fit(Xtr, ytr, lam, f"lambda {lam:<6}")
        ll = log_loss(predict(beta, Xva), valid)
        print(f"  lambda {lam:<6} -> validation log loss {ll:.5f}\n", flush=True)
        if best is None or ll < best[0]:
            best = (ll, lam, beta)
    _, lam, beta_tr = best
    print(f"Best: lambda {lam}\n")
    valid_models = {"regression": predict(beta_tr, Xva)}

    # 2. Refit on train + validate, then test once on the last season.
    print(f"2. Final regression fit on {train_s + [valid_s]}")
    both = train + valid
    Xall, mean, std = design(both)
    yall = np.array([r["goals"] for r in both])
    beta, steps, fit_secs = fit(Xall, yall, lam, "final       ")
    Xte, _, _ = design(test, mean, std)
    test_models = {"regression": predict(beta, Xte)}

    # 3. LightGBM: choose tree settings on the validation season, then refit on train + validate.
    lgb_info = None
    if not args.no_lightgbm:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            print("\n3. LightGBM skipped: it isn't installed (py -m pip install lightgbm).")
            args.no_lightgbm = True
    if not args.no_lightgbm:
        import lightgbm as lgb
        lgb_dev_name = f"CPU (LightGBM {lgb.__version__})"
        print(f"\n3. LightGBM on {lgb_dev_name}: trying {len(LGB_GRID)} tree settings, train on {train_s}, "
              f"stop early on {valid_s}")
        # Trees don't need the intercept column; the standardized inputs work as they are.
        Ltr, Lva, Lall, Lte = Xtr[:, 1:], Xva[:, 1:], Xall[:, 1:], Xte[:, 1:]
        yva = np.array([r["goals"] for r in valid])
        best_lgb = None
        for params in LGB_GRID:
            label = f"leaves {params['num_leaves']:<3} min leaf {params['min_data_in_leaf']:<4}"
            booster, secs = lgb_fit(Ltr, ytr, params, LGB_MAX_ROUNDS, (Lva, yva), label)
            ll = log_loss(lgb_predict(booster, Lva), valid)
            print(f"  {label} -> {booster.best_iteration} trees, validation log loss {ll:.5f}  ({secs:.1f}s)\n",
                  flush=True)
            if best_lgb is None or ll < best_lgb[0]:
                best_lgb = (ll, params, booster)
        _, lgb_params, lgb_tr = best_lgb
        trees = lgb_tr.best_iteration
        print(f"Best: {lgb_params} with {trees} trees\n")
        valid_models["lightgbm"] = lgb_predict(lgb_tr, Lva)
        print(f"   Final LightGBM fit on {train_s + [valid_s]} ({trees} trees)")
        lgb_final, lgb_secs = lgb_fit(Lall, yall, lgb_params, trees)
        test_models["lightgbm"] = lgb_predict(lgb_final, Lte)
        lgb_final.save_model("model_lightgbm.txt")
        gain = lgb_final.feature_importance(importance_type="gain")
        lgb_info = {"params": {**LGB_BASE, **lgb_params}, "trees": trees, "seconds": lgb_secs, "device": lgb_dev_name,
                    "importance": sorted(((n, float(g / gain.sum())) for (n, _), g in zip(FEATURES, gain)),
                                         key=lambda ng: -ng[1])}

    valid_preds, valid_days = predictions(valid, valid_models)
    test_preds, test_days = predictions(test, test_models)
    print(f"\n=== Validation season {valid_s} (models trained on {train_s} only) ===")
    report(valid_preds, valid_days)
    print(f"\n=== Test season {test_s} (never seen while choosing or fitting) ===")
    report(test_preds, test_days)

    # 4. The models and how they compare, side by side.
    scores = {"validation": summary(valid_preds), "test": summary(test_preds)}
    names = list(valid_preds)
    short = {"v1": "v1", "v2": "v2", "regression": "Regression", "lightgbm": "LightGBM"}
    print("\n" + "=" * 78)
    print("MODELS")
    print(f"  Regression  Poisson regression (log link), L2 penalty lambda = {lam}, {len(FEATURES)} inputs + intercept")
    print(f"              {steps} Newton steps, {fit_secs:.2f}s on {device_name}")
    if lgb_info:
        print(f"  LightGBM    Gradient-boosted trees, Poisson objective, {lgb_info['trees']} trees, "
              f"{lgb_params['num_leaves']} leaves, min {lgb_params['min_data_in_leaf']} players per leaf")
        print(f"              learning rate {LGB_BASE['learning_rate']}, {lgb_info['seconds']:.2f}s on {lgb_info['device']}")
    print(f"  Trained on  {train_s + [valid_s]}: {len(both)} player-games")
    print(f"  Tested on   {test_s}: {len(test)} player-games over {test_days} days")
    for season, key in (("Validation " + str(valid_s), "validation"), ("Test " + str(test_s), "test")):
        print(f"\n{season.upper()} (lower log loss / Brier is better; higher top-N is better)")
        print(f"  {'':<16}" + "".join(f"{short[m]:>13}" for m in names))
        for metric, label, fmt in [("log_loss", "Log loss", "{:.4f}"), ("brier", "Brier", "{:.4f}"),
                                   ("top10", "Top 10 scored", "{:.1%}"), ("top25", "Top 25 scored", "{:.1%}"),
                                   ("top50", "Top 50 scored", "{:.1%}")]:
            vals = [scores[key][m][metric] for m in names]
            best = min(vals) if metric in ("log_loss", "brier") else max(vals)
            print(f"  {label:<16}" + "".join(f"{fmt.format(v) + (' *' if v == best else '  '):>13}" for v in vals))
    print("  * = best in that row")

    # Luck test on every out-of-sample day (validation + test), for the newest model against the rest.
    combined = {m: valid_preds[m] + test_preds[m] for m in names}
    challenger = "lightgbm" if lgb_info else "regression"
    print(f"\nLUCK TEST: {short[challenger]}'s extra scorers vs each model, {valid_days + test_days} days, 95% range")
    for top in (10, 25):
        for other in [m for m in names if m != challenger]:
            d, lo, hi = luck_test(combined, challenger, other, top)
            verdict = "real difference" if lo > 0 or hi < 0 else "could be luck"
            print(f"  Top {top:<3} vs {short[other]:<11}{d:>+5}   ({lo:+d} to {hi:+d})  {verdict}")

    print("\nREGRESSION WEIGHTS (standardized inputs; + raises xG, - lowers it)")
    biggest = max(abs(b) for b in beta[1:]) or 1
    for (name, _), b in sorted(zip(FEATURES, beta[1:]), key=lambda fb: -abs(fb[1])):
        print(f"  {name:<24}{b:+.3f}  {'#' * round(abs(b) / biggest * 30)}")
    if lgb_info:
        print("\nLIGHTGBM IMPORTANCE (share of the improvement each input's splits produced)")
        top_share = lgb_info["importance"][0][1] or 1
        for name, share in lgb_info["importance"]:
            print(f"  {name:<24}{share:6.1%}  {'#' * round(share / top_share * 30)}")
    print("=" * 78)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"type": "poisson_regression", "features": [n for n, _ in FEATURES], "mean": mean.tolist(),
                   "std": std.tolist(), "intercept": float(beta[0]), "weights": [float(b) for b in beta[1:]],
                   "lambda": lam, "trained_on": train_s + [valid_s], "player_games": len(both),
                   "device": device_name, "fit_seconds": round(fit_secs, 3), "scores": scores,
                   "lightgbm": lgb_info and {**lgb_info, "file": "model_lightgbm.txt"}}, f, indent=1)
    saved = f"{args.out}" + (" and model_lightgbm.txt" if lgb_info else "")
    print(f"Saved {saved} (total {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
