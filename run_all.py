"""
Run the whole project:  python run_all.py

Data split (by season, in date order):
  season 1        burn-in: training data only, never scored
  season 2        validation: used to choose the model's hyperparameters
  seasons 3..N    test: the only seasons any headline number comes from
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from epl import plots
from epl.backtest import favourite_longshot, place_bets, summarise
from epl.data import load_matches
from epl.evaluate import (calibration_slope, calibration_table, fit_blend_weight, log_loss_per_match,
                          outcome_index, paired_bootstrap, score_table)
from epl.odds import overround, remove_margin_power, remove_margin_proportional
from epl.walkforward import final_ratings, walk_forward

MODEL_COLS = ["p_model_h", "p_model_d", "p_model_a"]
MARKET_COLS = ["p_mkt_h", "p_mkt_d", "p_mkt_a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw", help="folder with the season CSVs")
    ap.add_argument("--out", default="outputs", help="where results and charts go")
    ap.add_argument("--quick", action="store_true", help="tiny tuning grid, for a fast check")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- stage 1
    matches, report = load_matches(args.data, C.MARKET_ODDS)
    seasons = sorted(matches.season.unique())
    print(f"Loaded {report['matches']} matches from {len(report['files'])} files")
    for s, n in report["matches_per_season"].items():
        print(f"  {s}: {n} matches")
    print(f"  odds used: {report['odds_sources']}")
    if report["incomplete_seasons"]:
        print(f"  NOTE: incomplete seasons {report['incomplete_seasons']} (fine if it's the current one)")
    if len(seasons) < 3:
        raise SystemExit("Need at least 3 seasons: burn-in, validation, and test.")
    burn_in, validation, test_seasons = seasons[0], seasons[1], seasons[2:]
    print(f"Split -> burn-in {burn_in} | validation {validation} | test {', '.join(test_seasons)}")

    # ---------------------------------------------------------------- stage 2
    odds = matches[["odds_h", "odds_d", "odds_a"]].to_numpy()
    matches["overround"] = overround(odds)
    matches[MARKET_COLS] = remove_margin_proportional(odds)
    matches[["p_pow_h", "p_pow_d", "p_pow_a"]] = remove_margin_power(odds)
    print(f"Average bookmaker margin: {matches.overround.mean():.2%}")

    # ---------------------------------------------------------------- stage 3
    xi_grid = [0.0, 0.003] if args.quick else C.XI_GRID
    ridge_grid = [5.0] if args.quick else C.RIDGE_GRID
    tuning = []
    print("Tuning on validation season (by log loss, NOT by profit)...")
    for xi in xi_grid:
        for ridge in ridge_grid:
            v = walk_forward(matches, [validation], xi, ridge, C.MAX_GOALS)
            ll = log_loss_per_match(v[MODEL_COLS].to_numpy(), outcome_index(v.result)).mean()
            tuning.append({"xi": xi, "ridge": ridge, "val_log_loss": ll})
            print(f"  xi={xi:<7} ridge={ridge:<5} log loss={ll:.4f}")
    tuning = pd.DataFrame(tuning).sort_values("val_log_loss")
    best = tuning.iloc[0]
    xi, ridge = float(best.xi), float(best.ridge)
    print(f"Chosen: xi={xi}, ridge={ridge}")
    if (len(ridge_grid) > 1 and ridge in (min(ridge_grid), max(ridge_grid))) or \
       (len(xi_grid) > 1 and xi == max(xi_grid)):
        print("  NOTE: best value is at the edge of the grid; consider widening it in config.py "
              "(and say so in the README, since that is a choice made after seeing validation data).")

    preds = walk_forward(matches, [validation] + test_seasons, xi, ridge, C.MAX_GOALS)
    # (walk_forward keeps every column of the matches table, so market probs come along)
    val, test = preds[preds.season == validation], preds[preds.season.isin(test_seasons)].copy()

    # ---------------------------------------------------------------- stage 4a
    y = outcome_index(test.result)
    P_model, P_mkt = test[MODEL_COLS].to_numpy(), test[MARKET_COLS].to_numpy()
    w = fit_blend_weight(val[MODEL_COLS].to_numpy(), val[MARKET_COLS].to_numpy(),
                         outcome_index(val.result))
    train_rates = matches[matches.season.isin([burn_in, validation])].result.value_counts(normalize=True)
    base = np.tile([train_rates.get(k, 0) for k in "HDA"], (len(test), 1))
    scores = score_table({
        "Model (Poisson)": P_model,
        "Market (proportional)": P_mkt,
        "Market (power method)": test[["p_pow_h", "p_pow_d", "p_pow_a"]].to_numpy(),
        f"Blend ({w:.2f} model)": w * P_model + (1 - w) * P_mkt,
        "Base rates only": base,
    }, y)
    diff = paired_bootstrap(log_loss_per_match(P_model, y), log_loss_per_match(P_mkt, y),
                            C.BOOTSTRAP_SAMPLES, C.SEED)
    cal_model, cal_mkt = calibration_table(P_model, y), calibration_table(P_mkt, y)
    draw_check = {"model_avg_draw_prob": float(P_model[:, 1].mean()),
                  "market_avg_draw_prob": float(P_mkt[:, 1].mean()),
                  "actual_draw_rate": float((y == 1).mean())}

    # ---------------------------------------------------------------- stage 4b
    bets = place_bets(test, C.EDGE_THRESHOLD, C.STAKE)
    headline = summarise(bets, C.STAKE, C.BOOTSTRAP_SAMPLES, C.SEED)
    has_best = test[["best_h", "best_d", "best_a"]].notna().all(axis=1).mean() > 0.9
    best_bets = place_bets(test, C.EDGE_THRESHOLD, C.STAKE, ["best_h", "best_d", "best_a"]) \
        if has_best else bets.iloc[0:0]
    sensitivity = pd.DataFrame(
        {e: summarise(place_bets(test, e, C.STAKE), C.STAKE, 2000, C.SEED) for e in C.SENSITIVITY_EDGES}).T
    by_season = {s: summarise(place_bets(test[test.season == s], C.EDGE_THRESHOLD, C.STAKE), C.STAKE, 2000)
                 for s in test_seasons}
    # Market-pricing question: no model involved, so every match with odds can be used
    fl, fl_gap = favourite_longshot(matches, C.ODDS_BANDS, C.BOOTSTRAP_SAMPLES, C.SEED)
    bets_by_type = bets.groupby("bet_on").apply(lambda b: pd.Series({
        "bets": len(b), "avg_odds": b.odds.mean(), "roi": b.profit.sum() / (C.STAKE * len(b))}),
        include_groups=False).reindex(["H", "D", "A"]).dropna()
    ratings, home_adv = final_ratings(matches, xi, ridge)

    # Diagnosis: why do the "value" bets lose? Compare how spread-out and how
    # well-calibrated the two sets of probabilities are.
    fav_idx = P_mkt.argmax(axis=1)
    rows = np.arange(len(test))
    diagnosis = {
        "sd_home_prob_model": float(P_model[:, 0].std()),
        "sd_home_prob_market": float(P_mkt[:, 0].std()),
        "favourite_prob_market": float(P_mkt[rows, fav_idx].mean()),
        "favourite_prob_model": float(P_model[rows, fav_idx].mean()),
        "favourite_win_rate": float((y == fav_idx).mean()),
        "calibration_slope_model": calibration_slope(P_model[:, 0], (y == 0).astype(int)),
        "calibration_slope_market": calibration_slope(P_mkt[:, 0], (y == 0).astype(int)),
        "bets_model_prob": float(bets.model_prob.mean()) if len(bets) else None,
        "bets_market_prob": float((1 / bets.odds).mean()) if len(bets) else None,
        "bets_hit_rate": float(bets.won.mean()) if len(bets) else None,
    }

    # Robustness: the sharpest book (Pinnacle), on the test matches where it exists
    pin_cols = ["pin_h", "pin_d", "pin_a"]
    tp = test[test[pin_cols].notna().all(axis=1)].copy()
    pinnacle = None
    if len(tp) >= 100:
        yp = outcome_index(tp.result)
        P_pin = remove_margin_proportional(tp[pin_cols].to_numpy())
        P_mod_p = tp[MODEL_COLS].to_numpy()
        pin_bets = place_bets(tp, C.EDGE_THRESHOLD, C.STAKE, pin_cols)
        pinnacle = {
            "matches": int(len(tp)),
            "avg_margin": float(overround(tp[pin_cols].to_numpy()).mean()),
            "log_loss_model": float(log_loss_per_match(P_mod_p, yp).mean()),
            "log_loss_pinnacle": float(log_loss_per_match(P_pin, yp).mean()),
            "log_loss_avg_market_same_matches": float(log_loss_per_match(tp[MARKET_COLS].to_numpy(), yp).mean()),
            "model_minus_pinnacle": paired_bootstrap(log_loss_per_match(P_mod_p, yp),
                                                     log_loss_per_match(P_pin, yp), C.BOOTSTRAP_SAMPLES, C.SEED),
            "strategy_at_pinnacle_odds": summarise(pin_bets, C.STAKE, C.BOOTSTRAP_SAMPLES, C.SEED),
        }

    # ---------------------------------------------------------------- stage 5
    plots.profit_chart(bets, best_bets, C.STAKE, C.EDGE_THRESHOLD, out / "profit.png")
    plots.calibration_chart(cal_model, cal_mkt, out / "calibration.png")
    plots.longshot_chart(fl, out / "favourite_longshot.png")
    test.to_csv(out / "test_predictions.csv", index=False)
    bets.to_csv(out / "bets.csv", index=False)

    results = {
        "data": report, "split": {"burn_in": burn_in, "validation": validation, "test": test_seasons},
        "test_matches": int(len(test)), "avg_margin_test": float(test.overround.mean()),
        "chosen": {"xi": xi, "ridge": ridge, "half_life_days": (np.log(2) / xi) if xi else None},
        "tuning": tuning.to_dict("records"),
        "scores": scores.to_dict("index"), "model_minus_market_log_loss": diff,
        "blend_weight_on_model": w, "draws": draw_check,
        "strategy": headline, "strategy_best_price": summarise(best_bets, C.STAKE) if has_best else None,
        "strategy_by_season": by_season,
        "sensitivity": sensitivity.to_dict("index"), "favourite_longshot": fl.to_dict("index"), "longshot_gap": fl_gap,
        "bets_by_type": bets_by_type.to_dict("index"),
        "home_advantage_goals_multiplier": float(np.exp(home_adv)),
        "pinnacle_check": pinnacle, "diagnosis": diagnosis,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    summary = build_summary(results, scores, sensitivity, fl, ratings, cal_model)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {out.resolve()}")


def build_summary(r, scores, sens, fl, ratings, cal_model):
    s, d = r["strategy"], r["model_minus_market_log_loss"]
    better = "the market" if d["mean_diff"] > 0 else "my model"
    sig = "excludes" if (d["ci_low"] > 0 or d["ci_high"] < 0) else "includes"
    lines = [
        "# Results summary",
        "",
        f"Test seasons: {', '.join(r['split']['test'])} ({r['test_matches']} matches, never used for tuning). "
        f"Average bookmaker margin: {r['avg_margin_test']:.2%}.",
        "",
        "## Accuracy (lower is better)",
        "",
        scores.round(4).to_markdown(),
        "",
        f"Model minus market log loss: {d['mean_diff']:+.4f} (95% CI {d['ci_low']:+.4f} to {d['ci_high']:+.4f}). "
        f"So {better} was more accurate; the interval {sig} zero.",
        f"Blend weight on model fitted on validation: {r['blend_weight_on_model']:.2f} "
        "(0 would mean the market already knows everything the model knows).",
        f"Draws: model average {r['draws']['model_avg_draw_prob']:.1%}, market "
        f"{r['draws']['market_avg_draw_prob']:.1%}, actual {r['draws']['actual_draw_rate']:.1%}.",
        "",
        "## Betting strategy (rule fixed in advance)",
        "",
    ]
    if s["bets"]:
        lines += [
            f"{s['bets']} bets, £{s['staked']:,.0f} staked, profit £{s['profit']:,.2f}, "
            f"ROI {s['roi']:+.2%} (95% CI {s['roi_ci_low']:+.2%} to {s['roi_ci_high']:+.2%}), "
            f"hit rate {s['hit_rate']:.1%} at average odds {s['avg_odds']:.2f}, "
            f"max drawdown £{s['max_drawdown']:,.0f}.",
            "",
            "Sensitivity to the edge threshold (for transparency; the headline uses the pre-set one):",
            "",
            sens[["bets", "roi", "roi_ci_low", "roi_ci_high"]].round(3).to_markdown(),
        ]
    else:
        lines.append("No bets met the threshold.")
    g = r["diagnosis"]
    if g["bets_model_prob"] is not None:
        lines += [
            "",
            "## Why the value bets lose",
            "",
            f"Spread of home-win probabilities: model sd {g['sd_home_prob_model']:.3f}, market sd {g['sd_home_prob_market']:.3f}. "
            f"Calibration slope (home win, logit; 1 = perfect, above 1 = underconfident): model {g['calibration_slope_model']:.2f}, "
            f"market {g['calibration_slope_market']:.2f}.",
            f"Favourites: market says {g['favourite_prob_market']:.1%}, model says {g['favourite_prob_model']:.1%}, "
            f"they actually won {g['favourite_win_rate']:.1%}.",
            f"On the bets placed: model said {g['bets_model_prob']:.1%}, market said {g['bets_market_prob']:.1%} (1/odds), "
            f"actual win rate {g['bets_hit_rate']:.1%}.",
        ]
    p = r.get("pinnacle_check")
    if p:
        d2, sp = p["model_minus_pinnacle"], p["strategy_at_pinnacle_odds"]
        lines += [
            "",
            "## Robustness: Pinnacle closing odds (where available)",
            "",
            f"{p['matches']} test matches, Pinnacle margin {p['avg_margin']:.2%}. Log loss: model {p['log_loss_model']:.4f}, "
            f"Pinnacle {p['log_loss_pinnacle']:.4f}, market average {p['log_loss_avg_market_same_matches']:.4f}. "
            f"Model minus Pinnacle {d2['mean_diff']:+.4f} (95% CI {d2['ci_low']:+.4f} to {d2['ci_high']:+.4f}).",
            (f"Strategy at Pinnacle odds: {sp['bets']} bets, ROI {sp['roi']:+.2%} "
             f"(95% CI {sp['roi_ci_low']:+.2%} to {sp['roi_ci_high']:+.2%})." if sp.get("bets") else "No bets."),
        ]
    lines += [
        "",
        "## By test season (headline rule)",
        "",
        pd.DataFrame(r["strategy_by_season"]).T[["bets", "roi", "roi_ci_low", "roi_ci_high"]].round(3).to_markdown(),
        "",
        "## What the strategy bet on",
        "",
        pd.DataFrame(r["bets_by_type"]).T.round(3).to_markdown(),
        "",
        "## Favourite-longshot check (backing every outcome, all matches, no model involved)",
        "",
        fl.round(3).to_markdown(),
        "",
        f"Longshots (odds 5+) minus favourites (odds under 2), ROI: {r['longshot_gap']['longshot_minus_favourite_roi']:+.1%} "
        f"(95% CI {r['longshot_gap']['ci_low']:+.1%} to {r['longshot_gap']['ci_high']:+.1%}; match-level bootstrap).",
        "",
        "## Current team ratings (log scale; higher = better)",
        "",
        ratings.round(3).to_markdown(),
        "",
        f"Home advantage: home teams score {r['home_advantage_goals_multiplier']:.2f}x as many goals, all else equal.",
        f"Chosen time decay: xi={r['chosen']['xi']}"
        + (f" (half-life {r['chosen']['half_life_days']:.0f} days)" if r['chosen']['half_life_days'] else " (no decay)")
        + f", ridge={r['chosen']['ridge']}.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
