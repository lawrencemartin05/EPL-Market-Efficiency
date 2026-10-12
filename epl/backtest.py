"""
Stage 4b: would the model have made money?

Rule (fixed in config.py before running): for each match, compute expected value
EV = model_probability * decimal_odds - 1 for home, draw and away. If the best EV is
above the threshold, stake a flat amount on that outcome. At most one bet per match.

"Value" is defined by expected value, not by "my probability is higher than the
market's", because a 2-point gap on a 1.30 favourite is worth far less than a
2-point gap on a 6.00 outsider.
"""
import numpy as np
import pandas as pd

from .evaluate import outcome_index

ODDS_COLS = ["odds_h", "odds_d", "odds_a"]
MODEL_COLS = ["p_model_h", "p_model_d", "p_model_a"]
OUTCOMES = np.array(["H", "D", "A"])


def place_bets(df: pd.DataFrame, edge: float, stake: float, odds_cols=ODDS_COLS) -> pd.DataFrame:
    P = df[MODEL_COLS].to_numpy()
    O = df[odds_cols].to_numpy()
    ev = P * O - 1
    pick = ev.argmax(axis=1)
    rows = np.arange(len(df))
    take = (ev[rows, pick] > edge) & np.isfinite(O[rows, pick])
    y = outcome_index(df["result"])

    bets = df.loc[take, ["date", "season", "home", "away", "result"]].copy()
    bets["bet_on"] = OUTCOMES[pick[take]]
    bets["odds"] = O[rows, pick][take]
    bets["model_prob"] = P[rows, pick][take]
    bets["ev"] = ev[rows, pick][take]
    bets["won"] = pick[take] == y[take]
    bets["profit"] = np.where(bets["won"], stake * (bets["odds"] - 1), -stake)
    bets["cum_profit"] = bets["profit"].cumsum()
    return bets.reset_index(drop=True)


def max_drawdown(cum_profit: np.ndarray) -> float:
    """Largest peak-to-trough fall in the running total (a standard risk measure)."""
    if len(cum_profit) == 0:
        return 0.0
    curve = np.concatenate([[0.0], cum_profit])
    return float((np.maximum.accumulate(curve) - curve).max())


def summarise(bets: pd.DataFrame, stake: float, n_boot=10_000, seed=42) -> dict:
    n = len(bets)
    if n == 0:
        return {"bets": 0}
    profit = bets["profit"].to_numpy()
    roi = profit.sum() / (stake * n)
    # Bootstrap: resample the bets themselves to see how much ROI could vary by luck.
    rng = np.random.default_rng(seed)
    boots = profit[rng.integers(0, n, size=(n_boot, n))].sum(axis=1) / (stake * n)
    return {
        "bets": n,
        "staked": stake * n,
        "profit": float(profit.sum()),
        "roi": float(roi),
        "roi_ci_low": float(np.percentile(boots, 2.5)),
        "roi_ci_high": float(np.percentile(boots, 97.5)),
        "prob_roi_positive": float((boots > 0).mean()),
        "hit_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "max_drawdown": max_drawdown(bets["cum_profit"].to_numpy()),
    }


def favourite_longshot(df: pd.DataFrame, bands, n_boot=10_000, seed=42):
    """ROI from betting 1 unit on EVERY outcome, grouped by odds band.

    If the market prices risk evenly, every band loses about the same (the margin).
    If longshots lose much more, the market overprices unlikely outcomes: the
    favourite-longshot bias, which mirrors people overpaying for lottery-like payoffs.

    Confidence intervals resample whole MATCHES (a cluster bootstrap), because the
    three outcomes of one match are linked: exactly one of them wins.
    Also returns the gap between longshots (odds >= 5) and favourites (odds < 2).
    """
    O = df[ODDS_COLS].to_numpy()
    y = outcome_index(df["result"])
    won = np.arange(3)[None, :] == y[:, None]
    profit = np.where(won, O - 1, -1.0)                       # per match, per outcome
    edges = np.asarray(bands)
    band = np.digitize(O, edges) - 1                          # band index per cell
    nb = len(edges) - 1
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" if edges[i + 1] < 1000 else f"{edges[i]:g}+" for i in range(nb)]

    # per-match sums of profit and counts in each band -> fast cluster bootstrap
    P = np.zeros((len(df), nb))
    N = np.zeros((len(df), nb))
    for b in range(nb):
        m = band == b
        P[:, b] = np.where(m, profit, 0).sum(axis=1)
        N[:, b] = m.sum(axis=1)
    long_ = edges[:-1] >= 5
    fav = edges[1:] <= 2

    rng = np.random.default_rng(seed)
    Pb, Nb = [], []
    for _ in range(0, n_boot, 500):                           # chunks keep memory small
        idx = rng.integers(0, len(df), size=(500, len(df)))
        Pb.append(P[idx].sum(axis=1))
        Nb.append(N[idx].sum(axis=1))
    Pb, Nb = np.vstack(Pb)[:n_boot], np.vstack(Nb)[:n_boot]   # (n_boot, nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        roi_b = Pb / Nb
        gap_b = Pb[:, long_].sum(1) / Nb[:, long_].sum(1) - Pb[:, fav].sum(1) / Nb[:, fav].sum(1)

    implied = np.array([(1 / O[band == b]).mean() if (band == b).any() else np.nan for b in range(nb)])
    actual = np.array([won[band == b].mean() if (band == b).any() else np.nan for b in range(nb)])
    t = pd.DataFrame({
        "outcomes": N.sum(0).astype(int),
        "implied_win_rate": implied,
        "actual_win_rate": actual,
        "roi": P.sum(0) / N.sum(0),
        "roi_ci_low": np.nanpercentile(roi_b, 2.5, axis=0),
        "roi_ci_high": np.nanpercentile(roi_b, 97.5, axis=0),
    }, index=labels)
    t = t[t.outcomes > 0]
    point_gap = P[:, long_].sum() / N[:, long_].sum() - P[:, fav].sum() / N[:, fav].sum()
    gap = {"longshot_minus_favourite_roi": float(point_gap),
           "ci_low": float(np.nanpercentile(gap_b, 2.5)), "ci_high": float(np.nanpercentile(gap_b, 97.5)),
           "share_longshots_worse": float((gap_b < 0).mean())}
    return t, gap
