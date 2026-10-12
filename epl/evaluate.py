"""
Stage 4a: how accurate are the probabilities?

Three proper scoring rules. "Proper" means you get the best expected score only by
reporting your true belief, so none of them can be gamed by over- or under-confidence.

* Brier score: squared error between the probability vector and what happened.
* Log loss: -log(probability you gave the actual result). Punishes confident misses
  hard. This is the one used to tune the model.
* RPS (ranked probability score): like Brier but respects the ordering H > D > A, so
  calling a home win when it was a draw is a smaller error than calling an away win.
  It is the standard metric in the football forecasting literature.

Lower is better for all three.
"""
import numpy as np
import pandas as pd

RESULT_INDEX = {"H": 0, "D": 1, "A": 2}


def outcome_index(results) -> np.ndarray:
    return np.array([RESULT_INDEX[r] for r in results])


def one_hot(y: np.ndarray) -> np.ndarray:
    o = np.zeros((len(y), 3))
    o[np.arange(len(y)), y] = 1
    return o


def brier_per_match(P, y):
    return np.sum((P - one_hot(y)) ** 2, axis=1)


def log_loss_per_match(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))


def rps_per_match(P, y):
    cp, co = np.cumsum(P, axis=1)[:, :2], np.cumsum(one_hot(y), axis=1)[:, :2]
    return np.sum((cp - co) ** 2, axis=1) / 2


METRICS = {"Brier": brier_per_match, "Log loss": log_loss_per_match, "RPS": rps_per_match}


def score_table(forecasts: dict, y) -> pd.DataFrame:
    rows = {name: {m: f(P, y).mean() for m, f in METRICS.items()} for name, P in forecasts.items()}
    return pd.DataFrame(rows).T


def paired_bootstrap(loss_a, loss_b, n=10_000, seed=42):
    """Is A really better than B, or could the gap be luck?

    Resamples matches (keeping each match's pair of losses together) and returns the
    mean difference A - B with a 95% interval. If the interval excludes 0, the
    difference is unlikely to be noise.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return {"mean_diff": d.mean(), "ci_low": np.percentile(boots, 2.5),
            "ci_high": np.percentile(boots, 97.5), "share_a_better": (boots < 0).mean()}


def calibration_table(P, y, bins=10) -> pd.DataFrame:
    """Pool every (match, outcome) probability, bucket them, compare with reality.

    A calibrated forecaster's 30% calls happen about 30% of the time.
    """
    p = P.ravel()
    hit = one_hot(y).ravel()
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    t = pd.DataFrame({"bin": b, "p": p, "hit": hit}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("hit", "mean"), n=("p", "size"))
    return t[t.n >= 10]


def fit_blend_weight(P_model, P_market, y) -> float:
    """Best weight w for  w * model + (1 - w) * market,  by log loss.

    If the market already contains everything the model knows, w comes out at 0.
    A positive w (fitted on validation, checked on test) means the model adds
    information the market lacks. This is a forecast-encompassing test.
    """
    grid = np.linspace(0, 1, 101)
    losses = [log_loss_per_match(w * P_model + (1 - w) * P_market, y).mean() for w in grid]
    return float(grid[int(np.argmin(losses))])


def calibration_slope(p, hit) -> float:
    """Fit  P(hit) = logistic(a + b * logit(p)).  b = 1 is perfect.

    b > 1: the forecasts are too timid (real outcomes are more extreme than predicted).
    b < 1: the forecasts are overconfident.
    """
    from scipy.optimize import minimize
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    hit = np.asarray(hit, dtype=float)

    def nll(c):
        z = c[0] + c[1] * x
        return np.sum(np.logaddexp(0, z) - hit * z)

    return float(minimize(nll, [0.0, 1.0]).x[1])
