"""
Stage 2: turn bookmaker odds into probabilities.

Decimal odds of 2.00 pay back 2 for every 1 staked, so the "fair" probability is 1/2.
Summed over home/draw/away, 1/odds comes to more than 1 (e.g. 1.05). The extra is
the overround: the bookmaker's built-in margin, like an insurer's expense and profit
loading on top of the pure risk premium.
"""
import numpy as np
from scipy.optimize import brentq


def implied_probs(odds) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds) -> np.ndarray:
    """Margin as a fraction, e.g. 0.05 means the book adds up to 105%."""
    return implied_probs(odds).sum(axis=-1) - 1.0


def remove_margin_proportional(odds) -> np.ndarray:
    """Scale the three implied probabilities down so they sum to exactly 1.

    Assumes the bookmaker spreads the margin evenly across outcomes in proportion to
    their probability. Simple and standard, but see the power method below.
    """
    q = implied_probs(odds)
    return q / q.sum(axis=-1, keepdims=True)


def remove_margin_power(odds) -> np.ndarray:
    """Find k so that sum(q_i ** k) = 1, and use p_i = q_i ** k.

    Raising to a power k > 1 shrinks small probabilities proportionally more than
    large ones. That loads more of the margin onto longshots, which matches the
    well-documented favourite-longshot bias. Used as a robustness check.
    """
    q = np.atleast_2d(implied_probs(odds))
    out = np.empty_like(q)
    for i, row in enumerate(q):
        if row.sum() <= 1.0:          # no margin (or an arbitrage): nothing to remove
            out[i] = row / row.sum()
            continue
        k = brentq(lambda k: np.sum(row ** k) - 1.0, 1.0, 10.0)
        out[i] = row ** k
    return out
