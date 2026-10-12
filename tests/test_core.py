"""
Run with:  python -m pytest -q

The most important test is test_no_lookahead: it scrambles every result AFTER a
cut-off date and checks that predictions made up to that date are identical. If the
model peeked at the future, they would change.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epl.backtest import max_drawdown, place_bets  # noqa: E402
from epl.evaluate import brier_per_match, log_loss_per_match, rps_per_match  # noqa: E402
from epl.model import PoissonRatings, outcome_probs  # noqa: E402
from epl.odds import overround, remove_margin_power, remove_margin_proportional  # noqa: E402
from epl.walkforward import walk_forward  # noqa: E402


def fake_league(seasons=3, teams=8, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(teams)]
    att = dict(zip(names, rng.normal(0, 0.3, teams)))
    dfn = dict(zip(names, rng.normal(0, 0.3, teams)))
    rows, day = [], pd.Timestamp("2020-08-15")
    for s in range(seasons):
        day = pd.Timestamp(f"{2020 + s}-08-15")
        for k, (h, a) in enumerate([(h, a) for h in names for a in names if h != a]):
            lh, la = np.exp(0.1 + 0.25 + att[h] - dfn[a]), np.exp(0.1 + att[a] - dfn[h])
            rows.append({"date": day + pd.Timedelta(days=7 * (k // 4)), "home": h, "away": a,
                         "hg": rng.poisson(lh), "ag": rng.poisson(la)})
    df = pd.DataFrame(rows)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    df["season"] = df.date.map(lambda d: f"{d.year if d.month >= 7 else d.year - 1}")
    return df.sort_values("date").reset_index(drop=True), att, dfn


# ------------------------------------------------------------------ odds
def test_margin_removal():
    odds = np.array([[2.00, 3.50, 4.00]])
    assert overround(odds)[0] == pytest.approx(0.5 + 1 / 3.5 + 0.25 - 1)
    for f in (remove_margin_proportional, remove_margin_power):
        p = f(odds)
        assert p.sum() == pytest.approx(1.0)
        assert p[0, 0] > p[0, 2] > 0
    # power method takes proportionally more margin off the longshot
    prop, powr = remove_margin_proportional(odds), remove_margin_power(odds)
    assert powr[0, 2] < prop[0, 2] and powr[0, 0] > prop[0, 0]


# ------------------------------------------------------------------ model
def test_outcome_probs_sum_to_one_and_make_sense():
    p = outcome_probs(np.array([1.5, 1.0]), np.array([1.0, 1.0]))
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 0] > p[0, 2]                 # stronger home side more likely to win
    assert p[1, 0] == pytest.approx(p[1, 2])  # equal rates: symmetric


def test_model_recovers_true_ratings():
    df, att, _ = fake_league(seasons=10, teams=8)
    days = (df.date.max() + pd.Timedelta(days=1) - df.date).dt.days
    m = PoissonRatings(xi=0.0, ridge=0.5).fit(df.home, df.away, df.hg, df.ag, days)
    est = np.array([m.attack_[t] for t in att])
    true = np.array(list(att.values()))
    assert np.corrcoef(est, true)[0, 1] > 0.9
    assert 0.1 < m.home_adv_ < 0.4


def test_fit_refuses_same_day_data():
    with pytest.raises(ValueError):
        PoissonRatings().fit(["A"], ["B"], [1], [0], [0])


def test_no_lookahead():
    df, _, _ = fake_league(seasons=3, teams=8)
    seasons = sorted(df.season.unique())
    val_dates = df[df.season == seasons[1]].date.sort_values()
    cutoff = val_dates.iloc[len(val_dates) // 2]

    before = walk_forward(df, seasons[1:], xi=0.002, ridge=2.0)
    scrambled = df.copy()
    future = scrambled.date > cutoff
    rng = np.random.default_rng(1)
    scrambled.loc[future, "hg"] = rng.integers(0, 8, future.sum())
    scrambled.loc[future, "ag"] = rng.integers(0, 8, future.sum())
    after = walk_forward(scrambled, seasons[1:], xi=0.002, ridge=2.0)

    cols = ["p_model_h", "p_model_d", "p_model_a"]
    early = before.date <= cutoff
    assert early.sum() > 0
    assert np.allclose(before.loc[early, cols].to_numpy(), after.loc[early, cols].to_numpy())
    # and the scramble really did change later predictions, so the test has teeth
    assert not np.allclose(before.loc[~early, cols].to_numpy(), after.loc[~early, cols].to_numpy())


# ------------------------------------------------------------------ scoring & betting
def test_scores_perfect_and_uniform():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    uniform = np.full((3, 3), 1 / 3)
    assert brier_per_match(perfect, y).sum() == pytest.approx(0)
    assert rps_per_match(perfect, y).sum() == pytest.approx(0)
    assert log_loss_per_match(uniform, y).mean() == pytest.approx(np.log(3))


def test_betting_rule_and_profit():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "season": "x",
        "home": ["A", "C"], "away": ["B", "D"], "result": ["H", "A"],
        "p_model_h": [0.60, 0.40], "p_model_d": [0.25, 0.30], "p_model_a": [0.15, 0.30],
        "odds_h": [2.00, 2.40], "odds_d": [3.60, 3.30], "odds_a": [6.00, 3.10],
    })
    bets = place_bets(df, edge=0.05, stake=10)
    # match 1: EV home = 0.6*2-1 = +0.20 -> bet home, wins +10
    # match 2: best EV = 0.30*3.30-1 = -0.01 -> no bet
    assert len(bets) == 1 and bets.bet_on[0] == "H" and bets.profit[0] == pytest.approx(10)
    assert max_drawdown(np.array([10, 0, 5, -20])) == pytest.approx(30)
