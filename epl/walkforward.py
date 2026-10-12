"""
Walk-forward prediction: the no-look-ahead rule, enforced in code.

For every match date D, the model is refitted using ONLY matches played before D,
then predicts the matches on D. Nothing from D or later can leak in. The test file
tests/test_no_lookahead.py proves this by scrambling all future results and checking
that earlier predictions do not change.

Promoted teams: they arrive with no Premier League history in the data. Starting them
at league-average (rating 0) would systematically overrate them early in the season.
Instead their prior is the average rating of the teams they replaced (the relegated
ones). The list of teams in a season is public before it starts, so using it is not
look-ahead.
"""
import numpy as np
import pandas as pd

from .model import PoissonRatings


def walk_forward(matches: pd.DataFrame, predict_seasons, xi: float, ridge: float,
                 max_goals: int = 10) -> pd.DataFrame:
    m = matches.sort_values("date").reset_index(drop=True)
    season_order = sorted(m["season"].unique())
    season_teams = {s: set(g.home) | set(g.away) for s, g in m.groupby("season")}

    out = []
    for date in sorted(m.loc[m.season.isin(predict_seasons), "date"].unique()):
        date = pd.Timestamp(date)
        train = m[m.date < date]                     # strictly before: the key rule
        today = m[m.date == date]
        if len(train) < 50:
            continue
        season = today.season.iloc[0]
        teams_now = season_teams[season]
        pos = season_order.index(season)
        prev_teams = season_teams[season_order[pos - 1]] if pos > 0 else set()
        promoted = teams_now - prev_teams if prev_teams else set()
        relegated = prev_teams - teams_now

        days_ago = (date - train.date).dt.days.to_numpy()
        args = (train.home, train.away, train.hg, train.ag, days_ago)

        model = PoissonRatings(xi=xi, ridge=ridge, max_goals=max_goals)
        model.fit(*args, teams=teams_now)
        if promoted and relegated:
            rel = [t for t in relegated if t in model.attack_]
            if rel:
                prior = (np.mean([model.attack_[t] for t in rel]),
                         np.mean([model.defence_[t] for t in rel]))
                model.fit(*args, teams=teams_now, priors={t: prior for t in promoted})

        p, lam_h, lam_a = model.predict_proba(today.home.to_numpy(), today.away.to_numpy())
        block = today.copy()
        block[["p_model_h", "p_model_d", "p_model_a"]] = p
        block["xg_home"], block["xg_away"] = lam_h, lam_a
        block["train_matches"] = len(train)
        out.append(block)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def final_ratings(matches: pd.DataFrame, xi: float, ridge: float) -> pd.DataFrame:
    """Ratings as of the day after the last match, for the README."""
    date = matches.date.max() + pd.Timedelta(days=1)
    days = (date - matches.date).dt.days.to_numpy()
    model = PoissonRatings(xi=xi, ridge=ridge).fit(
        matches.home, matches.away, matches.hg, matches.ag, days,
        teams=set(matches[matches.season == matches.season.max()].home))
    last = set(matches[matches.season == matches.season.max()].home)
    return model.ratings_table().loc[lambda t: t.index.isin(last)], model.home_adv_
