"""
Make fake seasons in exactly the football-data.co.uk column format.

Used to test the pipeline end to end, and as a sanity check: on fake data we KNOW
the true probabilities, so we can confirm the model and the market both land close
to them and that nothing is leaking.

    python scripts/make_synthetic_data.py --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


def outcome_probs(lh, la, g=10):
    k = np.arange(g + 1)
    grid = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    p = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
    return p / p.sum()


def add_margin(p, margin, rng, noise):
    """Market view = truth plus noise, then a margin loaded mostly onto longshots."""
    q = np.exp(np.log(p) + rng.normal(0, noise, 3))
    q /= q.sum()
    # power method in reverse: find k<1 with sum(q**k) = 1 + margin
    lo, hi = 0.3, 1.0
    for _ in range(60):
        k = (lo + hi) / 2
        lo, hi = (lo, k) if np.sum(q ** k) < 1 + margin else (k, hi)
    return np.round(1 / q ** k, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seasons", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool = [f"Team {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    att = dict(zip(pool, rng.normal(0, 0.25, len(pool))))
    dfn = dict(zip(pool, rng.normal(0, 0.2, len(pool))))
    strength = {t: att[t] + dfn[t] for t in pool}
    league = sorted(pool, key=strength.get, reverse=True)[:20]
    mu, home_adv = 0.15, 0.22

    for s in range(args.seasons):
        year = 2021 + s
        teams = list(league)
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        start = pd.Timestamp(f"{year}-08-14")
        rows = []
        for i, (h, a) in enumerate(fixtures):
            date = start + pd.Timedelta(days=7 * (i // 10) + int(rng.integers(0, 3)))
            for t in (h, a):                          # ratings drift through the season
                att[t] += rng.normal(0, 0.006)
                dfn[t] += rng.normal(0, 0.006)
            lh = np.exp(mu + home_adv + att[h] - dfn[a])
            la = np.exp(mu + att[a] - dfn[h])
            hg, ag = rng.poisson(lh), rng.poisson(la)
            p = outcome_probs(lh, la)
            ps_open = add_margin(p, 0.025, rng, 0.10)
            ps_close = add_margin(p, 0.025, rng, 0.05)   # closing = better informed
            avg_close = add_margin(p, 0.055, rng, 0.06)
            b365_close = add_margin(p, 0.06, rng, 0.06)
            max_close = np.round(avg_close * 1.04, 2)
            row = {"Div": "E0", "Date": date.strftime("%d/%m/%Y"), "Time": "15:00",
                   "HomeTeam": h, "AwayTeam": a, "FTHG": hg, "FTAG": ag,
                   "FTR": "H" if hg > ag else "D" if hg == ag else "A"}
            for pre, o in [("PS", ps_open), ("PSC", ps_close), ("AvgC", avg_close),
                           ("B365C", b365_close), ("MaxC", max_close)]:
                row.update({pre + "H": o[0], pre + "D": o[1], pre + "A": o[2]})
            if s == args.seasons - 1:                  # test the fallback when Pinnacle is missing
                row.update({"PSCH": "", "PSCD": "", "PSCA": ""})
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("Date", key=lambda d: pd.to_datetime(d, dayfirst=True))
        df.to_csv(out / f"E0_{str(year)[-2:]}{str(year + 1)[-2:]}.csv", index=False)

        # relegation: bottom 3 by noisy strength, replaced by best 3 outside
        table = sorted(teams, key=lambda t: att[t] + dfn[t] + rng.normal(0, 0.15))
        down = table[:3]
        outside = sorted([t for t in pool if t not in teams], key=lambda t: att[t] + dfn[t], reverse=True)
        league = [t for t in teams if t not in down] + outside[:3]
        print(f"{year}-{year + 1}: {len(df)} matches, relegated {down}, promoted {outside[:3]}")


if __name__ == "__main__":
    main()
