"""
Stage 3: the Poisson team-ratings model.

Each team has an attack rating and a defence rating. For a match, home vs away:

    expected home goals  = exp(mu + home_adv + attack[home] - defence[away])
    expected away goals  = exp(mu            + attack[away] - defence[home])

mu is the baseline log scoring rate, home_adv is the home-field boost. Ratings are on
a log scale, so +0.1 attack means about 10% more goals against an average defence.

Why Poisson? Goals are rare, roughly independent events spread across 90 minutes,
with a fairly stable rate. That is exactly the setting in which counts follow a
Poisson distribution (the same reasoning actuaries use for claim counts).

Fitting is by maximum likelihood with two additions:
  * time decay: recent matches count more (weight = exp(-xi * days_ago));
  * a ridge penalty pulling each rating towards a prior, so teams with little data
    don't get extreme ratings. This is a credibility-weighting idea: a rating is a
    blend of the team's own evidence and a prior, weighted by how much evidence
    there is.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


class PoissonRatings:
    def __init__(self, xi: float = 0.003, ridge: float = 5.0, max_goals: int = 10):
        self.xi = xi
        self.ridge = ridge
        self.max_goals = max_goals

    # ------------------------------------------------------------------ fitting
    def fit(self, home, away, hg, ag, days_ago, teams=None, priors=None):
        """
        home, away: team names; hg, ag: goals; days_ago: days before the prediction
        date (must all be > 0, i.e. strictly in the past).
        teams: extra teams to include (e.g. promoted teams with no history yet).
        priors: {team: (attack_prior, defence_prior)}; teams not listed get (0, 0).
        """
        home = np.asarray(home)
        away = np.asarray(away)
        hg = np.asarray(hg, dtype=float)
        ag = np.asarray(ag, dtype=float)
        days_ago = np.asarray(days_ago, dtype=float)
        if np.any(days_ago <= 0):
            raise ValueError("Training data must be strictly before the prediction date.")

        names = sorted(set(home) | set(away) | set(teams or []))
        index = {t: i for i, t in enumerate(names)}
        n = len(names)
        hi = np.array([index[t] for t in home])
        ai = np.array([index[t] for t in away])

        priors = priors or {}
        att0 = np.array([priors.get(t, (0.0, 0.0))[0] for t in names])
        def0 = np.array([priors.get(t, (0.0, 0.0))[1] for t in names])
        w = np.exp(-self.xi * days_ago)

        def objective(theta):
            mu, h = theta[0], theta[1]
            att, dfn = theta[2:2 + n], theta[2 + n:]
            eta_h = mu + h + att[hi] - dfn[ai]
            eta_a = mu + att[ai] - dfn[hi]
            lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)
            # Negative Poisson log-likelihood (dropping log(y!), which has no parameters)
            nll = np.sum(w * (lam_h - hg * eta_h)) + np.sum(w * (lam_a - ag * eta_a))
            nll += self.ridge * (np.sum((att - att0) ** 2) + np.sum((dfn - def0) ** 2))

            # Analytic gradient: d nll / d eta = w * (lambda - goals)
            r_h, r_a = w * (lam_h - hg), w * (lam_a - ag)
            g_att, g_def = np.zeros(n), np.zeros(n)
            np.add.at(g_att, hi, r_h)
            np.add.at(g_att, ai, r_a)
            np.add.at(g_def, ai, -r_h)
            np.add.at(g_def, hi, -r_a)
            g_att += 2 * self.ridge * (att - att0)
            g_def += 2 * self.ridge * (dfn - def0)
            grad = np.concatenate([[r_h.sum() + r_a.sum(), r_h.sum()], g_att, g_def])
            return nll, grad

        start = np.concatenate([[np.log(max(hg.mean(), 0.1)), 0.2], att0, def0])
        res = minimize(objective, start, jac=True, method="L-BFGS-B")
        if not res.success and "ABNORMAL" not in str(res.message):
            raise RuntimeError(f"Model fit failed: {res.message}")

        self.teams_ = names
        self.index_ = index
        self.mu_, self.home_adv_ = res.x[0], res.x[1]
        self.attack_ = dict(zip(names, res.x[2:2 + n]))
        self.defence_ = dict(zip(names, res.x[2 + n:]))
        return self

    # --------------------------------------------------------------- predicting
    def expected_goals(self, home, away):
        att = np.array([self.attack_[t] for t in home]), np.array([self.attack_[t] for t in away])
        dfn = np.array([self.defence_[t] for t in home]), np.array([self.defence_[t] for t in away])
        lam_h = np.exp(self.mu_ + self.home_adv_ + att[0] - dfn[1])
        lam_a = np.exp(self.mu_ + att[1] - dfn[0])
        return lam_h, lam_a

    def predict_proba(self, home, away):
        lam_h, lam_a = self.expected_goals(home, away)
        return outcome_probs(lam_h, lam_a, self.max_goals), lam_h, lam_a

    def ratings_table(self):
        import pandas as pd
        t = pd.DataFrame({"attack": self.attack_, "defence": self.defence_})
        t["overall"] = t["attack"] + t["defence"]
        return t.sort_values("overall", ascending=False)


def outcome_probs(lam_h, lam_a, max_goals: int = 10) -> np.ndarray:
    """Home/draw/away probabilities from two independent Poisson goal counts.

    Builds the full score grid P(home = i, away = j) = P(i) * P(j), then adds up the
    cells below the diagonal (home win), on it (draw), and above it (away win).
    """
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], np.asarray(lam_h)[:, None])
    pa = poisson.pmf(goals[None, :], np.asarray(lam_a)[:, None])
    grid = ph[:, :, None] * pa[:, None, :]          # [match, home goals, away goals]
    home_win = np.tril(np.ones((max_goals + 1,) * 2), -1).astype(bool)
    p_h = grid[:, home_win].sum(axis=1)
    p_d = np.einsum("nii->n", grid)
    p_a = grid[:, home_win.T].sum(axis=1)
    p = np.column_stack([p_h, p_d, p_a])
    return p / p.sum(axis=1, keepdims=True)         # tiny correction for the 10-goal cap
