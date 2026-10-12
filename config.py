"""
Every modelling and betting choice lives here, fixed BEFORE looking at test results.

Why this file matters in an interview: if you tune the betting threshold until the
backtest makes money, you have overfitted. Writing the rules down first is the
equivalent of a pre-registered study. Change these only for a stated reason, and
never because a test-set number looked bad.
"""

# --- Which odds count as "the market" -----------------------------------------
# Decided before any real-data results were seen. football-data.co.uk stopped
# publishing Pinnacle odds partway through 2025-26, so a Pinnacle-first rule would
# silently switch benchmark (and margin) in the middle of the test set. The primary
# benchmark is therefore the market-average closing price, available for every match.
# Pinnacle closing is reported separately as a robustness check where it exists.
MARKET_ODDS = "AvgC"      # prefix in the CSV: AvgCH / AvgCD / AvgCA

# --- Betting rules (decided in advance) ---------------------------------------
STAKE = 10.0              # flat stake in GBP per bet
EDGE_THRESHOLD = 0.05     # bet only if expected value > +5% (model_prob * odds - 1)
# Reported for transparency only. The headline result always uses EDGE_THRESHOLD.
SENSITIVITY_EDGES = [0.0, 0.02, 0.05, 0.10, 0.15]

# --- Model hyperparameters (tuned on the VALIDATION season only) --------------
# xi = time-decay per day. Weight of a match played d days ago = exp(-xi * d).
# 0.0 = all history counts equally; 0.003 = half-life of about 231 days.
XI_GRID = [0.0, 0.0015, 0.003, 0.005, 0.0075, 0.01]   # widened after the first real run hit the 0.005 edge
# Ridge penalty: pulls team ratings towards a prior. Stops wild ratings for teams
# with little data (e.g. promoted teams in August).
RIDGE_GRID = [1.0, 5.0, 20.0, 50.0]

MAX_GOALS = 10            # score grid 0-10 goals each; P(>10 goals) is negligible

# --- Statistics ----------------------------------------------------------------
BOOTSTRAP_SAMPLES = 10_000
SEED = 42

# Odds bands for the favourite-longshot analysis (decimal odds, lower edge inclusive)
ODDS_BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1000.0]
