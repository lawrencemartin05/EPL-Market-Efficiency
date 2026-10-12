# Results summary

Test seasons: 2023-24, 2024-25, 2025-26, 2026-27 (1190 matches, never used for tuning). Average bookmaker margin: 4.67%.

## Accuracy (lower is better)

|                       |   Brier |   Log loss |    RPS |
|:----------------------|--------:|-----------:|-------:|
| Model (Poisson)       |  0.5919 |     0.9932 | 0.2031 |
| Market (proportional) |  0.5727 |     0.9637 | 0.1943 |
| Market (power method) |  0.5728 |     0.9636 | 0.1943 |
| Blend (0.00 model)    |  0.5727 |     0.9637 | 0.1943 |
| Base rates only       |  0.6514 |     1.0757 | 0.2323 |

Model minus market log loss: +0.0295 (95% CI +0.0178 to +0.0405). So the market was more accurate; the interval excludes zero.
Blend weight on model fitted on validation: 0.00 (0 would mean the market already knows everything the model knows).
Draws: model average 23.0%, market 23.6%, actual 24.8%.

## Betting strategy (rule fixed in advance)

960 bets, £9,600 staked, profit £-1,677.40, ROI -17.47% (95% CI -28.89% to -5.04%), hit rate 20.5% at average odds 5.49, max drawdown £1,729.

Sensitivity to the edge threshold (for transparency; the headline uses the pre-set one):

|      |   bets |    roi |   roi_ci_low |   roi_ci_high |
|-----:|-------:|-------:|-------------:|--------------:|
| 0    |   1128 | -0.163 |       -0.268 |        -0.057 |
| 0.02 |   1062 | -0.168 |       -0.274 |        -0.047 |
| 0.05 |    960 | -0.175 |       -0.291 |        -0.048 |
| 0.1  |    833 | -0.159 |       -0.288 |        -0.021 |
| 0.15 |    704 | -0.135 |       -0.286 |         0.02  |

## Why the value bets lose

Spread of home-win probabilities: model sd 0.145, market sd 0.190. Calibration slope (home win, logit; 1 = perfect, above 1 = underconfident): model 1.29, market 1.10.
Favourites: market says 54.0%, model says 49.4%, they actually won 54.6%.
On the bets placed: model said 31.5%, market said 24.7% (1/odds), actual win rate 20.5%.

## Robustness: Pinnacle closing odds (where available)

970 test matches, Pinnacle margin 2.96%. Log loss: model 0.9776, Pinnacle 0.9448, market average 0.9445. Model minus Pinnacle +0.0328 (95% CI +0.0203 to +0.0456).
Strategy at Pinnacle odds: 847 bets, ROI -15.82% (95% CI -28.52% to -2.65%).

## By test season (headline rule)

|         |   bets |    roi |   roi_ci_low |   roi_ci_high |
|:--------|-------:|-------:|-------------:|--------------:|
| 2023-24 |    318 | -0.325 |       -0.524 |        -0.11  |
| 2024-25 |    312 | -0.022 |       -0.236 |         0.225 |
| 2025-26 |    293 | -0.183 |       -0.374 |         0.011 |
| 2026-27 |     37 | -0.107 |       -0.655 |         0.505 |

## What the strategy bet on

|    |   bets |   avg_odds |    roi |
|:---|-------:|-----------:|-------:|
| H  |    394 |      4.353 | -0.24  |
| D  |      1 |      7.25  | -1     |
| A  |    565 |      6.287 | -0.128 |

## Favourite-longshot check (backing every outcome, all matches, no model involved)

|       |   outcomes |   implied_win_rate |   actual_win_rate |    roi |   roi_ci_low |   roi_ci_high |
|:------|-----------:|-------------------:|------------------:|-------:|-------------:|--------------:|
| 1-1.5 |        439 |              0.765 |             0.743 | -0.032 |       -0.087 |         0.022 |
| 1.5-2 |        727 |              0.581 |             0.563 | -0.032 |       -0.094 |         0.03  |
| 2-3   |       1071 |              0.413 |             0.401 | -0.035 |       -0.092 |         0.024 |
| 3-5   |       2542 |              0.268 |             0.256 | -0.043 |       -0.099 |         0.013 |
| 5-10  |        862 |              0.159 |             0.144 | -0.111 |       -0.253 |         0.035 |
| 10+   |        209 |              0.075 |             0.053 | -0.326 |       -0.699 |         0.111 |

Longshots (odds 5+) minus favourites (odds under 2), ROI: -12.1% (95% CI -27.9% to +4.2%; match-level bootstrap).

## Current team ratings (log scale; higher = better)

|                |   attack |   defence |   overall |
|:---------------|---------:|----------:|----------:|
| Man City       |    0.329 |     0.175 |     0.504 |
| Arsenal        |    0.125 |     0.298 |     0.424 |
| Brighton       |    0.3   |     0.111 |     0.411 |
| Brentford      |    0.113 |     0.103 |     0.216 |
| Man United     |    0.179 |     0.02  |     0.199 |
| Liverpool      |    0.09  |     0.099 |     0.189 |
| Leeds          |   -0.011 |     0.178 |     0.167 |
| Hull           |   -0.071 |     0.162 |     0.091 |
| Bournemouth    |    0.027 |     0.037 |     0.064 |
| Everton        |   -0.073 |     0.111 |     0.038 |
| Nott'm Forest  |   -0.059 |     0.095 |     0.036 |
| Newcastle      |    0.118 |    -0.119 |    -0.001 |
| Chelsea        |    0.137 |    -0.183 |    -0.046 |
| Aston Villa    |    0.014 |    -0.071 |    -0.057 |
| Fulham         |   -0.147 |     0.016 |    -0.131 |
| Sunderland     |   -0.048 |    -0.091 |    -0.139 |
| Ipswich        |    0.001 |    -0.216 |    -0.215 |
| Crystal Palace |   -0.083 |    -0.155 |    -0.237 |
| Tottenham      |   -0.225 |    -0.081 |    -0.306 |
| Coventry       |   -0.281 |    -0.103 |    -0.384 |

Home advantage: home teams score 1.20x as many goals, all else equal.
Chosen time decay: xi=0.0075 (half-life 92 days), ridge=5.0.
