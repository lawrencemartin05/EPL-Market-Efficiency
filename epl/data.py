"""
Stage 1: load football-data.co.uk CSVs into one clean table.

Every season file is called E0.csv on the website, so we never trust file names.
The season is worked out from each match date instead (August to May).
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Which odds to treat as "the market", in order of preference.
# Closing odds (the last price before kick-off) are used first because they contain
# the most information: every bet placed during the week has already moved them.
# Pinnacle is preferred because it is a low-margin bookmaker that welcomes sharp
# bettors, so its prices are widely treated as the most efficient.
ODDS_SOURCES = [
    ("PSC", "Pinnacle closing"),
    ("AvgC", "Market-average closing"),
    ("B365C", "Bet365 closing"),
    ("PS", "Pinnacle pre-closing"),
    ("Avg", "Market-average pre-closing"),
    ("BbAv", "BetBrain average (older files)"),
    ("B365", "Bet365 pre-closing"),
]
# Best price across all bookmakers. Used only as an optimistic sensitivity check,
# because in practice you could not reliably get the best price on every bet.
BEST_PRICE_SOURCES = ["MaxC", "Max", "BbMx"]


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="warn")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


def _parse_dates(s: pd.Series) -> pd.Series:
    """football-data uses dd/mm/yyyy in recent files and dd/mm/yy in older ones."""
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = d.isna()
    d[missing] = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
    return d


def season_label(date: pd.Timestamp) -> str:
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _pick_odds(df: pd.DataFrame, sources) -> tuple[np.ndarray, np.ndarray]:
    """For each row, take H/D/A odds from the first source with three valid prices."""
    odds = np.full((len(df), 3), np.nan)
    label = np.full(len(df), None, dtype=object)
    for prefix, name in sources:
        trio = np.column_stack([_numeric(df, prefix + x).to_numpy() for x in "HDA"])
        valid = np.all(trio > 1.0, axis=1) & np.isnan(odds[:, 0])
        odds[valid] = trio[valid]
        label[valid] = name
    return odds, label


def load_matches(folder: str | Path, primary: str | None = None) -> tuple[pd.DataFrame, dict]:
    """primary: CSV prefix to use first for the market odds (e.g. "AvgC"); the rest of
    ODDS_SOURCES is the fallback chain."""
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {folder.resolve()}. Download E0.csv for each season from "
            "football-data.co.uk/englandm.php and put them in this folder."
        )

    frames = []
    for f in files:
        raw = _read_csv(f)
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        raw["_file"] = f.name
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)

    df = pd.DataFrame({
        "date": _parse_dates(raw["Date"]),
        "home": raw["HomeTeam"].astype(str).str.strip(),
        "away": raw["AwayTeam"].astype(str).str.strip(),
        "hg": pd.to_numeric(raw["FTHG"], errors="coerce"),
        "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
        "file": raw["_file"],
    })
    sources = ODDS_SOURCES
    if primary:
        sources = [s for s in ODDS_SOURCES if s[0] == primary] + [s for s in ODDS_SOURCES if s[0] != primary]
    odds, source = _pick_odds(raw, sources)
    pin, _ = _pick_odds(raw, [("PSC", "Pinnacle closing")])
    best, _ = _pick_odds(raw, [(p, p) for p in BEST_PRICE_SOURCES])
    df[["odds_h", "odds_d", "odds_a"]] = odds
    df["odds_source"] = source
    df[["best_h", "best_d", "best_a"]] = best
    df[["pin_h", "pin_d", "pin_a"]] = pin          # robustness check only (often missing)

    n_before = len(df)
    df = df.dropna(subset=["date", "hg", "ag", "odds_h"])
    dropped_invalid = n_before - len(df)

    # The same season downloaded twice ("E0 (1).csv") would double-count matches.
    n_before = len(df)
    df = df.drop_duplicates(subset=["date", "home", "away"])
    dropped_duplicates = n_before - len(df)

    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    df["season"] = df["date"].map(season_label)
    df = df.sort_values(["date", "home"]).reset_index(drop=True)

    per_season = df.groupby("season").size()
    report = {
        "files": [f.name for f in files],
        "matches": int(len(df)),
        "matches_per_season": per_season.to_dict(),
        "incomplete_seasons": [s for s, n in per_season.items() if n < 380],
        "odds_sources": df["odds_source"].value_counts().to_dict(),
        "dropped_invalid_rows": int(dropped_invalid),
        "dropped_duplicate_rows": int(dropped_duplicates),
    }
    return df, report
