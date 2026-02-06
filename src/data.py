"""
data.py - Mockdata generation and loading for Premier League statistics.

Generates realistic match statistics with intentional outliers to demonstrate
why mean values can be misleading in statistical analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Premier League teams (2023/24 season)
PREMIER_LEAGUE_TEAMS = [
    "Arsenal",
    "Aston Villa", 
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Luton Town",
    "Manchester City",
    "Manchester United",
    "Newcastle",
    "Nottingham Forest",
    "Sheffield United",
    "Tottenham",
    "West Ham",
    "Wolverhampton",
    "Burnley"
]

# Realistic base statistics ranges for each metric
# Format: (mean, std, min_val, max_val)
STAT_PROFILES = {
    "throw_ins": {
        "base": (22, 5, 12, 35),
        "outlier_high": (42, 5, 38, 50),  # Extreme weather/defensive play
        "outlier_low": (8, 2, 5, 12)       # Very dominant possession
    },
    "fouls": {
        "base": (11, 3, 5, 18),
        "outlier_high": (22, 3, 19, 28),   # Heated derby/aggressive play
        "outlier_low": (3, 1, 1, 5)        # Very clean game
    },
    "shots": {
        "base": (12, 4, 5, 22),
        "outlier_high": (32, 5, 28, 40),   # Attacking masterclass
        "outlier_low": (2, 1, 0, 4)        # Parked the bus / dominated
    }
}

# Team-specific modifiers (some teams have characteristic playing styles)
TEAM_MODIFIERS = {
    "Manchester City": {"throw_ins": -3, "shots": 4, "fouls": -2},
    "Liverpool": {"throw_ins": -2, "shots": 3, "fouls": -1},
    "Arsenal": {"throw_ins": -1, "shots": 2, "fouls": 0},
    "Burnley": {"throw_ins": 4, "fouls": 3, "shots": -2},
    "Sheffield United": {"throw_ins": 3, "fouls": 2, "shots": -1},
    "Luton Town": {"throw_ins": 2, "fouls": 1, "shots": -1},
    "Chelsea": {"throw_ins": 0, "shots": 1, "fouls": 0},
    "Tottenham": {"throw_ins": -1, "shots": 2, "fouls": 1},
    "Newcastle": {"throw_ins": 1, "shots": 1, "fouls": 1},
    "West Ham": {"throw_ins": 2, "fouls": 2, "shots": 0},
}


def generate_match_stat(metric: str, team: str, is_outlier: bool = False, 
                        outlier_type: str = "high") -> int:
    """Generate a single match statistic with optional outlier behavior."""
    profile = STAT_PROFILES[metric]
    
    if is_outlier:
        params = profile[f"outlier_{outlier_type}"]
    else:
        params = profile["base"]
    
    mean, std, min_val, max_val = params
    
    # Apply team modifier
    modifier = TEAM_MODIFIERS.get(team, {}).get(metric, 0)
    mean += modifier
    
    # Generate value with normal distribution, clipped to range
    value = np.random.normal(mean, std)
    value = int(np.clip(value, min_val, max_val))
    
    return value


def generate_mock_data(num_matches_per_team: int = 38, 
                       outlier_probability: float = 0.08,
                       seed: int = 42) -> pd.DataFrame:
    """
    Generate mock Premier League statistics data.
    
    Args:
        num_matches_per_team: Number of matches to generate per team (default 38 = full season)
        outlier_probability: Probability of a match being an outlier (default 8%)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: date, team, opponent, home_away, throw_ins, fouls, shots
    """
    np.random.seed(seed)
    
    records = []
    start_date = datetime(2024, 8, 17)  # Season start
    
    for team in PREMIER_LEAGUE_TEAMS:
        # Get list of opponents (all other teams)
        opponents = [t for t in PREMIER_LEAGUE_TEAMS if t != team]
        
        for match_num in range(num_matches_per_team):
            # Select opponent (cycle through, with some repeats for home/away)
            opponent = opponents[match_num % len(opponents)]
            
            # Determine home/away (alternate, roughly)
            home_away = "Home" if match_num % 2 == 0 else "Away"
            
            # Generate match date (roughly one match per week)
            match_date = start_date + timedelta(days=match_num * 7 + np.random.randint(-2, 3))
            
            # Determine if this match is an outlier
            is_outlier = np.random.random() < outlier_probability
            
            # If outlier, randomly choose which metric(s) and direction
            outlier_metrics = set()
            outlier_directions = {}
            if is_outlier:
                # 70% chance of single metric outlier, 30% chance of multiple
                num_outlier_metrics = 1 if np.random.random() < 0.7 else np.random.randint(2, 4)
                outlier_metrics = set(np.random.choice(
                    ["throw_ins", "fouls", "shots"], 
                    size=min(num_outlier_metrics, 3), 
                    replace=False
                ))
                for metric in outlier_metrics:
                    outlier_directions[metric] = "high" if np.random.random() < 0.7 else "low"
            
            # Generate statistics
            throw_ins = generate_match_stat(
                "throw_ins", team, 
                "throw_ins" in outlier_metrics,
                outlier_directions.get("throw_ins", "high")
            )
            fouls = generate_match_stat(
                "fouls", team,
                "fouls" in outlier_metrics,
                outlier_directions.get("fouls", "high")
            )
            shots = generate_match_stat(
                "shots", team,
                "shots" in outlier_metrics,
                outlier_directions.get("shots", "high")
            )
            
            records.append({
                "date": match_date.strftime("%Y-%m-%d"),
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "throw_ins": throw_ins,
                "fouls": fouls,
                "shots": shots
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    
    return df


def ensure_mock_data_exists(data_path: str = None) -> str:
    """
    Ensure mock data CSV exists, generate if missing.
    
    Returns:
        Path to the CSV file
    """
    if data_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "mock_pl_stats.csv"
    else:
        data_path = Path(data_path)
    
    # Create directory if needed
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"Genererar mockdata till {data_path}...")
        df = generate_mock_data()
        df.to_csv(data_path, index=False)
        print(f"Klar! Genererade {len(df)} matchrader.")
    
    return str(data_path)


def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the mock Premier League statistics data.
    
    Args:
        data_path: Optional path to CSV file. If None, uses default location.
    
    Returns:
        DataFrame with match statistics
    """
    csv_path = ensure_mock_data_exists(data_path)
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def get_team_matches(df: pd.DataFrame, team: str, n: int = None, 
                     latest: bool = True) -> pd.DataFrame:
    """
    Get matches for a specific team.
    
    Args:
        df: Full statistics DataFrame
        team: Team name
        n: Number of matches to return (None = all)
        latest: If True, return latest matches; if False, return earliest
    
    Returns:
        DataFrame filtered to team's matches
    """
    team_df = df[df["team"] == team].copy()
    team_df = team_df.sort_values("date", ascending=not latest)
    
    if n is not None:
        team_df = team_df.head(n)
    
    return team_df.sort_values("date")


def get_available_teams() -> list:
    """Return list of available team names."""
    return PREMIER_LEAGUE_TEAMS.copy()


def get_available_metrics() -> list:
    """Return list of available statistical metrics."""
    return ["throw_ins", "fouls", "shots"]


# Generate data when module is run directly
if __name__ == "__main__":
    print("Genererar Premier League mockdata...")
    df = generate_mock_data()
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "mock_pl_stats.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSparat till: {output_path}")
    print(f"Totalt antal rader: {len(df)}")
    print(f"Antal lag: {df['team'].nunique()}")
    print(f"Matcher per lag: {len(df) // df['team'].nunique()}")
    
    print("\nExempel på data:")
    print(df.head(10).to_string())
    
    print("\nStatistik-sammanfattning:")
    print(df[["throw_ins", "fouls", "shots"]].describe())
