"""
tools.py - Tool functions that the LLM can call.

These functions provide the AI with access to data and statistical analysis.
Each function returns structured data that the model uses to formulate responses.
"""

import json
from typing import List, Dict, Any, Optional
import pandas as pd

from src.data import load_data, get_team_matches, get_available_teams, get_available_metrics
from src.stats import (
    calculate_stats_with_outlier_analysis,
    compare_metrics,
    detect_outliers_combined,
    format_stats_summary_swedish
)


# Global data cache
_data_cache: Optional[pd.DataFrame] = None


def get_data() -> pd.DataFrame:
    """Get cached data or load it."""
    global _data_cache
    if _data_cache is None:
        _data_cache = load_data()
    return _data_cache


def get_team_summary(team: str, n: int = 10, metrics: List[str] = None) -> Dict[str, Any]:
    """
    Get statistical summary for a team's last N matches.
    
    This function is called by the AI to get comprehensive statistics including:
    - Mean, median, trimmed mean for each metric
    - Outlier detection results
    - Specific outlier matches
    - How mean changes when outliers are removed
    
    Args:
        team: Team name (e.g., "Arsenal", "Liverpool")
        n: Number of recent matches to analyze (default 10)
        metrics: List of metrics to analyze (default: all available)
    
    Returns:
        Dictionary with team summary statistics
    """
    df = get_data()
    
    # Validate team
    available_teams = get_available_teams()
    if team not in available_teams:
        # Try case-insensitive match
        team_lower = team.lower()
        matches = [t for t in available_teams if t.lower() == team_lower]
        if matches:
            team = matches[0]
        else:
            return {
                "error": f"Laget '{team}' hittades inte. Tillgängliga lag: {', '.join(available_teams)}"
            }
    
    # Get team matches
    team_df = get_team_matches(df, team, n=n, latest=True)
    
    if len(team_df) == 0:
        return {"error": f"Inga matcher hittades för {team}"}
    
    # Default to all metrics
    if metrics is None:
        metrics = get_available_metrics()
    
    # Validate metrics
    available_metrics = get_available_metrics()
    metrics = [m for m in metrics if m in available_metrics]
    
    if not metrics:
        return {"error": f"Inga giltiga metrics. Tillgängliga: {', '.join(available_metrics)}"}
    
    # Calculate statistics for each metric
    results = {
        "team": team,
        "num_matches": len(team_df),
        "date_range": {
            "from": str(team_df["date"].min())[:10],
            "to": str(team_df["date"].max())[:10]
        },
        "metrics": {}
    }
    
    for metric in metrics:
        results["metrics"][metric] = calculate_stats_with_outlier_analysis(team_df, metric)
    
    # Add match list for context
    results["matches"] = team_df[["date", "opponent", "home_away"] + metrics].to_dict("records")
    for match in results["matches"]:
        match["date"] = str(match["date"])[:10]
    
    return results


def compare_teams(team_a: str, team_b: str, n: int = 10, 
                  metrics: List[str] = None) -> Dict[str, Any]:
    """
    Compare statistics between two teams for their last N matches.
    
    This function helps analyze matchups by comparing:
    - All key statistics for both teams
    - Outlier patterns
    - Which team has more consistent/variable performance
    
    Args:
        team_a: First team name
        team_b: Second team name
        n: Number of recent matches to analyze
        metrics: List of metrics to compare
    
    Returns:
        Dictionary with comparison results
    """
    df = get_data()
    available_teams = get_available_teams()
    
    # Validate and normalize team names
    def normalize_team(team):
        if team in available_teams:
            return team
        team_lower = team.lower()
        matches = [t for t in available_teams if t.lower() == team_lower]
        return matches[0] if matches else None
    
    team_a = normalize_team(team_a)
    team_b = normalize_team(team_b)
    
    if not team_a:
        return {"error": f"Lag A hittades inte. Tillgängliga lag: {', '.join(available_teams)}"}
    if not team_b:
        return {"error": f"Lag B hittades inte. Tillgängliga lag: {', '.join(available_teams)}"}
    
    # Get data for both teams
    df_a = get_team_matches(df, team_a, n=n, latest=True)
    df_b = get_team_matches(df, team_b, n=n, latest=True)
    
    # Default to all metrics
    if metrics is None:
        metrics = get_available_metrics()
    
    # Compare each metric
    comparisons = {}
    for metric in metrics:
        comparisons[metric] = compare_metrics(df_a, df_b, team_a, team_b, metric)
    
    # Summary insights
    insights = generate_comparison_insights(comparisons, team_a, team_b)
    
    return {
        "team_a": team_a,
        "team_b": team_b,
        "num_matches": n,
        "comparisons": comparisons,
        "insights": insights
    }


def get_outlier_matches(team: str, n: int = 10, metric: str = "throw_ins",
                        method: str = "both") -> Dict[str, Any]:
    """
    Get detailed information about outlier matches for a team.
    
    Args:
        team: Team name
        n: Number of recent matches to analyze
        metric: The metric to analyze for outliers
        method: Detection method - "iqr", "mad", or "both"
    
    Returns:
        Dictionary with outlier match details
    """
    df = get_data()
    available_teams = get_available_teams()
    
    # Normalize team name
    if team not in available_teams:
        team_lower = team.lower()
        matches = [t for t in available_teams if t.lower() == team_lower]
        if matches:
            team = matches[0]
        else:
            return {"error": f"Laget '{team}' hittades inte."}
    
    # Validate metric
    available_metrics = get_available_metrics()
    if metric not in available_metrics:
        return {"error": f"Metric '{metric}' finns inte. Tillgängliga: {', '.join(available_metrics)}"}
    
    # Get team matches
    team_df = get_team_matches(df, team, n=n, latest=True)
    
    if len(team_df) == 0:
        return {"error": f"Inga matcher hittades för {team}"}
    
    # Get full analysis
    analysis = calculate_stats_with_outlier_analysis(team_df, metric)
    
    # Detect outliers with specified method
    data = team_df[metric].values
    
    if method == "iqr":
        from src.stats import detect_outliers_iqr
        outliers = detect_outliers_iqr(data)
        method_desc = "IQR-metoden (1.5 × interkvartilområdet)"
    elif method == "mad":
        from src.stats import detect_outliers_mad
        outliers = detect_outliers_mad(data)
        method_desc = "MAD-metoden (robust z-score > 3.5)"
    else:
        outliers = detect_outliers_combined(data)
        method_desc = "Kombinerad IQR + MAD"
    
    # Build detailed outlier list
    outlier_details = []
    for idx in range(len(team_df)):
        if outliers[idx]:
            row = team_df.iloc[idx]
            outlier_details.append({
                "date": str(row["date"])[:10],
                "opponent": row["opponent"],
                "home_away": row["home_away"],
                "value": int(row[metric]),
                "deviation_from_median": float(row[metric] - analysis["stats"]["median"]),
                "is_high": row[metric] > analysis["stats"]["median"]
            })
    
    return {
        "team": team,
        "metric": metric,
        "method": method_desc,
        "num_matches_analyzed": len(team_df),
        "stats": analysis["stats"],
        "num_outliers": len(outlier_details),
        "outlier_matches": outlier_details,
        "mean_impact": analysis["outlier_analysis"]["mean_difference"],
        "mean_without_outliers": analysis["outlier_analysis"]["mean_without_outliers"]
    }


def get_available_teams_list() -> Dict[str, List[str]]:
    """
    Get list of all available teams.
    
    Returns:
        Dictionary with list of team names
    """
    return {
        "teams": get_available_teams(),
        "count": len(get_available_teams())
    }


def generate_comparison_insights(comparisons: Dict, team_a: str, team_b: str) -> List[str]:
    """Generate Swedish insights from comparison data."""
    insights = []
    
    metric_names = {
        "throw_ins": "inkast",
        "fouls": "frisparkar",
        "shots": "skott"
    }
    
    for metric, comp in comparisons.items():
        metric_name = metric_names.get(metric, metric)
        diff = comp["differences"]
        
        # Check for significant outlier impact
        a_outliers = comp["team_a"]["outliers"]["num_combined_outliers"]
        b_outliers = comp["team_b"]["outliers"]["num_combined_outliers"]
        
        if a_outliers > 0 or b_outliers > 0:
            insights.append(
                f"För {metric_name}: {team_a} har {a_outliers} outliers, "
                f"{team_b} har {b_outliers} outliers. "
                f"Medelvärdet kan vara missvisande!"
            )
        
        # Check if mean vs median differ
        if abs(diff["mean"] - diff["median"]) > 2:
            insights.append(
                f"Stor skillnad mellan medel ({diff['mean']:.1f}) och median ({diff['median']:.1f}) "
                f"för {metric_name} - outliers påverkar jämförelsen."
            )
    
    return insights


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_team_summary",
            "description": "Hämta statistiksammanfattning för ett lags senaste N matcher. Inkluderar medelvärde, median, trimmat medelvärde, outlier-analys och specifika outlier-matcher. Använd denna för att analysera ett enskilt lags prestationer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Lagets namn, t.ex. 'Arsenal', 'Liverpool', 'Manchester City'"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Antal senaste matcher att analysera (5-38)",
                        "default": 10
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista med metrics att analysera: 'throw_ins', 'fouls', 'shots'. Utelämna för alla.",
                        "default": ["throw_ins", "fouls", "shots"]
                    }
                },
                "required": ["team"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_teams",
            "description": "Jämför statistik mellan två lag för deras senaste N matcher. Visar skillnader i medelvärde, median och trimmat medelvärde, samt outlier-analys för båda lagen. Perfekt för matchup-analys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_a": {
                        "type": "string",
                        "description": "Första lagets namn"
                    },
                    "team_b": {
                        "type": "string",
                        "description": "Andra lagets namn"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Antal senaste matcher att analysera för varje lag",
                        "default": 10
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista med metrics att jämföra",
                        "default": ["throw_ins", "fouls", "shots"]
                    }
                },
                "required": ["team_a", "team_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_outlier_matches",
            "description": "Hämta detaljerad information om outlier-matcher för ett lag. Visar exakt vilka matcher som är outliers, hur mycket de avviker, och hur de påverkar medelvärdet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Lagets namn"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Antal senaste matcher att analysera",
                        "default": 10
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["throw_ins", "fouls", "shots"],
                        "description": "Vilken metric att analysera för outliers",
                        "default": "throw_ins"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["iqr", "mad", "both"],
                        "description": "Outlier-detektionsmetod: 'iqr', 'mad', eller 'both'",
                        "default": "both"
                    }
                },
                "required": ["team"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_teams_list",
            "description": "Hämta lista över alla tillgängliga lag i databasen.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# Map function names to actual functions
TOOL_FUNCTIONS = {
    "get_team_summary": get_team_summary,
    "compare_teams": compare_teams,
    "get_outlier_matches": get_outlier_matches,
    "get_available_teams_list": get_available_teams_list
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool function and return JSON result.
    
    Args:
        name: Tool function name
        arguments: Arguments to pass to the function
    
    Returns:
        JSON string with the result
    """
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Okänt verktyg: {name}"})
    
    try:
        func = TOOL_FUNCTIONS[name]
        result = func(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"Fel vid körning av {name}: {str(e)}"})
