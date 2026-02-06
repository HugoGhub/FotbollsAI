"""
stats.py - Robust statistical measures and outlier detection.

Implements IQR-based and MAD-based (robust z-score) outlier detection,
along with robust measures like median and trimmed mean.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from scipy import stats as scipy_stats


def calculate_iqr_bounds(data: np.ndarray, k: float = 1.5) -> Tuple[float, float]:
    """
    Calculate IQR-based bounds for outlier detection.
    
    Args:
        data: Array of values
        k: IQR multiplier (default 1.5 for standard outliers, 3.0 for extreme)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    return lower_bound, upper_bound


def calculate_mad(data: np.ndarray) -> float:
    """
    Calculate Median Absolute Deviation (MAD).
    
    MAD = median(|x_i - median(x)|)
    
    Args:
        data: Array of values
    
    Returns:
        MAD value
    """
    median = np.median(data)
    return np.median(np.abs(data - median))


def calculate_robust_zscore(data: np.ndarray) -> np.ndarray:
    """
    Calculate robust z-scores using MAD.
    
    Modified z-score = 0.6745 * (x - median) / MAD
    
    The constant 0.6745 makes it comparable to standard z-scores
    for normally distributed data.
    
    Args:
        data: Array of values
    
    Returns:
        Array of robust z-scores
    """
    median = np.median(data)
    mad = calculate_mad(data)
    
    # Avoid division by zero
    if mad == 0:
        return np.zeros_like(data, dtype=float)
    
    # 0.6745 is the 0.75th quantile of the standard normal distribution
    return 0.6745 * (data - median) / mad


def detect_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Detect outliers using IQR method.
    
    Args:
        data: Array of values
        k: IQR multiplier (default 1.5)
    
    Returns:
        Boolean array where True indicates outlier
    """
    lower, upper = calculate_iqr_bounds(data, k)
    return (data < lower) | (data > upper)


def detect_outliers_mad(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Detect outliers using robust z-score (MAD method).
    
    Args:
        data: Array of values
        threshold: Robust z-score threshold (default 3.5, recommended by Iglewicz and Hoaglin)
    
    Returns:
        Boolean array where True indicates outlier
    """
    robust_z = calculate_robust_zscore(data)
    return np.abs(robust_z) > threshold


def detect_outliers_combined(data: np.ndarray, iqr_k: float = 1.5, 
                              mad_threshold: float = 3.5,
                              method: str = "both") -> np.ndarray:
    """
    Detect outliers using combined IQR and MAD methods.
    
    Args:
        data: Array of values
        iqr_k: IQR multiplier
        mad_threshold: MAD z-score threshold
        method: "iqr", "mad", or "both" (union of both methods)
    
    Returns:
        Boolean array where True indicates outlier
    """
    if method == "iqr":
        return detect_outliers_iqr(data, iqr_k)
    elif method == "mad":
        return detect_outliers_mad(data, mad_threshold)
    else:  # both
        iqr_outliers = detect_outliers_iqr(data, iqr_k)
        mad_outliers = detect_outliers_mad(data, mad_threshold)
        return iqr_outliers | mad_outliers


def trimmed_mean(data: np.ndarray, trim_percent: float = 0.1) -> float:
    """
    Calculate trimmed mean (cutting off extreme values from both ends).
    
    Args:
        data: Array of values
        trim_percent: Proportion to trim from each end (default 0.1 = 10%)
    
    Returns:
        Trimmed mean value
    """
    return scipy_stats.trim_mean(data, trim_percent)


def calculate_robust_stats(data: np.ndarray, trim_percent: float = 0.1) -> Dict[str, float]:
    """
    Calculate comprehensive robust statistics.
    
    Args:
        data: Array of values
        trim_percent: Proportion to trim for trimmed mean
    
    Returns:
        Dictionary with mean, median, trimmed_mean, std, mad, min, max, n
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "trimmed_mean": float(trimmed_mean(data, trim_percent)),
        "std": float(np.std(data)),
        "mad": float(calculate_mad(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "n": len(data)
    }


def calculate_stats_with_outlier_analysis(
    df: pd.DataFrame, 
    metric: str,
    iqr_k: float = 1.5,
    mad_threshold: float = 3.5,
    trim_percent: float = 0.1
) -> Dict[str, Any]:
    """
    Calculate statistics with detailed outlier analysis for a metric.
    
    Args:
        df: DataFrame with match data
        metric: Column name for the metric
        iqr_k: IQR multiplier for outlier detection
        mad_threshold: MAD z-score threshold
        trim_percent: Trim proportion for trimmed mean
    
    Returns:
        Dictionary with comprehensive statistics and outlier info
    """
    data = df[metric].values
    
    # Basic stats
    stats = calculate_robust_stats(data, trim_percent)
    
    # Outlier detection
    iqr_outliers = detect_outliers_iqr(data, iqr_k)
    mad_outliers = detect_outliers_mad(data, mad_threshold)
    combined_outliers = iqr_outliers | mad_outliers
    
    # IQR bounds
    lower_bound, upper_bound = calculate_iqr_bounds(data, iqr_k)
    
    # Robust z-scores
    robust_z = calculate_robust_zscore(data)
    
    # Calculate mean without outliers
    non_outlier_data = data[~combined_outliers]
    mean_without_outliers = float(np.mean(non_outlier_data)) if len(non_outlier_data) > 0 else stats["mean"]
    
    # Identify outlier matches
    outlier_indices = np.where(combined_outliers)[0]
    outlier_matches = []
    for idx in outlier_indices:
        row = df.iloc[idx]
        outlier_matches.append({
            "date": str(row["date"])[:10] if hasattr(row["date"], "strftime") else str(row["date"])[:10],
            "opponent": row["opponent"],
            "home_away": row["home_away"],
            "value": int(row[metric]),
            "robust_zscore": float(robust_z[idx]),
            "is_iqr_outlier": bool(iqr_outliers[idx]),
            "is_mad_outlier": bool(mad_outliers[idx])
        })
    
    return {
        "metric": metric,
        "stats": stats,
        "outlier_analysis": {
            "iqr_bounds": {"lower": lower_bound, "upper": upper_bound},
            "num_iqr_outliers": int(np.sum(iqr_outliers)),
            "num_mad_outliers": int(np.sum(mad_outliers)),
            "num_combined_outliers": int(np.sum(combined_outliers)),
            "outlier_matches": outlier_matches,
            "mean_without_outliers": mean_without_outliers,
            "mean_difference": stats["mean"] - mean_without_outliers
        }
    }


def compare_metrics(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    team_a: str,
    team_b: str,
    metric: str,
    trim_percent: float = 0.1
) -> Dict[str, Any]:
    """
    Compare a metric between two teams.
    
    Args:
        df_a: DataFrame for team A
        df_b: DataFrame for team B
        team_a: Name of team A
        team_b: Name of team B
        metric: Metric to compare
        trim_percent: Trim proportion for trimmed mean
    
    Returns:
        Dictionary with comparison results
    """
    stats_a = calculate_stats_with_outlier_analysis(df_a, metric, trim_percent=trim_percent)
    stats_b = calculate_stats_with_outlier_analysis(df_b, metric, trim_percent=trim_percent)
    
    # Calculate differences
    diff_mean = stats_a["stats"]["mean"] - stats_b["stats"]["mean"]
    diff_median = stats_a["stats"]["median"] - stats_b["stats"]["median"]
    diff_trimmed = stats_a["stats"]["trimmed_mean"] - stats_b["stats"]["trimmed_mean"]
    
    return {
        "metric": metric,
        "team_a": {
            "name": team_a,
            "stats": stats_a["stats"],
            "outliers": stats_a["outlier_analysis"]
        },
        "team_b": {
            "name": team_b,
            "stats": stats_b["stats"],
            "outliers": stats_b["outlier_analysis"]
        },
        "differences": {
            "mean": diff_mean,
            "median": diff_median,
            "trimmed_mean": diff_trimmed,
            "interpretation": interpret_difference(diff_mean, diff_median, diff_trimmed, metric)
        }
    }


def interpret_difference(diff_mean: float, diff_median: float, 
                         diff_trimmed: float, metric: str) -> str:
    """
    Generate Swedish interpretation of differences between teams.
    """
    metric_names = {
        "throw_ins": "inkast",
        "fouls": "frisparkar (mot sig)",
        "shots": "skott"
    }
    metric_name = metric_names.get(metric, metric)
    
    # Check if mean and median differ significantly
    if abs(diff_mean - diff_median) > 1:
        return (f"Skillnaden i medelvärde ({diff_mean:.1f}) och median ({diff_median:.1f}) "
                f"för {metric_name} tyder på att outliers påverkar jämförelsen. "
                f"Titta på trimmat medelvärde ({diff_trimmed:.1f}) för en rättvisare bild.")
    else:
        return (f"Medelvärde och median överensstämmer väl för {metric_name}, "
                f"vilket tyder på få extremvärden i denna jämförelse.")


def format_stats_summary_swedish(stats_result: Dict[str, Any]) -> str:
    """
    Format statistics result as Swedish text summary.
    """
    s = stats_result["stats"]
    o = stats_result["outlier_analysis"]
    metric = stats_result["metric"]
    
    metric_names = {
        "throw_ins": "Inkast",
        "fouls": "Frisparkar",
        "shots": "Skott"
    }
    metric_name = metric_names.get(metric, metric)
    
    summary = f"""
**{metric_name}:**
- Medelvärde: {s['mean']:.1f}
- Median: {s['median']:.1f}
- Trimmat medelvärde (10%): {s['trimmed_mean']:.1f}
- Min/Max: {s['min']:.0f} / {s['max']:.0f}
- Antal outliers: {o['num_combined_outliers']} (IQR: {o['num_iqr_outliers']}, MAD: {o['num_mad_outliers']})
"""
    
    if o['num_combined_outliers'] > 0:
        summary += f"- Medelvärde utan outliers: {o['mean_without_outliers']:.1f} "
        summary += f"(skillnad: {o['mean_difference']:+.1f})\n"
        
        if o['outlier_matches']:
            summary += "\n*Outlier-matcher:*\n"
            for match in o['outlier_matches'][:5]:  # Max 5
                summary += f"  - {match['date']}: vs {match['opponent']} ({match['home_away']}): {match['value']} (z={match['robust_zscore']:.1f})\n"
    
    return summary
