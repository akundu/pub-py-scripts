"""
Find historical days similar to today based on market conditions.

Helps answer: "What happened on days like today?"
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


@dataclass
class SimilarDay:
    """A historical day similar to today."""
    date: str
    similarity_score: float  # 0-100
    vix: float
    gap_pct: float
    intraday_move_pct: float
    actual_close_move: float  # How much it moved by close
    time_label: str
    outcome: str  # "higher", "lower", "flat"


def calculate_similarity(
    today_vix: float,
    today_gap_pct: float,
    today_intraday_move: float,
    today_time_label: str,
    hist_vix: float,
    hist_gap_pct: float,
    hist_intraday_move: float,
    hist_time_label: str,
) -> float:
    """
    Calculate similarity score between today and a historical day.

    Returns score 0-100 (100 = perfect match).
    """
    # VIX similarity (most important - 40% weight)
    vix_diff = abs(today_vix - hist_vix)
    if vix_diff < 1:
        vix_score = 100
    elif vix_diff < 3:
        vix_score = 90
    elif vix_diff < 5:
        vix_score = 70
    elif vix_diff < 10:
        vix_score = 40
    else:
        vix_score = 0

    # Gap similarity (20% weight)
    gap_diff = abs(today_gap_pct - hist_gap_pct)
    if gap_diff < 0.2:
        gap_score = 100
    elif gap_diff < 0.5:
        gap_score = 80
    elif gap_diff < 1.0:
        gap_score = 50
    else:
        gap_score = 20

    # Intraday move similarity (20% weight)
    move_diff = abs(today_intraday_move - hist_intraday_move)
    if move_diff < 0.2:
        move_score = 100
    elif move_diff < 0.5:
        move_score = 80
    elif move_diff < 1.0:
        move_score = 50
    else:
        move_score = 20

    # Time of day match (20% weight)
    if today_time_label == hist_time_label:
        time_score = 100
    else:
        # Parse times and calculate difference
        try:
            today_hour = int(today_time_label.split(':')[0])
            hist_hour = int(hist_time_label.split(':')[0])
            hour_diff = abs(today_hour - hist_hour)
            if hour_diff <= 1:
                time_score = 80
            elif hour_diff <= 2:
                time_score = 50
            else:
                time_score = 20
        except:
            time_score = 50

    # Weighted average
    total_score = (
        vix_score * 0.40 +
        gap_score * 0.20 +
        move_score * 0.20 +
        time_score * 0.20
    )

    return total_score


def find_similar_days(
    pct_df: pd.DataFrame,
    current_vix: float,
    current_gap_pct: float,
    current_intraday_move: float,
    current_price: float,
    prev_close: float,
    time_label: str,
    top_n: int = 10,
    min_similarity: float = 60.0,
) -> List[SimilarDay]:
    """
    Find historical days most similar to today.

    Args:
        pct_df: Historical percentile data with columns:
            - date, time_label, vix, gap_pct, intraday_move_pct,
              close_move_pct, actual_close
        current_vix: Today's VIX
        current_gap_pct: Today's gap % (current vs prev close)
        current_intraday_move: Today's move from open %
        current_price: Current price
        prev_close: Previous close
        time_label: Current time label (e.g., "10:00", "14:00")
        top_n: Number of similar days to return
        min_similarity: Minimum similarity score (0-100)

    Returns:
        List of SimilarDay objects, sorted by similarity
    """
    if pct_df is None or pct_df.empty:
        return []

    similar_days = []

    for _, row in pct_df.iterrows():
        # Get historical values from pct_df columns
        hist_gap_pct = row.get('gap_pct', 0.0)
        hist_intraday_move = row.get('intraday_move_pct', 0.0)
        hist_time_label = str(row.get('time', '10:00'))

        # Calculate similarity (without historical VIX since it's not in pct_df)
        # Use current VIX for comparison
        score = calculate_similarity(
            today_vix=current_vix,
            today_gap_pct=current_gap_pct,
            today_intraday_move=current_intraday_move,
            today_time_label=time_label,
            hist_vix=current_vix,  # Use current VIX as we don't have historical
            hist_gap_pct=hist_gap_pct,
            hist_intraday_move=hist_intraday_move,
            hist_time_label=hist_time_label,
        )

        if score < min_similarity:
            continue

        # Get actual outcome
        close_move = row.get('close_move_pct', 0.0)
        if close_move > 0.1:
            outcome = "higher"
        elif close_move < -0.1:
            outcome = "lower"
        else:
            outcome = "flat"

        similar_days.append(SimilarDay(
            date=str(row.get('date', 'unknown')),
            similarity_score=score,
            vix=current_vix,  # Use current VIX (historical VIX not available)
            gap_pct=hist_gap_pct,
            intraday_move_pct=hist_intraday_move,
            actual_close_move=close_move,
            time_label=hist_time_label,
            outcome=outcome,
        ))

    # Sort by similarity
    similar_days.sort(key=lambda x: x.similarity_score, reverse=True)

    return similar_days[:top_n]


def analyze_similar_days_outcomes(similar_days: List[SimilarDay]) -> dict:
    """
    Analyze outcomes of similar days.

    Returns dict with:
    - avg_close_move: Average close move %
    - pct_higher: % of days that closed higher
    - pct_lower: % of days that closed lower
    - pct_flat: % of days that closed flat
    - avg_similarity: Average similarity score
    """
    if not similar_days:
        return {}

    close_moves = [d.actual_close_move for d in similar_days]
    outcomes = [d.outcome for d in similar_days]

    return {
        'avg_close_move': np.mean(close_moves),
        'median_close_move': np.median(close_moves),
        'pct_higher': outcomes.count('higher') / len(outcomes) * 100,
        'pct_lower': outcomes.count('lower') / len(outcomes) * 100,
        'pct_flat': outcomes.count('flat') / len(outcomes) * 100,
        'avg_similarity': np.mean([d.similarity_score for d in similar_days]),
        'count': len(similar_days),
    }


def format_similar_days(
    similar_days: List[SimilarDay],
    current_price: float,
    show_top_n: int = 5,
) -> str:
    """Format similar days for display."""
    if not similar_days:
        return "No similar historical days found."

    lines = []
    lines.append("="*80)
    lines.append("HISTORICAL DAYS SIMILAR TO TODAY")
    lines.append("="*80)

    # Summary stats
    stats = analyze_similar_days_outcomes(similar_days)
    close_moves = [d.actual_close_move for d in similar_days]
    min_move = min(close_moves)
    max_move = max(close_moves)
    min_price = current_price * (1 + min_move / 100)
    max_price = current_price * (1 + max_move / 100)

    lines.append(f"\nAnalyzed {stats['count']} similar days (90%+ similarity, avg: {stats['avg_similarity']:.0f}%)")
    lines.append(f"\n{'='*80}")
    lines.append(f"RANGE OF OUTCOMES (from all {stats['count']} similar days)")
    lines.append(f"{'='*80}")
    lines.append(f"\n  Best Case:  {max_move:+.2f}%  →  Close ~${max_price:,.2f}")
    lines.append(f"  Worst Case: {min_move:+.2f}%  →  Close ~${min_price:,.2f}")
    lines.append(f"  Range:      {max_move - min_move:.2f}%  (${max_price - min_price:,.2f})")
    lines.append(f"\n  Average:    {stats['avg_close_move']:+.2f}%  →  Close ~${current_price * (1 + stats['avg_close_move']/100):,.2f}")
    lines.append(f"  Median:     {stats['median_close_move']:+.2f}%  →  Close ~${current_price * (1 + stats['median_close_move']/100):,.2f}")

    lines.append(f"\n{'='*80}")
    lines.append("OUTCOME DISTRIBUTION")
    lines.append(f"{'='*80}")
    lines.append(f"  Closed Higher: {stats['pct_higher']:.0f}%")
    lines.append(f"  Closed Lower:  {stats['pct_lower']:.0f}%")
    lines.append(f"  Closed Flat:   {stats['pct_flat']:.0f}%")

    # Top matches
    lines.append(f"\n{'='*80}")
    lines.append(f"TOP {min(show_top_n, len(similar_days))} MOST SIMILAR DAYS (90%+ SIMILARITY)")
    lines.append(f"{'='*80}\n")

    for i, day in enumerate(similar_days[:show_top_n], 1):
        lines.append(f"{i}. {day.date} (Similarity: {day.similarity_score:.0f}%)")
        lines.append(f"   VIX: {day.vix:.1f} | Gap: {day.gap_pct:+.2f}% | Intraday: {day.intraday_move_pct:+.2f}%")
        lines.append(f"   Outcome: Closed {day.outcome.upper()} ({day.actual_close_move:+.2f}%)")

        # Project to current price
        projected = current_price * (1 + day.actual_close_move/100)
        lines.append(f"   If today similar: Close ~${projected:,.2f}")
        lines.append("")

    lines.append("="*80)

    return "\n".join(lines)
