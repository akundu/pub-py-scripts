"""
Streak Analysis System

A comprehensive system for analyzing stock price streaks using data from db_server.py.
"""

__version__ = "1.0.0"
__author__ = "MarkDev"

from .data_provider import DbServerProvider
from .preprocess import prepare_data
from .streaks import compute_streak_stats
from .signals import suggest_thresholds
from .evaluation import evaluate_intervals
from .viz import plot_streak_histogram, plot_forward_returns
from .config import StreakConfig

__all__ = [
    "DbServerProvider",
    "prepare_data", 
    "compute_streak_stats",
    "suggest_thresholds",
    "evaluate_intervals",
    "plot_streak_histogram",
    "plot_forward_returns",
    "StreakConfig"
]
