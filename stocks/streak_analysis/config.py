"""
Configuration models for the streak analysis system.
"""

from typing import Optional, Literal, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
import yaml
import json
from pathlib import Path
import pandas as pd


class EvaluationIntervals(BaseModel):
    """Configuration for intervaled evaluation."""
    n_days: int = Field(default=365, description="Max lookback horizon")
    m_days: int = Field(default=90, description="Interval chunk length")
    
    @validator('m_days')
    def validate_m_days(cls, v, values):
        if 'n_days' in values and v > values['n_days']:
            raise ValueError('m_days cannot be greater than n_days')
        return v


class StreakConfig(BaseModel):
    """Main configuration for streak analysis."""
    
    # Required parameters
    symbol: str = Field(..., description="Stock symbol to analyze")
    
    # Data source parameters
    timeframe: Literal["realtime", "hourly", "daily"] = Field(
        default="daily", 
        description="Data timeframe"
    )
    
    # Time parameters
    lookback_days: Optional[int] = Field(
        default=90, 
        description="Number of days to look back"
    )
    date_range: Optional[tuple[date, date]] = Field(
        default=None,
        description="Start and end dates (overrides lookback_days)"
    )
    data_horizon_years: int = Field(
        default=5,
        description="Data horizon for hourly/daily data"
    )
    realtime_window_days: int = Field(
        default=7,
        description="Window for minute-level data"
    )
    
    # Analysis parameters
    min_streak_threshold: int = Field(
        default=0,
        description="Minimum consecutive bars to count as a streak"
    )
    aggregation_level: Literal["day", "week", "month"] = Field(
        default="day",
        description="Evaluation granularity"
    )
    evaluation_mode: Literal["close_to_close", "open_to_close", "hlc3"] = Field(
        default="close_to_close",
        description="Price calculation method"
    )
    
    # Intervaled evaluation
    evaluation_intervals: EvaluationIntervals = Field(
        default_factory=EvaluationIntervals,
        description="Intervaled evaluation configuration"
    )
    
    # Database connection
    db_host: str = Field(default="localhost", description="Database server host")
    db_port: int = Field(default=9002, description="Database server port")
    
    # Output settings
    export_csv: Optional[str] = Field(
        default=None,
        description="Path to export CSV files"
    )
    ascii_only: bool = Field(
        default=False,
        description="Force ASCII-only output"
    )
    no_plots: bool = Field(
        default=False,
        description="Skip generating plots"
    )
    
    # Random seed for reproducibility
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducible results"
    )
    
    @validator('date_range')
    def validate_date_range(cls, v):
        if v is not None:
            start, end = v
            if start >= end:
                raise ValueError('Start date must be before end date')
        return v
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'StreakConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'StreakConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        """Get the effective date range for data fetching."""
        if self.date_range is not None:
            start, end = self.date_range
            return (
                datetime.combine(start, datetime.min.time()),
                datetime.combine(end, datetime.max.time())
            )
        
        end = datetime.now()
        start = end - pd.Timedelta(days=self.lookback_days)
        return start, end


# Example configurations
EXAMPLE_CONFIGS = {
    "daily_5y": {
        "symbol": "TQQQ",
        "timeframe": "daily",
        "data_horizon_years": 5,
        "aggregation_level": "day",
        "evaluation_mode": "close_to_close"
    },
    "hourly_1y": {
        "symbol": "SPY",
        "timeframe": "hourly", 
        "data_horizon_years": 1,
        "aggregation_level": "day",
        "evaluation_mode": "close_to_close"
    },
    "realtime_7d": {
        "symbol": "QQQ",
        "timeframe": "realtime",
        "realtime_window_days": 7,
        "aggregation_level": "day",
        "evaluation_mode": "close_to_close"
    }
}
