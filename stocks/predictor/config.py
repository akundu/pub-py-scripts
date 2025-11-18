"""
Configuration models for the Next-Action and Magnitude Predictor.

Uses Pydantic for validation and type safety.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import numpy as np


class BinningConfig(BaseModel):
    """Configuration for feature binning."""
    
    return_bins: List[float] = Field(
        default=[-np.inf, -0.02, -0.01, -0.0025, 0.0025, 0.01, 0.02, np.inf],
        description="Return bins for discretization"
    )
    vol_bins: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Number of volume quantile bins"
    )
    streak_cap: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum streak length to consider"
    )
    flat_threshold_pct: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.01,
        description="Threshold for classifying 'flat' movements"
    )


class ModelConfig(BaseModel):
    """Configuration for individual models."""
    
    markov: bool = Field(default=True, description="Enable Markov chain model")
    gbdt: bool = Field(default=True, description="Enable GBDT model")
    logistic_quantile: bool = Field(default=True, description="Enable logistic + quantile regression")
    hmm: bool = Field(default=False, description="Enable HMM regime detection")
    
    # Markov-specific config
    markov_order: int = Field(default=1, ge=1, le=3, description="Markov chain order")
    laplace_alpha: float = Field(default=1.0, ge=0.0, description="Laplace smoothing parameter")
    
    # GBDT-specific config
    gbdt_max_depth: int = Field(default=6, ge=3, le=15, description="Maximum tree depth")
    gbdt_n_estimators: int = Field(default=100, ge=50, le=500, description="Number of estimators")
    gbdt_learning_rate: float = Field(default=0.1, ge=0.01, le=0.3, description="Learning rate")
    
    # Logistic regression config
    logit_c: float = Field(default=1.0, ge=0.01, le=100.0, description="Regularization strength")
    logit_max_iter: int = Field(default=1000, ge=100, le=10000, description="Maximum iterations")


class SelectionConfig(BaseModel):
    """Configuration for model selection and blending."""
    
    validation_window_bars: int = Field(
        default=60,
        ge=20,
        le=200,
        description="Number of bars for validation"
    )
    selection_metric: Literal["brier_then_pinball", "composite", "brier_only"] = Field(
        default="brier_then_pinball",
        description="Model selection metric"
    )
    blend: bool = Field(default=True, description="Enable model blending")
    blend_temp: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Temperature for softmax blending weights"
    )


class OutputConfig(BaseModel):
    """Configuration for output formatting."""
    
    ascii: bool = Field(default=True, description="Enable ASCII terminal output")
    plots: bool = Field(default=False, description="Enable plotting")
    export_csv: Optional[str] = Field(default=None, description="CSV export path")


class Config(BaseModel):
    """Main configuration for the predictor system."""
    
    # Required inputs
    symbol: str = Field(..., description="Stock symbol to predict")
    lookback_days: int = Field(..., ge=30, le=2000, description="Number of days to look back")
    
    # Optional inputs
    horizon_set: List[str] = Field(
        default=["1d", "1w", "1m"],
        description="Prediction horizons"
    )
    timeframe: Literal["daily", "hourly", "realtime"] = Field(
        default="daily",
        description="Data timeframe"
    )
    seasonality_years: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Years of data for seasonality"
    )
    
    # Sub-configurations
    bins: BinningConfig = Field(default_factory=BinningConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Data source config
    db_host: str = Field(default="localhost", description="Database server host")
    db_port: int = Field(default=9002, description="Database server port")
    db_timeout: float = Field(default=30.0, description="Database request timeout")
    
    # Random seed for reproducibility
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    
    @validator('horizon_set')
    def validate_horizons(cls, v):
        """Validate horizon set."""
        valid_horizons = ["1d", "1w", "1m", "3m", "6m", "1y"]
        for horizon in v:
            if horizon not in valid_horizons:
                raise ValueError(f"Invalid horizon '{horizon}'. Must be one of {valid_horizons}")
        return v
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol format."""
        if not v or not isinstance(v, str):
            raise ValueError("Symbol must be a non-empty string")
        return v.upper().strip()
    
    def get_horizon_days(self, horizon: str) -> int:
        """Convert horizon string to number of days."""
        horizon_map = {
            "1d": 1,
            "1w": 7,
            "1m": 30,
            "3m": 90,
            "6m": 180,
            "1y": 365
        }
        return horizon_map.get(horizon, 1)
    
    def get_db_url(self) -> str:
        """Get database server URL."""
        return f"http://{self.db_host}:{self.db_port}"
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
        use_enum_values = True


# Default configurations for common use cases
DEFAULT_CONFIG = Config(
    symbol="AAPL",
    lookback_days=365,
    horizon_set=["1d", "1w", "1m"],
    timeframe="daily"
)

QUICK_CONFIG = Config(
    symbol="AAPL", 
    lookback_days=180,
    horizon_set=["1d", "1w"],
    timeframe="daily",
    models=ModelConfig(
        markov=True,
        gbdt=True,
        logistic_quantile=False,
        hmm=False
    )
)

COMPREHENSIVE_CONFIG = Config(
    symbol="AAPL",
    lookback_days=730,
    horizon_set=["1d", "1w", "1m", "3m"],
    timeframe="daily",
    seasonality_years=5,
    models=ModelConfig(
        markov=True,
        gbdt=True,
        logistic_quantile=True,
        hmm=True
    ),
    selection=SelectionConfig(
        validation_window_bars=100,
        blend=True,
        blend_temp=0.8
    )
)
