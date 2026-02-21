"""
Next-Action and Magnitude Predictor

An ensemble prediction system that predicts next action (up/down/flat) and expected movement (%)
over different horizons using multiple complementary algorithms.
"""

__version__ = "1.0.0"
__author__ = "Stock Prediction System"

from .config import Config, ModelConfig, BinningConfig
from .data_provider import DbServerProvider
from .features import build_features, FeatureBuilder
from .models.markov_model import MarkovModel
from .models.gbdt import GBDTModel
from .models.logit_quant import LogisticQuantileModel
from .selection import ModelSelector, ModelBlender
from .inference import Predictor
from .eval import Evaluator
from .viz import Visualizer
from .cli import main

__all__ = [
    "Config",
    "ModelConfig", 
    "BinningConfig",
    "DbServerProvider",
    "build_features",
    "FeatureBuilder",
    "MarkovModel",
    "GBDTModel", 
    "LogisticQuantileModel",
    "ModelSelector",
    "ModelBlender",
    "Predictor",
    "Evaluator",
    "Visualizer",
    "main"
]
