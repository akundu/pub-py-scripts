"""
Models for the Next-Action and Magnitude Predictor.

This package contains the ensemble of prediction models:
- Markov Chain Model
- Gradient Boosted Decision Trees (GBDT)
- Logistic + Quantile Regression
- Optional HMM for regime detection
"""

from .markov_model import MarkovModel
from .gbdt import GBDTModel
from .logit_quant import LogisticQuantileModel

__all__ = [
    "MarkovModel",
    "GBDTModel", 
    "LogisticQuantileModel"
]
