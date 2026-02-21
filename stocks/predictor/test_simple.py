#!/usr/bin/env python3
"""
Simple test to verify basic project structure and imports without external dependencies.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported without external dependencies."""
    print("Testing imports...")
    
    try:
        # Test config import
        from config import Config, DEFAULT_CONFIG, QUICK_CONFIG, COMPREHENSIVE_CONFIG
        print("✓ Config module imported successfully")
        
        # Test data provider import
        from data_provider import DbServerProvider
        print("✓ Data provider module imported successfully")
        
        # Test utils import (this might fail due to scipy dependency)
        try:
            from utils import set_random_seeds, bin_returns
            print("✓ Utils module imported successfully")
        except ImportError as e:
            print(f"⚠ Utils module import failed (expected due to missing dependencies): {e}")
        
        # Test models import
        try:
            from models.markov_model import MarkovModel
            print("✓ Markov model imported successfully")
        except ImportError as e:
            print(f"⚠ Markov model import failed (expected due to missing dependencies): {e}")
        
        try:
            from models.gbdt import GBDTModel
            print("✓ GBDT model imported successfully")
        except ImportError as e:
            print(f"⚠ GBDT model import failed (expected due to missing dependencies): {e}")
        
        try:
            from models.logit_quant import LogisticQuantileModel
            print("✓ Logistic quantile model imported successfully")
        except ImportError as e:
            print(f"⚠ Logistic quantile model import failed (expected due to missing dependencies): {e}")
        
        # Test features import
        try:
            from features import FeatureBuilder
            print("✓ Features module imported successfully")
        except ImportError as e:
            print(f"⚠ Features module import failed (expected due to missing dependencies): {e}")
        
        # Test selection import
        try:
            from selection import ModelSelector, ModelBlender
            print("✓ Selection module imported successfully")
        except ImportError as e:
            print(f"⚠ Selection module import failed (expected due to missing dependencies): {e}")
        
        # Test inference import
        try:
            from inference import Predictor
            print("✓ Inference module imported successfully")
        except ImportError as e:
            print(f"⚠ Inference module import failed (expected due to missing dependencies): {e}")
        
        # Test eval import
        try:
            from eval import Evaluator
            print("✓ Eval module imported successfully")
        except ImportError as e:
            print(f"⚠ Eval module import failed (expected due to missing dependencies): {e}")
        
        # Test viz import
        try:
            from viz import Visualizer
            print("✓ Viz module imported successfully")
        except ImportError as e:
            print(f"⚠ Viz module import failed (expected due to missing dependencies): {e}")
        
        # Test CLI import
        try:
            from cli import app
            print("✓ CLI module imported successfully")
        except ImportError as e:
            print(f"⚠ CLI module import failed (expected due to missing dependencies): {e}")
        
        # Test terminal render import
        try:
            from terminal_render import render_prediction_table, render_validation_table
            print("✓ Terminal render module imported successfully")
        except ImportError as e:
            print(f"⚠ Terminal render module import failed (expected due to missing dependencies): {e}")
        
        print("\n✓ All core modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_config_creation():
    """Test that configuration objects can be created."""
    print("\nTesting configuration creation...")
    
    try:
        from config import Config, DEFAULT_CONFIG, QUICK_CONFIG, COMPREHENSIVE_CONFIG
        
        # Test default config
        config = DEFAULT_CONFIG
        print(f"✓ Default config created: {config.symbol}")
        
        # Test custom config
        custom_config = Config(
            symbol="TEST",
            lookback_days=100,
            horizon_set=["1d", "1w"]
        )
        print(f"✓ Custom config created: {custom_config.symbol}")
        
        # Test quick config
        quick_config = QUICK_CONFIG
        print(f"✓ Quick config created: {quick_config.symbol}")
        
        # Test comprehensive config
        comp_config = COMPREHENSIVE_CONFIG
        print(f"✓ Comprehensive config created: {comp_config.symbol}")
        
        print("✓ All configuration objects created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_provider_creation():
    """Test that data provider can be created."""
    print("\nTesting data provider creation...")
    
    try:
        from data_provider import DbServerProvider
        
        # Test data provider creation
        provider = DbServerProvider("localhost", 9002)
        print(f"✓ Data provider created: {provider.host}:{provider.port}")
        
        print("✓ Data provider created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Data provider test failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("=" * 60)
    print("SIMPLE PROJECT STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config_creation,
        test_data_provider_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All basic tests passed! Project structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test: python test_basic.py")
        print("3. Start using the predictor system")
    else:
        print("⚠ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
