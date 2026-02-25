#!/usr/bin/env python3
"""Deprecated: use predict_close.py instead."""
import warnings
warnings.warn(
    "predict_close_now.py is deprecated. Use predict_close.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export public API so existing imports keep working
from scripts.predict_close import (  # noqa: F401
    predict_close,
    predict_future_close,
    train_0dte_model,
    clear_cache_files,
    clear_model_files,
)

if __name__ == '__main__':
    import subprocess, sys
    sys.exit(subprocess.call(
        [sys.executable, __file__.replace('predict_close_now.py', 'predict_close.py')] + sys.argv[1:]
    ))
