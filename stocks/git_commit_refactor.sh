#!/bin/bash
# Git commit script for options_analyzer refactoring

set -e

echo "=== Options Analyzer Refactoring - Git Commit ==="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Show status
echo "Current git status:"
git status --short
echo ""

# Add all modified and new files
echo "Adding files to git..."
git add common/common.py
git add common/options_utils.py
git add common/options/
git add scripts/options_analyzer.py
git add git_commit_refactor.sh

# Remove old files if they still exist in git
if git ls-files --error-unmatch scripts/options_filters.py > /dev/null 2>&1; then
    git rm scripts/options_filters.py 2>/dev/null || true
fi
if git ls-files --error-unmatch scripts/options_workers.py > /dev/null 2>&1; then
    git rm scripts/options_workers.py 2>/dev/null || true
fi
if git ls-files --error-unmatch scripts/options_formatting.py > /dev/null 2>&1; then
    git rm scripts/options_formatting.py 2>/dev/null || true
fi
if git ls-files --error-unmatch scripts/options_spread.py > /dev/null 2>&1; then
    git rm scripts/options_spread.py 2>/dev/null || true
fi
if git ls-files --error-unmatch scripts/options_refresh.py > /dev/null 2>&1; then
    git rm scripts/options_refresh.py 2>/dev/null || true
fi

echo ""
echo "Files staged:"
git diff --cached --name-only
echo ""

# Create commit
COMMIT_MSG="Refactor options_analyzer.py: modularize, deduplicate, and reorganize

- Moved utility functions to common/common.py and common/options_utils.py
- Created common/options/ directory with modular options analysis modules:
  - options_filters.py: FilterExpression and FilterParser
  - options_workers.py: multiprocessing worker functions
  - options_spread.py: spread analysis functions
  - options_refresh.py: refresh functionality
  - options_formatting.py: output formatting functions
  - __init__.py: package initialization with exports
- Moved all options modules from scripts/ to common/options/ for better organization
- Updated all imports to use new common.options package structure
- Refactored scripts/options_analyzer.py to use new modules
- Removed ~1030 lines of duplicate code from options_analyzer.py
- Added function aliases for backward compatibility
- Fixed worker process imports to handle new module locations
- Improved code organization and maintainability"

echo "Commit message:"
echo "$COMMIT_MSG"
echo ""

read -p "Proceed with commit? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "$COMMIT_MSG"
    echo ""
    echo "✓ Commit created successfully!"
    echo ""
    echo "To push to remote:"
    echo "  git push"
else
    echo "Commit cancelled."
    exit 1
fi

