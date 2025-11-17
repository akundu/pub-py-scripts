#!/bin/bash
# Git commit script for all uncommitted changes

set -e

echo "=== Git Commit All Changes ==="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Show current status (tracked and untracked)
echo "Current git status (tracked + untracked):"
git status --short
echo ""

# Show tracked changes only (mods/deletes) for clarity
TRACKED_STATUS=$(git status --short -uno)
if [ -n "$TRACKED_STATUS" ]; then
    echo "Tracked changes (will be staged automatically):"
    echo "$TRACKED_STATUS" | sed 's/^/  - /'
else
    echo "No tracked changes detected."
fi
echo ""

# Get list of untracked files (excluding common ignore patterns)
UNTRACKED_FILES=$(git ls-files --others --exclude-standard | grep -v -E '\.DS_Store|\.pyc|__pycache__|\.log$|\.info$|access\.log|log$|results\.info' || true)

if [ -n "$UNTRACKED_FILES" ]; then
    echo "Untracked files (excluding temp files):"
    echo "$UNTRACKED_FILES" | head -20 | sed 's/^/  - /'
    if [ $(echo "$UNTRACKED_FILES" | wc -l) -gt 20 ]; then
        echo "  ... and $(($(echo "$UNTRACKED_FILES" | wc -l) - 20)) more"
    fi
    echo ""
fi

# Stage tracked changes (modifications + deletions)
echo "Staging tracked changes (modifications & deletions)..."
git add -u
echo "  ✓ Tracked changes staged via git add -u"

# Add fetch_options.py if it has changes (it might be untracked or modified)
if git status --short scripts/fetch_options.py 2>/dev/null | grep -qE '^[MAD]'; then
    echo "  ✓ Adding: scripts/fetch_options.py"
    git add scripts/fetch_options.py
elif [ -f "scripts/fetch_options.py" ] && ! git ls-files --error-unmatch scripts/fetch_options.py > /dev/null 2>&1; then
    echo "  ✓ Adding: scripts/fetch_options.py (untracked)"
    git add scripts/fetch_options.py
fi

# Add common/market_hours.py if it exists and is untracked
if [ -f "common/market_hours.py" ] && ! git ls-files --error-unmatch common/market_hours.py > /dev/null 2>&1; then
    echo "  ✓ Adding: common/market_hours.py"
    git add common/market_hours.py
fi

echo ""

# Always add the commit helper script itself
if [ -f "git_commit_all_changes.sh" ]; then
    git add git_commit_all_changes.sh
    echo "  ✓ Adding: git_commit_all_changes.sh"
fi

# Show what will be committed
echo ""
echo "Files staged for commit:"
STAGED_LIST=$(git diff --cached --name-only)
if [ -n "$STAGED_LIST" ]; then
    echo "$STAGED_LIST" | sed 's/^/  - /'
else
    echo "  (no files staged yet)"
fi
echo ""

# Create commit message based on changes
COMMIT_MSG="feat: enhance covered-call automation and clean up docs

- Update run_scripts/covered_call_generation.sh:
  * Add Gemini analysis cooldown + HTML output
  * Improve market-hours detection, filters, and sensible price thresholds
  * Increase TOP_N default and allow volume filter automation

- Remove outdated scripts/README_comprehensive_stock_data.md

- Add git_commit_all_changes.sh helper to stage & commit repo updates"

echo "Commit message:"
echo "----------------------------------------"
echo "$COMMIT_MSG"
echo "----------------------------------------"
echo ""

# Ask for confirmation
read -p "Proceed with commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Commit cancelled."
    exit 1
fi

# Create commit
git commit -m "$COMMIT_MSG"

echo ""
echo "✓ Commit created successfully!"
echo ""
echo "Current branch: $(git branch --show-current)"
echo ""
echo "To push to remote:"
echo "  git push"
echo ""
echo "To view the commit:"
echo "  git show HEAD"

