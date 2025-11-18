#!/bin/bash
# Script to commit all currently modified files to git
# Generated automatically based on current git status

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Git Commit Script for Modified Files ===${NC}"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Get list of modified files
MODIFIED_FILES=$(git status --short | grep "^ M" | awk '{print $2}')

if [ -z "$MODIFIED_FILES" ]; then
    echo -e "${YELLOW}No modified files found.${NC}"
    exit 0
fi

# Count modified files
FILE_COUNT=$(echo "$MODIFIED_FILES" | wc -l | tr -d ' ')
echo -e "${GREEN}Found $FILE_COUNT modified file(s):${NC}"
echo ""
echo "$MODIFIED_FILES" | nl
echo ""

# Show git status
echo -e "${YELLOW}Current git status:${NC}"
git status --short
echo ""

# Ask for confirmation
read -p "Do you want to commit all these files? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Commit cancelled.${NC}"
    exit 0
fi

# Ask for commit message
echo ""
echo -e "${YELLOW}Enter commit message (or press Enter for default):${NC}"
read -r COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="chore: update modified files

Updated files:
$(echo "$MODIFIED_FILES" | sed 's/^/- /')
"
fi

# Stage all modified files
echo ""
echo -e "${GREEN}Staging modified files...${NC}"
echo "$MODIFIED_FILES" | while read -r file; do
    if [ -f "$file" ]; then
        git add "$file"
        echo "  ✓ Staged: $file"
    else
        echo -e "  ${RED}✗ File not found: $file${NC}"
    fi
done

# Commit
echo ""
echo -e "${GREEN}Committing changes...${NC}"
if git commit -m "$COMMIT_MSG"; then
    echo ""
    echo -e "${GREEN}✓ Successfully committed $FILE_COUNT file(s)!${NC}"
    echo ""
    echo -e "${YELLOW}Recent commits:${NC}"
    git log --oneline -5
    echo ""
    echo -e "${YELLOW}To push to remote:${NC}"
    echo "  git push"
else
    echo -e "${RED}✗ Commit failed!${NC}"
    exit 1
fi


