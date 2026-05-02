#!/bin/bash
# bump_version.sh — update version across all files
#
# Usage: ./scripts/bump_version.sh 1.1.0
#
# Updates version in:
#   - pyproject.toml
#   - goodhart/__init__.py
#   - proofs/lakefile.toml

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.1.0"
    exit 1
fi

NEW="$1"
cd "$(dirname "$0")/.."

# Validate format
if ! echo "$NEW" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: version must be X.Y.Z (got: $NEW)"
    exit 1
fi

# Get current version
OLD=$(python -c "import goodhart; print(goodhart.__version__)")
echo "Bumping $OLD → $NEW"

# Update all files
sed -i.bak "s/version = \"$OLD\"/version = \"$NEW\"/" pyproject.toml
sed -i.bak "s/__version__ = \"$OLD\"/__version__ = \"$NEW\"/" goodhart/__init__.py
sed -i.bak "s/version = \"$OLD\"/version = \"$NEW\"/" proofs/lakefile.toml

# Clean up backups
rm -f pyproject.toml.bak goodhart/__init__.py.bak proofs/lakefile.toml.bak

# Verify
CHECK=$(python -c "import goodhart; print(goodhart.__version__)")
if [ "$CHECK" != "$NEW" ]; then
    echo "Error: version check failed (got $CHECK, expected $NEW)"
    exit 1
fi

echo "Updated to $NEW in:"
echo "  pyproject.toml"
echo "  goodhart/__init__.py"
echo "  proofs/lakefile.toml"
echo
echo "Next steps:"
echo "  git add pyproject.toml goodhart/__init__.py proofs/lakefile.toml"
echo "  git commit -m 'Release $NEW'"
echo "  git push origin main"
echo "  gh release create v$NEW --title 'v$NEW' --generate-notes"
