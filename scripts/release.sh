#!/usr/bin/env bash
set -euo pipefail

# --- Usage ---
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <VERSION>"
    echo "Example: $0 0.4.1"
    exit 1
fi

VERSION="$1"

# --- Validate semver format ---
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: VERSION must be in semver format (e.g. 1.2.3)"
    exit 1
fi

# --- Ensure clean working tree ---
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree is not clean. Commit or stash changes first."
    exit 1
fi

# --- Ensure on main branch ---
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: must be on 'main' branch (currently on '$BRANCH')"
    exit 1
fi

# --- Ensure GitHub Actions CI is green ---
echo "==> Checking GitHub Actions status..."
CI_STATUS=$(gh run list --branch main --limit 1 --json status,conclusion --jq '.[0]')
CI_CONCLUSION=$(echo "$CI_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('conclusion',''))")
CI_STATE=$(echo "$CI_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
if [[ "$CI_STATE" != "completed" ]]; then
    echo "Error: latest CI run on main is still '${CI_STATE}'. Wait for it to finish."
    exit 1
fi
if [[ "$CI_CONCLUSION" != "success" ]]; then
    echo "Error: latest CI run on main concluded with '${CI_CONCLUSION}'. Fix CI before releasing."
    echo "       See: gh run list --branch main --limit 1"
    exit 1
fi
echo "    CI is green."

echo "==> Releasing v${VERSION}"

# --- Run mypy ---
echo "==> Running mypy..."
uv run mypy src/
echo "    mypy passed."

# --- Bump version in source ---
sed -i '' "s/^__version__ = .*/__version__ = \"${VERSION}\"/" src/roomkit/_version.py
echo "    Updated src/roomkit/_version.py"

# --- Bump version in test ---
sed -i '' "s/assert roomkit.__version__ == .*/assert roomkit.__version__ == \"${VERSION}\"/" tests/test_public_api.py
echo "    Updated tests/test_public_api.py"

# --- Run tests ---
echo "==> Running tests..."
uv run pytest
echo "    Tests passed."

# --- Commit ---
git add src/roomkit/_version.py tests/test_public_api.py
git commit -m "Bump version to ${VERSION}"
echo "    Committed."

# --- Tag ---
git tag "v${VERSION}"
echo "    Tagged v${VERSION}."

# --- Build + publish ---
echo "==> Building..."
uv build
echo "==> Publishing to PyPI..."
if [[ -n "${UV_PUBLISH_TOKEN:-}" ]]; then
    uv publish
elif [[ -f "$HOME/.pypirc" ]]; then
    PYPI_TOKEN=$(python3 -c "import configparser; c = configparser.ConfigParser(); c.read('$HOME/.pypirc'); print(c.get('pypi', 'password'))")
    uv publish --username __token__ --password "$PYPI_TOKEN"
else
    echo "Error: No PyPI credentials found. Set UV_PUBLISH_TOKEN or create ~/.pypirc"
    exit 1
fi
echo "    Published."

# --- Push ---
echo "==> Pushing..."
git push && git push --tags
echo "    Pushed."

# --- GitHub Release ---
# Find the previous tag to generate the changelog range.
PREV_TAG=$(git tag --sort=-v:refname | grep -E '^v[0-9]' | sed -n '2p')
echo "==> Creating GitHub Release (v${VERSION}, since ${PREV_TAG:-scratch})..."

NOTES="## What's Changed"$'\n\n'
if [[ -n "${PREV_TAG:-}" ]]; then
    NOTES+=$(git log "${PREV_TAG}..v${VERSION}" --pretty=format:"- %s" \
        | grep -v "^- Bump version")
    NOTES+=$'\n\n'"**Full Changelog**: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/compare/${PREV_TAG}...v${VERSION}"
else
    NOTES+=$(git log "v${VERSION}" --pretty=format:"- %s" \
        | grep -v "^- Bump version")
fi

gh release create "v${VERSION}" \
    --title "v${VERSION}" \
    --notes "${NOTES}"
echo "    GitHub Release created."

echo ""
echo "==> Release v${VERSION} complete!"
