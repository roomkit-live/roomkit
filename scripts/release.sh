#!/usr/bin/env bash
set -euo pipefail

# --- Usage ---
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <VERSION>"
    echo "Example: $0 0.4.1"
    exit 1
fi

VERSION="$1"

# Pinned SBOM generator: a floating version would make the attached SBOM's
# format/tooling non-reproducible across releases. Bump deliberately.
CYCLONEDX_BOM_VERSION="7.3.0"

# --- Validate semver format (with optional pre-release suffix) ---
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-a-zA-Z0-9.]+)?$ ]]; then
    echo "Error: VERSION must be in semver format (e.g. 1.2.3 or 1.2.3a1)"
    exit 1
fi

# --- Ensure clean working tree ---
# A re-run after a mid-release failure may have already applied the version bump
# to these two files (sed runs before the commit). That is the ONLY permitted
# dirt — it lets the script resume; anything else must be committed or stashed.
UNEXPECTED="$(git status --porcelain \
    | grep -vE '(src/roomkit/_version\.py|tests/test_public_api\.py)$' || true)"
if [[ -n "$UNEXPECTED" ]]; then
    echo "Error: working tree has changes beyond the version bump. Commit or stash first:"
    echo "$UNEXPECTED"
    exit 1
fi

# --- Ensure on main branch ---
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: must be on 'main' branch (currently on '$BRANCH')"
    exit 1
fi

# A local tag v${VERSION} means a prior run already built, committed, and tagged
# this release — so this invocation is a RESUME, not a fresh release. That gates
# the PyPI safety check below and the dev-cycle detection.
TAG_EXISTS=0
if git rev-parse -q --verify "refs/tags/v${VERSION}" >/dev/null; then
    TAG_EXISTS=1
fi

# --- Resume shortcut: the release fully completed and the next dev cycle was
# opened, but its commit/push failed. The tree is on a .dev version AND the tag
# exists — nothing left to do but re-push (idempotent). ---
CURRENT_VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' src/roomkit/_version.py)
if [[ "$CURRENT_VERSION" == *.dev* && "$TAG_EXISTS" == "1" ]]; then
    echo "==> v${VERSION} already released (tree on ${CURRENT_VERSION}); re-pushing git state."
    git push && git push --tags
    echo "==> Done."
    exit 0
fi

# --- Refuse a version already on PyPI, UNLESS resuming (the local tag proves a
# prior run of THIS release; publish below skips files already uploaded). ---
echo "==> Checking PyPI for an existing ${VERSION}..."
if curl -sfL "https://pypi.org/pypi/roomkit/${VERSION}/json" -o /dev/null; then
    if [[ "$TAG_EXISTS" == "0" ]]; then
        echo "Error: roomkit ${VERSION} already on PyPI and no local tag v${VERSION} to"
        echo "       resume from — pick a new version."
        exit 1
    fi
    echo "    ${VERSION} is on PyPI, but tag v${VERSION} exists — resuming; publish skips existing files."
else
    echo "    ${VERSION} is not on PyPI."
fi

# --- Require a CHANGELOG entry for this version ---
if ! grep -qE "^## \[${VERSION}\]" CHANGELOG.md; then
    echo "Error: CHANGELOG.md has no '## [${VERSION}]' entry — write it before releasing."
    exit 1
fi
echo "    CHANGELOG.md documents ${VERSION}."

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

# --- Run type checker ---
echo "==> Running ty..."
uv run ty check src/roomkit/
echo "    ty passed."

# --- Bump version in source ---
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' "s/^__version__ = .*/__version__ = \"${VERSION}\"/" src/roomkit/_version.py
else
    sed -i "s/^__version__ = .*/__version__ = \"${VERSION}\"/" src/roomkit/_version.py
fi
echo "    Updated src/roomkit/_version.py"

# --- Bump version in test ---
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' "s/assert roomkit.__version__ == .*/assert roomkit.__version__ == \"${VERSION}\"/" tests/test_public_api.py
else
    sed -i "s/assert roomkit.__version__ == .*/assert roomkit.__version__ == \"${VERSION}\"/" tests/test_public_api.py
fi
echo "    Updated tests/test_public_api.py"

# --- Run tests ---
echo "==> Running tests..."
uv run pytest
echo "    Tests passed."

# --- Build + SBOM BEFORE any Git mutation ---
# Everything below the commit is irreversible or awkward to unwind, so the
# steps that can fail on a flaky network (the build download and, especially,
# fetching the pinned SBOM generator) run FIRST — while the tree is still clean
# and the version bump is only a local, un-committed sed edit. A failure here
# leaves nothing to undo: fix the network and re-run from scratch.
echo "==> Building..."
uv build
# Only ship artifacts for the current version — uploading the whole dist/ dir
# fails when older wheels from prior releases are still sitting there.
DIST_FILES=(
    "dist/roomkit-${VERSION}.tar.gz"
    "dist/roomkit-${VERSION}-py3-none-any.whl"
)
for f in "${DIST_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: expected build artifact missing: $f"
        exit 1
    fi
done

# --- Software Bill of Materials (CycloneDX) ---
# A per-release inventory of the runtime dependency tree (core + the `providers`
# extra), attached to the GitHub Release for downstream vulnerability and
# license audits. `--no-emit-project` lists the dependencies, not roomkit itself.
# The generator is pinned (CYCLONEDX_BOM_VERSION) for reproducibility.
echo "==> Generating SBOM (cyclonedx-bom==${CYCLONEDX_BOM_VERSION})..."
SBOM_FILE="dist/roomkit-${VERSION}.cdx.json"
uv export --extra providers --no-dev --frozen --no-emit-project --format requirements-txt \
    > dist/roomkit-sbom-requirements.txt
uvx --from "cyclonedx-bom==${CYCLONEDX_BOM_VERSION}" cyclonedx-py requirements \
    dist/roomkit-sbom-requirements.txt --of JSON -o "${SBOM_FILE}"
if [[ ! -s "$SBOM_FILE" ]]; then
    echo "Error: SBOM generation produced no output: ${SBOM_FILE}"
    exit 1
fi
echo "    Wrote ${SBOM_FILE}"

# --- Commit (idempotent: safe to re-run after a later step failed) ---
git add src/roomkit/_version.py tests/test_public_api.py
if git diff --cached --quiet; then
    echo "    Version ${VERSION} already committed — skipping."
else
    git commit -m "Bump version to ${VERSION}"
    echo "    Committed."
fi

# --- Tag (idempotent) ---
if git rev-parse -q --verify "refs/tags/v${VERSION}" >/dev/null; then
    echo "    Tag v${VERSION} already exists — skipping."
else
    git tag "v${VERSION}"
    echo "    Tagged v${VERSION}."
fi

# --- Push git state BEFORE publishing ---
# The PyPI upload below is irreversible; pushing the commit, tag, and GitHub
# Release first means a failed upload leaves git and PyPI consistent and the
# upload can simply be retried — PyPI is never ahead of the repository.
echo "==> Pushing..."
git push && git push --tags
echo "    Pushed."

# --- GitHub Release (idempotent) ---
# A resume after a failed PyPI upload must not die here: if the Release already
# exists from the prior run, skip creation and continue to the (retryable)
# publish step below.
if gh release view "v${VERSION}" >/dev/null 2>&1; then
    echo "==> GitHub Release v${VERSION} already exists — skipping create."
else
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

    PRERELEASE_FLAG=""
    if [[ "$VERSION" =~ [a-zA-Z] ]]; then
        PRERELEASE_FLAG="--prerelease"
    fi

    gh release create "v${VERSION}" \
        --title "v${VERSION}" \
        ${PRERELEASE_FLAG} \
        --notes "${NOTES}" \
        "${SBOM_FILE}#SBOM (CycloneDX)"
    echo "    GitHub Release created (SBOM attached)."
fi

# --- Publish to PyPI (last, irreversible step) ---
# `--check-url` skips files already on the index, so a re-run after a partial
# upload (e.g. sdist landed, wheel failed) uploads only what is missing instead
# of failing on the duplicate — making publish idempotent.
PYPI_INDEX="https://pypi.org/simple/roomkit/"
echo "==> Publishing to PyPI..."
if [[ -n "${UV_PUBLISH_TOKEN:-}" ]]; then
    uv publish --check-url "$PYPI_INDEX" "${DIST_FILES[@]}"
elif [[ -f "$HOME/.pypirc" ]]; then
    PYPI_TOKEN=$(python3 -c "import configparser; c = configparser.ConfigParser(); c.read('$HOME/.pypirc'); print(c.get('pypi', 'password'))")
    uv publish --check-url "$PYPI_INDEX" --username __token__ --password "$PYPI_TOKEN" "${DIST_FILES[@]}"
else
    echo "Error: No PyPI credentials found. Set UV_PUBLISH_TOKEN or create ~/.pypirc"
    exit 1
fi
echo "    Published."

# --- Open the next development cycle ---
# Leaving main on the released version makes _version.py / `git describe` lie
# about every commit after a release (they look like the release). Move main onto
# a dev marker of the next minor (e.g. 0.18.0 -> 0.19.0.dev0) so builds from main
# are identifiable as pre-release. The release artifact published above already
# carries the real version; this only affects the source tree going forward.
# Prereleases (alpha/beta/rc) leave main as-is.
if [[ ! "$VERSION" =~ [a-zA-Z] ]]; then
    DEV_VERSION=$(python3 -c "p='${VERSION}'.split('.'); print(f'{p[0]}.{int(p[1]) + 1}.0.dev0')")
    echo "==> Opening development of ${DEV_VERSION}..."
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' "s/^__version__ = .*/__version__ = \"${DEV_VERSION}\"/" src/roomkit/_version.py
    else
        sed -i "s/^__version__ = .*/__version__ = \"${DEV_VERSION}\"/" src/roomkit/_version.py
    fi
    git add src/roomkit/_version.py
    if git diff --cached --quiet; then
        echo "    Already on ${DEV_VERSION}."
    else
        git commit -m "Begin ${DEV_VERSION} development"
    fi
    git push
    echo "    main now on ${DEV_VERSION}."
fi

echo ""
echo "==> Release v${VERSION} complete!"
