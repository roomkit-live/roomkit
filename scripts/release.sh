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

# --- Validate the PEP 440 forms this Python release workflow supports ---
# Reject SemVer's hyphenated prerelease spelling (1.2.3-rc.1): Python
# normalizes it to 1.2.3rc1, which otherwise makes the built filenames differ
# from DIST_FILES below and leaves a half-finished release.
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
    echo "Error: VERSION must be a PEP 440 release (e.g. 1.2.3 or 1.2.3rc1)"
    exit 1
fi

# --- Ensure clean working tree ---
# A re-run after a mid-release failure may have already applied the version bump
# to this file (sed runs before the commit). That is the ONLY permitted
# dirt — it lets the script resume; anything else must be committed or stashed.
UNEXPECTED="$(git status --porcelain \
    | grep -vE '^.. src/roomkit/_version\.py$' || true)"
if [[ -n "$UNEXPECTED" ]]; then
    echo "Error: working tree has changes beyond the version bump. Commit or stash first:"
    echo "$UNEXPECTED"
    exit 1
fi

# The resumable dirty file must contain *only* the version assignment. Merely
# checking its path would allow unrelated code or data added to _version.py to
# be swept into the release commit under the guise of a version bump.
if [[ ! -f src/roomkit/_version.py ]] \
    || [[ "$(grep -c '^' src/roomkit/_version.py)" != "1" ]] \
    || ! grep -qEx '__version__ = "[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+|\.dev[0-9]+)?"' \
        src/roomkit/_version.py; then
    echo "Error: src/roomkit/_version.py must contain only one valid __version__ assignment."
    exit 1
fi

# --- Ensure on main branch ---
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
    echo "Error: must be on 'main' branch (currently on '$BRANCH')"
    exit 1
fi

# A valid local tag v${VERSION} means a prior run already built, committed, and
# tagged this release. Validate its tree and exact position before allowing it
# to gate the PyPI safety check or the dev-cycle shortcut.
TAG_EXISTS=0
RELEASE_COMMIT_EXISTS=0
RELEASE_TAG_SHA=""
RELEASE_PARENT_SHA=""
RESUME_DEV_CHILD=0
if git rev-parse -q --verify "refs/tags/v${VERSION}" >/dev/null; then
    TAG_EXISTS=1
    RELEASE_TAG_SHA=$(git rev-parse "v${VERSION}^{}")
    TAG_VERSION=$(git show "${RELEASE_TAG_SHA}:src/roomkit/_version.py" 2>/dev/null \
        | sed -n 's/^__version__ = "\(.*\)"/\1/p')
    if [[ "$TAG_VERSION" != "$VERSION" ]]; then
        echo "Error: resume tag v${VERSION} contains version '${TAG_VERSION:-missing}', expected '${VERSION}'."
        exit 1
    fi

    TAG_CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r "$RELEASE_TAG_SHA")
    if [[ "$TAG_CHANGED_FILES" != "src/roomkit/_version.py" ]]; then
        echo "Error: resume tag v${VERSION} must point to a version-only commit."
        echo "       Tagged commit changes: ${TAG_CHANGED_FILES:-none}"
        exit 1
    fi
    if ! RELEASE_PARENT_SHA=$(git rev-parse "${RELEASE_TAG_SHA}^" 2>/dev/null); then
        echo "Error: resume tag v${VERSION} has no code-bearing parent commit."
        exit 1
    fi

    HEAD_SHA=$(git rev-parse HEAD)
    if [[ "$HEAD_SHA" != "$RELEASE_TAG_SHA" ]]; then
        HEAD_PARENT_SHA=$(git rev-parse HEAD^ 2>/dev/null || true)
        if [[ "$HEAD_PARENT_SHA" != "$RELEASE_TAG_SHA" ]]; then
            echo "Error: resume tag v${VERSION} must point to HEAD or its direct parent."
            echo "       Refusing to mix a stale tag with artifacts built from the current HEAD."
            exit 1
        fi
        HEAD_CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)
        if [[ "$HEAD_CHANGED_FILES" != "src/roomkit/_version.py" ]]; then
            echo "Error: the commit after resume tag v${VERSION} must only open the next dev cycle."
            echo "       HEAD changes: ${HEAD_CHANGED_FILES:-none}"
            exit 1
        fi
        RESUME_DEV_CHILD=1
    fi
fi

# A failure between the version commit and tag creation leaves a clean,
# version-only HEAD but no tag. Recognize that exact state as resumable and use
# its code-bearing parent for CI, just as the tagged resume path does. Refuse a
# target version committed together with other files: release tags are required
# to point at a version-only commit so a re-run can prove what happened.
HEAD_VERSION=$(git show HEAD:src/roomkit/_version.py 2>/dev/null \
    | sed -n 's/^__version__ = "\(.*\)"/\1/p')
if [[ "$TAG_EXISTS" == "0" && "$HEAD_VERSION" == "$VERSION" ]]; then
    HEAD_CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)
    if [[ "$HEAD_CHANGED_FILES" != "src/roomkit/_version.py" ]]; then
        echo "Error: committed release version ${VERSION} must be in a version-only commit."
        echo "       HEAD changes: ${HEAD_CHANGED_FILES:-none}"
        exit 1
    fi
    if ! RELEASE_PARENT_SHA=$(git rev-parse HEAD^ 2>/dev/null); then
        echo "Error: version-only release commit for ${VERSION} has no code-bearing parent."
        exit 1
    fi
    RELEASE_COMMIT_EXISTS=1
    RELEASE_TAG_SHA=$(git rev-parse HEAD)
fi

# --- Resume shortcut: the release fully completed and the next dev cycle was
# already COMMITTED, but its push failed. Read the version from HEAD (not the
# worktree): if the dev-cycle commit itself failed, the bump is only staged, so
# the worktree shows .dev while HEAD is still the release commit — that case
# must fall through and re-run to finish the commit, not exit here. ---
if [[ "$RESUME_DEV_CHILD" == "1" ]]; then
    if [[ "$VERSION" =~ [a-zA-Z] ]]; then
        echo "Error: prerelease tag v${VERSION} cannot have an automatic next-dev commit."
        exit 1
    fi
    IFS='.' read -r VERSION_MAJOR VERSION_MINOR _VERSION_PATCH <<< "$VERSION"
    EXPECTED_DEV_VERSION="${VERSION_MAJOR}.$((VERSION_MINOR + 1)).0.dev0"
    if [[ "$HEAD_VERSION" != "$EXPECTED_DEV_VERSION" ]]; then
        echo "Error: the commit after resume tag v${VERSION} has version '${HEAD_VERSION:-missing}'."
        echo "       Expected the release script's next cycle '${EXPECTED_DEV_VERSION}'."
        exit 1
    fi
    echo "==> v${VERSION} already released (HEAD on ${HEAD_VERSION}); re-pushing git state."
    git push
    git push origin "refs/tags/v${VERSION}"
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
# A version heading alone is not enough: leaving real entries under Unreleased
# publishes code with incomplete notes. When the conventional heading exists,
# require the target version to be the next release section and the gap between
# them to contain only whitespace.
if grep -qE '^## \[Unreleased\]' CHANGELOG.md; then
    NEXT_CHANGELOG_VERSION=$(awk '
        /^## \[Unreleased\]/ { seen = 1; next }
        seen && /^## \[/ { print; exit }
    ' CHANGELOG.md | sed -E 's/^## \[([^]]+)\].*/\1/')
    if [[ "$NEXT_CHANGELOG_VERSION" != "$VERSION" ]]; then
        echo "Error: CHANGELOG.md must place '${VERSION}' immediately after [Unreleased]."
        exit 1
    fi
    UNRELEASED_CONTENT=$(awk '
        /^## \[Unreleased\]/ { inside = 1; next }
        inside && /^## \[/ { exit }
        inside && NF { print }
    ' CHANGELOG.md)
    if [[ -n "$UNRELEASED_CONTENT" ]]; then
        echo "Error: CHANGELOG.md [Unreleased] still contains entries. Move them into [${VERSION}]."
        exit 1
    fi
fi
echo "    CHANGELOG.md documents ${VERSION}."

# --- Ensure GitHub Actions CI is green ---
# Filter on the CI workflow: other workflows on main (e.g. Dependabot Updates)
# can fail for reasons unrelated to code health and must not block a release.
echo "==> Checking GitHub Actions status..."
# A resumed release has already created its version-only commit and tag. The
# code-bearing parent is the commit whose CI authorized the release; a fresh
# release checks HEAD directly. Query by commit so a green run for the previous
# main revision can never win a race with the current run appearing in GitHub.
if [[ "$TAG_EXISTS" == "1" || "$RELEASE_COMMIT_EXISTS" == "1" ]]; then
    CI_SHA="$RELEASE_PARENT_SHA"
else
    CI_SHA=$(git rev-parse HEAD)
fi
CI_STATUS=$(gh run list --workflow CI --branch main --commit "$CI_SHA" --limit 1 \
    --json status,conclusion,headSha --jq '.[0] // {}')
CI_CONCLUSION=$(echo "$CI_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('conclusion',''))")
CI_STATE=$(echo "$CI_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
CI_HEAD_SHA=$(echo "$CI_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('headSha',''))")
if [[ -z "$CI_HEAD_SHA" ]]; then
    echo "Error: no CI run found for commit ${CI_SHA}. Wait for GitHub Actions to start."
    exit 1
fi
if [[ "$CI_HEAD_SHA" != "$CI_SHA" ]]; then
    echo "Error: CI reported commit ${CI_HEAD_SHA}, expected ${CI_SHA}."
    exit 1
fi
if [[ "$CI_STATE" != "completed" ]]; then
    echo "Error: CI for ${CI_SHA} is still '${CI_STATE}'. Wait for it to finish."
    exit 1
fi
if [[ "$CI_CONCLUSION" != "success" ]]; then
    echo "Error: latest CI run on main concluded with '${CI_CONCLUSION}'. Fix CI before releasing."
    echo "       See: gh run list --workflow CI --branch main --limit 1"
    exit 1
fi
echo "    CI is green for ${CI_SHA}."

echo "==> Releasing v${VERSION}"

# --- Run type checker ---
echo "==> Running ty..."
uv run ty check src/roomkit/
echo "    ty passed."

# --- Check model catalogs against upstream ---
# The offline catalogs in providers/*/models.py are the one thing the test
# suite structurally cannot validate: a catalog that has fallen a lineup behind
# is still internally consistent, so every test passes while the library ships
# ids the vendor has retired. This is the only moment that reliably catches it.
#
# Runs before any mutation. Findings stop the release; an unreachable mirror
# (exit 2) only warns, because blocking a release on someone else's outage
# trades one problem for a worse one.
if [[ -n "${SKIP_MODEL_CHECK:-}" ]]; then
    echo "==> Skipping model catalog check (SKIP_MODEL_CHECK set)."
else
    echo "==> Checking model catalogs..."
    set +e
    uv run python scripts/check_models.py
    MODEL_CHECK_STATUS=$?
    set -e
    if [[ $MODEL_CHECK_STATUS -eq 1 ]]; then
        echo "Error: model catalogs are out of date (see above)."
        echo "       Update providers/*/models.py, or re-run with SKIP_MODEL_CHECK=1"
        echo "       if this release must go out before the catalogs are refreshed."
        exit 1
    elif [[ $MODEL_CHECK_STATUS -ne 0 ]]; then
        echo "    Warning: could not verify model catalogs — continuing."
    fi
fi

# --- Bump version in source ---
if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' "s/^__version__ = .*/__version__ = \"${VERSION}\"/" src/roomkit/_version.py
else
    sed -i "s/^__version__ = .*/__version__ = \"${VERSION}\"/" src/roomkit/_version.py
fi
echo "    Updated src/roomkit/_version.py"

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
git add src/roomkit/_version.py
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
git push
git push origin "refs/tags/v${VERSION}"
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
    PYPI_TOKEN=$(python3 -c 'import configparser, sys; c = configparser.ConfigParser(); c.read(sys.argv[1]); print(c.get("pypi", "password"))' "$HOME/.pypirc")
    # Keep the credential out of argv: command lines are visible to other
    # processes on many systems, while uv supports dedicated environment vars.
    UV_PUBLISH_USERNAME=__token__ UV_PUBLISH_PASSWORD="$PYPI_TOKEN" \
        uv publish --check-url "$PYPI_INDEX" "${DIST_FILES[@]}"
    unset PYPI_TOKEN
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
