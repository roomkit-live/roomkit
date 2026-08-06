---
name: roomkit-release
description: Cut a roomkit release end to end — pre-flight state, version choice, full CHANGELOG review and promote, docs/derived artifacts, local gates, CI green on HEAD, `make release VERSION=x.y.z`, then verify tag + GitHub Release + SBOM + PyPI + next dev cycle all landed. Use for /roomkit-release, "release roomkit", "sors la x.y.z", "on publie une version".
---

# Release roomkit

Take roomkit from "main looks ready" to a published, consistent release: PyPI, the
`vX.Y.Z` tag, the GitHub Release with its SBOM, the CHANGELOG entry, and main
reopened on the next dev cycle — all agreeing with each other.

`scripts/release.sh` (behind `make release VERSION=x.y.z`) is the authority on the
irreversible part. **This skill is everything the script cannot check**: is the
CHANGELOG actually right, is the version number the right one, are the docs and
derived artifacts in sync, is HEAD pushed and CI-green — plus a verification pass
afterwards. Never re-implement what the script does; never work around a gate it
raises.

Version argument, if given (`/roomkit-release 0.43.0`), is a proposal to
sanity-check, not an order — step 2 still runs.

## Guardrails (read once, hold for the whole run)

- **Narrate in French.** Everything the user reads — step headers, findings,
  questions, the final report — in French. Everything written into the repo
  (CHANGELOG prose, commit messages, code) stays in **English**, matching
  `git log`.
- **Never pipe a gate through `tail`/`head`.** `make all 2>&1 | tail -20` reports
  *tail's* exit code; a green "exit 0" then proves nothing. Run gates unpiped and
  read the real exit status.
- **Never edit `src/roomkit/_version.py` by hand.** The script owns it.
- **Never hand-tag, hand-publish, or hand-create the GitHub Release** to "finish"
  a failed run. The script is idempotent and resumable — fix the cause, re-run the
  same `make release VERSION=…`. See *Resume states* at the end.
- **Stop and ask** before: the version number (step 2), promoting the CHANGELOG
  (step 3), and launching `make release` (step 7, the tail of which is
  irreversible). Everything else is autonomous.
- **The working tree must be empty in `git status --porcelain`, untracked files
  included** — the script only tolerates a dirty `src/roomkit/_version.py`.
  `.claude/` is gitignored, so this skill's own file never counts.
- **Never fake completion.** A run ends *published + verified*, or *stopped* with
  the exact blocker and the state main was left in.

## Step-by-step tracking (do this first)

The user wants to see where the release is at every moment.

1. Load the task tools once: `ToolSearch("select:TaskCreate,TaskUpdate,TaskList")`.
2. Create the nine tasks below **before step 1**, in order:
   `1. État des lieux` · `2. Numéro de version` · `3. Revue du CHANGELOG` ·
   `4. Docs et artefacts dérivés` · `5. Portes locales` · `6. Push + CI verte` ·
   `7. make release` · `8. Vérification post-release` · `9. Suites`
3. Exactly one task `in_progress` at a time; mark it `completed` before starting
   the next.
4. Close every step with one line in the transcript, in French, in this shape:

   ```
   ✅ Étape 3/9 — CHANGELOG : 11 commits depuis v0.41.4, 9 couverts, 2 chore ignorés. Promu en [0.42.0] — 2026-08-06.
   ⚠️  Étape 4/9 — Docs : llms-full.txt régénéré (docs/c7 avait bougé), à committer.
   ⛔ Étape 6/9 — CI : run 1234 rouge sur test_voice_pipeline. Release arrêtée.
   ```

   A skipped step gets a line too, saying why it was skipped.

---

## 1. État des lieux

Read-only. Establish the truth before touching anything:

```bash
git fetch --tags --prune
git status --porcelain                     # must be empty
git rev-parse --abbrev-ref HEAD            # must be main
git log --oneline origin/main..HEAD        # must be empty — nothing unpushed
grep '^__version__' src/roomkit/_version.py
git describe --tags --abbrev=0             # last tag
gh release list --limit 5
curl -s https://pypi.org/pypi/roomkit/json | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"
```

Report a compact table: branch · tree · HEAD sha · dev version · last tag · last
GitHub Release · latest on PyPI.

**Flag any disagreement between those last four** — a tag with no GitHub Release,
PyPI ahead of the tags, a `## [x.y.z]` section in the CHANGELOG with no tag. This
repo has a history of partial releases; an inconsistency found here is a decision
for the user, not something to silently release on top of.

Blockers: dirty tree, not on main, unpushed commits. Stop and report.

## 2. Numéro de version

```bash
LAST=$(git describe --tags --abbrev=0)
git log --oneline "$LAST"..HEAD
git diff --stat "$LAST"..HEAD -- src/ pyproject.toml
```

Classify the range: breaking public-API change, new feature, fix only. The
convention in this 0.x history is **minor for features or breaking changes, patch
for a fix-only cut** (0.41.1 was the first patch and the flow handled it
unchanged: the next dev cycle is always `{minor+1}.0.dev0`, so a patch cut from an
open minor cycle returns main to the same dev version).

Then verify the candidate is free:

```bash
git rev-parse -q --verify "refs/tags/vX.Y.Z"          # must not exist
curl -sfL "https://pypi.org/pypi/roomkit/X.Y.Z/json" -o /dev/null && echo ON-PYPI
grep -n "^## \[X.Y.Z\]" CHANGELOG.md                   # only the pending promote
```

Propose the number with the one-line rationale (`AskUserQuestion` when there is a
real choice) and get an explicit confirmation. PEP 440 only: `1.2.3` or `1.2.3rc1`
— the script rejects `1.2.3-rc.1`.

## 3. Revue du CHANGELOG

The step the user cares most about. `CHANGELOG.md` is **hand-maintained** — the
script only greps for `## [X.Y.Z]`, it cannot tell you the content is right.

**a. Coverage.** List every commit in `$LAST..HEAD` and map each to a bullet in
`## [Unreleased]`. Report the unmapped ones explicitly. Only `chore`, `test`,
internal refactors and CI plumbing may legitimately be absent — a `feat` or `fix`
that touched `src/` and has no entry is a defect: write it, or have the user say
it is deliberate.

**b. Content.** Read the whole section and check:

- Keep a Changelog headings, in order: `### Added`, `### Changed`,
  `### Deprecated`, `### Removed`, `### Fixed`, `### Security`.
- **The house voice**: entries are prose that state the motivating case and what
  changed for the reader — the bold lead sentence, then why the workaround did not
  exist, then the contract. Read the `## [0.42.0]` entry as the model. Commit
  subjects pasted as bullets are not acceptable.
- Breaking changes marked as this file marks them — `### Changed — BREAKING` or a
  `**BREAKING — …**` lead — with the migration spelled out.
- New public API mentioned, and actually exported from `roomkit/__init__.py`.
- An optional-dependency floor raised in `pyproject.toml` gets its own entry (the
  0.37.0 lesson: code adopted a `buzzkit` 0.1.4-only API while the extra still
  declared `>=0.1.3` → TypeError for anyone resolving the floor; a lockfile bump
  does not protect installs).
- Nothing internal or private leaks into a public file.

**c. Promote.** Show the user the exact edit, then apply it:

- `## [Unreleased]` stays in place, empty; insert `## [X.Y.Z] — YYYY-MM-DD` below
  it (em dash `—`, today's date) and move the content under it.
- Footer links at the bottom of the file:
  ```
  [Unreleased]: https://github.com/roomkit-live/roomkit/compare/vX.Y.Z...HEAD
  [X.Y.Z]: https://github.com/roomkit-live/roomkit/compare/vPREV...vX.Y.Z
  ```

If the promote was already done in an earlier session (a `## [X.Y.Z] — DATE`
section with no tag), do not redo it: verify the date is still right, verify the
two footer links, and say so.

## 4. Docs et artefacts dérivés

```bash
make llms-full                 # rebuilds llms-full.txt from docs/c7/
git status --porcelain llms-full.txt
make docs                      # strict MkDocs build of ../roomkit-docs
```

- A dirty `llms-full.txt` means `docs/c7/` drifted — commit the regenerated file
  with the release prep in step 6.
- `make docs` builds `../roomkit-docs/mkdocs.yml` from roomkit's own env (its
  `mkdocstrings` directives import the real package). Broken internal links report
  as INFO and do **not** fail `--strict` — grep the output for
  `does not contain an anchor` too.
- `../roomkit-docs` must be clean and pushed. Per the project CLAUDE.md, a new
  feature owes a guide in `docs/guides/`, a `features.md` section, a `mkdocs.yml`
  nav entry, and a runnable `examples/` script. Missing docs is a finding to raise,
  not a release blocker to decide alone.
- Check `README.md` / `llms.txt` for counts or minimum versions this release makes
  stale.

## 5. Portes locales

Unpiped, in this order — cheapest signal first:

Unpiped, cheapest signal first. **Pick the lane first** — the suite runs three
times across a release (here, on CI at step 6, and inside `release.sh` at step 7),
and a doc-only cut does not need all three.

Decide with:

```bash
git diff --name-only HEAD                      # what this release prep is about to commit
git log --oneline origin/main..HEAD            # must be empty — HEAD is pushed
gh run list --workflow CI --branch main --commit "$(git rev-parse HEAD)" --limit 1 \
    --json status,conclusion --jq '.[0]'
```

**Fast lane** — only when *both* hold: every pending path is documentation
(`CHANGELOG.md`, `llms-full.txt`, `llms.txt`, `README.md`, `docs/`, `*.md` under
`examples/`), **and** HEAD already has a `completed` + `success` CI run. Then:

```bash
make lint typecheck security
make check-models
```

**Full lane** — anything else, i.e. a pending path under `src/`, `tests/`,
`scripts/`, `examples/*.py`, `pyproject.toml`, `uv.lock` or `.github/`, or a HEAD
whose CI is not green yet:

```bash
make all            # lint + typecheck + security + test
make check-models   # providers/*/models.py vs upstream
make audit          # pip-audit: core gates, extras only report
```

State in the step's closing line which lane ran and why.

What the fast lane skips is covered twice over: CI re-runs the whole suite and
both `pip-audit` passes on the release-prep commit at step 6, and `release.sh`
re-runs `ty` + `pytest` at step 7 — a Markdown edit cannot break either
differently. What it keeps is what is *not* covered elsewhere in time to matter:
lint/format (a stray formatting error would fail CI at step 6 and cost a whole
cycle) and `check-models`, the one thing the test suite structurally cannot catch
— a catalog a lineup behind is still self-consistent, and it kills the release at
step 7 instead. Exit 1 = stale catalogs, fix `providers/*/models.py`; exit 2 =
mirror unreachable, only a warning.

## 6. Push + CI verte

Commit the release prep (CHANGELOG promote, regenerated `llms-full.txt`, doc
fixes) with an English `docs(changelog):`-style message, staging **explicit
paths**, and **push it**.

**This is the step where the old habit is now wrong.** `release.sh` gates on the
CI run for the *exact HEAD sha*:

```bash
gh run list --workflow CI --branch main --commit "$(git rev-parse HEAD)" --limit 1
```

So the previously-used "commit the changelog locally and let the release push
carry it" pattern now aborts with `no CI run found for commit …`. HEAD must be
pushed, and its CI run must be `completed` + `success`, before `make release`.

```bash
gh run list --workflow CI --branch main --limit 1        # get the run id
gh run watch <id> --exit-status                          # blocks until it settles
```

Red CI stops the release. Note that only the `CI` workflow counts — a failing
*Dependabot Updates* run on main is not a blocker (the guard filters on
`--workflow CI` for exactly that reason).

## 7. `make release VERSION=X.Y.Z`

Ask for an explicit go: everything from the PyPI upload on is irreversible.

```bash
make release VERSION=X.Y.Z
```

Long-running (~7k tests, then build, SBOM, push, GitHub Release, publish) — run it
in the background and report each `==>` milestone as it lands. The script's own
sequence, so nothing surprises:

PyPI-exists check → CHANGELOG entry → CI green on HEAD → `ty` → model catalogs →
bump `_version.py` → `pytest` → `uv build` → SBOM (`cyclonedx-bom` pinned) →
commit `Bump version to X.Y.Z` → tag `vX.Y.Z` → **push git state** → GitHub
Release with the SBOM attached → **publish to PyPI last** → commit + push
`Begin {minor+1}.0.dev0`.

`SKIP_MODEL_CHECK=1 make release …` exists for an unreachable mirror — use it only
with the user's agreement, and say so in the final report.

On failure: read the error, fix the cause, re-run the identical command. Do not
improvise around it.

## 8. Vérification post-release

Prove all five faces agree — do not take the script's last line as proof:

```bash
git ls-remote --tags origin | grep "refs/tags/vX.Y.Z$"
gh release view "vX.Y.Z" --json isDraft,assets --jq '{draft:.isDraft, assets:[.assets[].name]}'
curl -sfL "https://pypi.org/pypi/roomkit/X.Y.Z/json" -o /dev/null && echo "PyPI OK"
grep '^__version__' src/roomkit/_version.py     # {minor+1}.0.dev0
git status --porcelain && git log --oneline origin/main..HEAD   # both empty
grep -n "^## \[X.Y.Z\]" CHANGELOG.md
```

The GitHub Release must carry the `roomkit-X.Y.Z.cdx.json` SBOM asset. Report the
final table in French: tag ✅ · Release + SBOM ✅ · PyPI ✅ · main sur `A.B.0.dev0`
✅ · CHANGELOG ✅. Any ✗ is stated plainly with what is missing.

## 9. Suites

- **Downstream consumers that pin roomkit**: if the release ships an API they
  already call, their floor bump is *required*, not cosmetic — bump the pin and
  relock in the same commit. Gotcha: right after a publish,
  `uv lock --upgrade-package roomkit` fails with "only roomkit<=PREV is available"
  (stale index cache); add `--refresh-package roomkit` and it resolves at once.
  Deployment is a separate, explicit step — never assumed done.
- Record the release wherever this repo's release history is kept: version, date,
  what shipped, the lane and flow variant used, and anything new learned. Append to
  the existing record rather than starting a new one.

---

## Resume states (when a run died mid-release)

`release.sh` recognises exactly these, and re-running the same command is the
correct move in all of them:

| State left behind | What the script does on re-run |
|---|---|
| Dirty `_version.py` only, nothing committed | Resumes from the bump; every other dirty file is refused |
| Version-only commit, no tag | Uses its code-bearing parent for the CI gate, tags, continues |
| Tag exists, push or publish failed | Skips build/commit/tag, retries push, Release, publish (`--check-url` skips files already uploaded) |
| Released and dev-cycle commit made, push failed | Re-pushes the commit and the tag, exits |

Two invariants it enforces and you must not defeat: a release tag points at a
**version-only** commit, and the tag must be HEAD or HEAD's direct parent — so
never fold other files into a release commit, and never let unrelated commits pile
on top of a half-finished release.
