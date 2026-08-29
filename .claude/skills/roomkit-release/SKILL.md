---
name: roomkit-release
description: Cut a roomkit release end to end — pre-flight state, version choice, full CHANGELOG review and promote, docs/derived artifacts, local gates, CI green on HEAD, `make release VERSION=x.y.z`, then verify tag + GitHub Release + SBOM + PyPI + next dev cycle all landed. Every decision is parked in Luge's HITL inbox, never asked in the terminal. Use for /roomkit-release, "release roomkit", "cut a release", "publish x.y.z".
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

- **Everything in English** — the narration, the final report, and everything
  written into the repo (CHANGELOG prose, commit messages, code), matching
  `git log`. The one exception is what lands on Luge: a HITL ask is written in
  the language of the person who answers it, French by default.
- **Never pipe a gate through `tail`/`head`.** `make all 2>&1 | tail -20` reports
  *tail's* exit code; a green "exit 0" then proves nothing. Run gates unpiped and
  read the real exit status.
- **Never edit `src/roomkit/_version.py` by hand.** The script owns it.
- **Never hand-tag, hand-publish, or hand-create the GitHub Release** to "finish"
  a failed run. The script is idempotent and resumable — fix the cause, re-run the
  same `make release VERSION=…`. See *Resume states* at the end.
- **Three decisions are the human's, and every one of them is asked through
  Luge's HITL inbox** — the version number (step 2), promoting the CHANGELOG
  (step 3), and launching `make release` (step 7, the tail of which is
  irreversible). Never `AskUserQuestion`, never a terminal prompt: the protocol
  is *Asking through Luge* below. Everything else is autonomous.
- **The working tree must be empty in `git status --porcelain`, untracked files
  included** — the script only tolerates a dirty `src/roomkit/_version.py`. This
  skill's own file is tracked (`.claude/skills/` is the one exception to the
  ignored `.claude/*`), so an edit to it is a change like any other: commit it
  before releasing.
- **Never fake completion.** A run ends *published + verified*, *parked* on a
  named HITL task, or *stopped* with the exact blocker — the last two always
  with the state main was left in.

## Step-by-step tracking (do this first)

The run must show where it stands at every moment.

1. Load the task tools once: `ToolSearch("select:TaskCreate,TaskUpdate,TaskList")`.
2. Create the nine tasks below **before step 1**, in order:
   `1. State of play` · `2. Version number` · `3. CHANGELOG review` ·
   `4. Docs and derived artifacts` · `5. Local gates` · `6. Push + CI green` ·
   `7. make release` · `8. Post-release verification` · `9. Follow-ups`
3. Exactly one task `in_progress` at a time; mark it `completed` before starting
   the next.
4. Close every step with one line in the transcript, in this shape:

   ```
   ✅ Step 3/9 — CHANGELOG: 11 commits since v0.41.4, 9 covered, 2 chore skipped. Promoted to [0.42.0] — 2026-08-06.
   ⚠️  Step 4/9 — Docs: llms-full.txt regenerated (docs/c7 had drifted), to commit.
   ⛔ Step 6/9 — CI: run 1234 red on test_voice_pipeline. Release stopped.
   ⏸ Step 7/9 — Go: HITL 3f2a… unanswered after 5h. Parked — prep pushed, CI green, nothing released.
   ```

   A skipped step gets a line too, saying why it was skipped; a step parked on
   an unanswered ask gets a ⏸ line naming the task id.

## Asking through Luge (HITL)

Every question this run puts to a human is parked in Luge's HITL inbox — not in
this terminal. A release is as likely to be started by a scheduler or another
agent as by someone sitting in front of it, and a question asked in a terminal
nobody is watching is a run that hangs on nothing. The inbox is where the human
looks: it rings, it keeps a deadline, and the answer is readable back by a
later run.

```bash
export LUGE_CLI_JSON=1                  # once — never the ellipsized human view
luge-cli --profile prod auth show       # the inbox lives on prod, like the team channel
```

**Three asks, two kinds.** The version (step 2) is a `choice` — the candidates
enumerate, the human taps. The CHANGELOG promote (step 3) and the go (step 7)
are `approval`s — approve, or reject with a comment. **An approval body carries
no question mark and offers no alternative**: `approve`/`reject` cannot answer
"OK, or would you rather…", and the only way out is a rejection that reads as a
plan that was wrong. Anything still open is a `text` or `choice` ask first; the
approval comes after, once the answer is in.

The body is the positional argument and the inbox renders it as markdown —
write the whole ask there, structured, in a file under the session's scratchpad
directory, and skip `--context` (it appends a second block and splits the ask in
two):

```bash
cat > "$SCRATCH/ask-version.md" <<'EOF'
<the ask — each step below says what its body carries>
EOF
luge-cli --profile prod hitl create "$(cat "$SCRATCH/ask-version.md")" \
    -t "roomkit release — version" --kind choice -o "0.62.0 (minor)" -o "0.61.1 (patch)" \
    --asked-by "roomkit-release" --wait 900
```

`--wait` returns two outcomes; **branch on `answered`, never on the exit code**
(an elapsed wait exits 0 on purpose):

- **`answered: true`** — read `task.response`: a `choice` answers
  `{"value", "label", "comment"?}`, an `approval` `{"decision":
  "approve"|"reject", "comment"?}`. **A rejection's comment is the
  instruction**: apply it, then park a fresh ask for the corrected version —
  never proceed on a rejected ask, never treat the comment as optional.
- **`answered: false`** — nobody was there yet. Not the end of the run: write
  one ⏸ line in the transcript (task id, what the answer unblocks), then keep
  waiting on the same task in half-hour polls:

  ```bash
  for i in 1 2 3 4 5 6 7 8 9 10; do
      luge-cli --profile prod hitl show "$TASK_ID" --wait 1800   # answered: true → continue
  done
  ```

  An elapsed poll is silent — no re-ask, no second task. **Ten polls elapsed
  (five hours): park and stop.** Close the step with its ⏸ line, state exactly
  what main holds (prep committed? pushed? CI green?), and end the run without
  proceeding on an assumption. The question stays live in the inbox.

**A run that resumes picks the parked ask up, never re-asks it.** Step 1 lists
the inbox; a pending or answered `roomkit-release` task is the previous run's
state — read it with `hitl show <id>` and continue at the step it belongs to.
Never treat a timeout as an answer, never answer for the human (the CLI refuses
it server-side anyway).

---

## 1. State of play

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
luge-cli --profile prod hitl list      # a roomkit-release task here = a run to resume
```

**A `roomkit-release` task in the inbox is a previous run's state**, not a fresh
start: pending → `hitl show <id> --wait 1800` and resume at that step; answered →
read the response and continue from there. Never park a second ask for a
question that is already parked.

Report a compact table: branch · tree · HEAD sha · dev version · last tag · last
GitHub Release · latest on PyPI.

**Any disagreement between those last four is a blocker** — a tag with no GitHub
Release, PyPI ahead of the tags, a `## [x.y.z]` section with no tag. Report it and
let the user decide; do not release on top of it.

Blockers: dirty tree, not on main, unpushed commits. Stop and report.

## 2. Version number

```bash
LAST=$(git describe --tags --abbrev=0)
git log --oneline "$LAST"..HEAD
git diff --stat "$LAST"..HEAD -- src/ pyproject.toml
```

Classify the range: breaking public-API change, new feature, fix only. In this 0.x
line **features and breaking changes take a minor, a fix-only cut takes a patch**.
Either way the script reopens main on `{minor+1}.0.dev0`, so a patch cut from an
open minor cycle returns main to the dev version it already had.

Then verify the candidate is free:

```bash
git rev-parse -q --verify "refs/tags/vX.Y.Z"          # must not exist
curl -sfL "https://pypi.org/pypi/roomkit/X.Y.Z/json" -o /dev/null && echo ON-PYPI
grep -n "^## \[X.Y.Z\]" CHANGELOG.md                   # only the pending promote
```

Park the number as a `choice` ask — the candidate first, the other bump second.
The body carries the commit list, the classification in one line, and the three
availability checks with their results. The answer's `value` is the version; a
different number in its `comment` is the instruction — verify that one is free
the same way, and continue with it. PEP 440 only: `1.2.3` or `1.2.3rc1` — the
script rejects `1.2.3-rc.1`.

## 3. CHANGELOG review

`CHANGELOG.md` is **hand-maintained**; the script only greps for `## [X.Y.Z]` and
cannot judge what is under it.

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
  exist, then the contract. Read the most recent released entry as the model.
  Commit subjects pasted as bullets are not acceptable.
- Breaking changes marked as this file marks them — `### Changed — BREAKING` or a
  `**BREAKING — …**` lead — with the migration spelled out.
- New public API mentioned, and actually exported from `roomkit/__init__.py`.
- A raised optional-dependency floor gets its own entry — and check the reverse:
  code that adopted a new API of an optional dep without raising that extra's
  floor in `pyproject.toml` ships an AttributeError/TypeError to anyone resolving
  the old floor. A lockfile bump does not protect installs.
- Nothing internal or private leaks into a public file.

**c. Promote.** Park the exact edit as an `approval` ask; apply it once approved:

- `## [Unreleased]` stays in place, empty; insert `## [X.Y.Z] — YYYY-MM-DD` below
  it (em dash `—`, today's date) and move the content under it.
- Footer links at the bottom of the file:
  ```
  [Unreleased]: https://github.com/roomkit-live/roomkit/compare/vX.Y.Z...HEAD
  [X.Y.Z]: https://github.com/roomkit-live/roomkit/compare/vPREV...vX.Y.Z
  ```

The approval body is the coverage table from (a) — every commit and the bullet
that covers it, or why it has none — the findings from (b) with what was
rewritten, and the exact heading and footer lines that will land. Rejected: the
comment says what to rewrite; rewrite, re-park. Approved: apply the edit.

If the promote was already done in an earlier session (a `## [X.Y.Z] — DATE`
section with no tag), do not redo it: verify the date is still right, verify the
two footer links, and say so.

## 4. Docs and derived artifacts

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

## 5. Local gates

Unpiped. **Pick the lane first**, with:

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

State in the step's closing line which lane ran.

The fast lane is sound because what it drops runs twice more anyway: CI covers the
suite and both `pip-audit` passes on the prep commit at step 6, and `release.sh`
re-runs `ty` + `pytest` at step 7. It keeps the two things those cover too late —
lint/format, which would fail CI at step 6 and cost a full cycle, and
`check-models`, which no test can replace (a catalog a release behind is still
self-consistent) and which aborts `make release` at step 7. `check-models` exit 1
= stale catalogs, fix `providers/*/models.py`; exit 2 = mirror unreachable, warning
only.

## 6. Push + CI green

Commit the release prep (CHANGELOG promote, regenerated `llms-full.txt`, doc
fixes) with a `docs(changelog):`-style message, staging **explicit paths**, and
**push it**.

`release.sh` resolves the CI run for the *exact HEAD sha*:

```bash
gh run list --workflow CI --branch main --commit "$(git rev-parse HEAD)" --limit 1
```

HEAD must therefore be pushed, and its run `completed` + `success`, before
`make release`. A prep commit left local aborts the release with
`no CI run found for commit …`.

```bash
gh run list --workflow CI --branch main --limit 1        # get the run id
gh run watch <id> --exit-status                          # blocks until it settles
```

Red CI stops the release. Only the `CI` workflow counts — the guard filters on
`--workflow CI`, so a failing *Dependabot Updates* run on main is not a blocker.

## 7. `make release VERSION=X.Y.Z`

The go is an `approval` ask — everything from the PyPI upload on is
irreversible. The body states the version, HEAD's sha and the CI run that is
green on it, the lane that ran at step 5, the CHANGELOG section the script will
lift, and that approving launches PyPI + tag + GitHub Release in one command.
Rejected: stop — the comment is the instruction, and nothing has been published.

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

## 8. Post-release verification

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
final table: tag ✅ · Release + SBOM ✅ · PyPI ✅ · main on `A.B.0.dev0` ✅ ·
CHANGELOG ✅. Any ✗ is stated plainly with what is missing.

## 9. Follow-ups

- **Downstream consumers that pin roomkit**: if the release ships an API they
  already call, bump the pin and relock in the same commit — required, not
  cosmetic. Right after a publish, `uv lock --upgrade-package roomkit` fails with
  "only roomkit<=PREV is available" (stale index cache); add
  `--refresh-package roomkit`. Deploying is a separate, explicit step — never
  report it as done.
- Record the release wherever this repo keeps its release history: version, date,
  what shipped.

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
