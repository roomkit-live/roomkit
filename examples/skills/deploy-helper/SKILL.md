---
name: deploy-helper
description: Check deployment status and roll back releases via the ops scripts
license: MIT
---

# Deploy Helper Skill

This skill drives deployments through its bundled scripts. Never describe a
deployment as done from memory -- run the script and report its output.

1. Run `check-status.sh` before and after any deployment action.
2. To roll back, run `rollback.sh` with the target version as `--version`.
3. Report the raw script output to the user along with your reading of it.

This skill is useless without script execution: if the host has not
configured a script executor, there is nothing it can do.
