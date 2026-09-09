---
name: quote-policy
description: Synthetic benchmark quote workflow with a reference, inventory tool and calculation script.
allowed_tools: inventory
---

For a benchmark quote, follow this workflow:
1. Read `rules.md` with read_skill_reference.
2. Call inventory to obtain quantity and unit price. This tool is gated by this skill.
3. Run `total.py` with run_skill_script, passing quantity and unit_price as strings.
4. Answer with the script's total and the policy marker from the reference.

Reuse this skill on follow-up quotes. These rules apply only in this room.
