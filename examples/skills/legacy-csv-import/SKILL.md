---
name: legacy-csv-import
description: Import CSV exports from the legacy v1 system (semicolon-separated, latin-1, DD/MM/YYYY dates)
license: MIT
---

# Legacy CSV Import Skill

The legacy v1 system exports CSV files that break naive parsers. When helping
a user import one, apply these rules:

1. **Encoding** -- files are latin-1, never UTF-8. Decode accordingly or
   accented names will mojibake.
2. **Delimiter** -- semicolon (`;`), because values contain commas.
3. **Dates** -- `DD/MM/YYYY`. Parse explicitly; locale-guessing swaps day and
   month for the first twelve days of each month.
4. **Sentinels** -- the string `NULL` and the value `-999` both mean "no
   value" and must map to None, not survive as data.
5. **Header drift** -- v1 renamed columns twice. Consult the column-map
   reference before assuming a header means what it says.

Always read the column-map reference first, then produce parsing code that
handles all five rules and show the user two or three parsed sample rows so
they can confirm the mapping before a full import.
