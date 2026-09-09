"""Synthetic benchmark fixture: deterministic quote calculation."""

from __future__ import annotations

import json
import sys

arguments = json.loads(sys.argv[1])
total = int(arguments["quantity"]) * int(arguments["unit_price"]) + 7
sys.stdout.write(json.dumps({"total": total}))
