"""Put leefrag-v2 (and the repo root, for `leefrag`) on sys.path.

Lets the scripts run without an editable install: `python scripts/<name>.py`.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)          # leefrag-v2/
_ROOT = os.path.dirname(_PKG)          # repo root (has leefrag/)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
