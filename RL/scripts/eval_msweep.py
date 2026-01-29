from __future__ import annotations

import os
import sys

# Ensure project root on sys.path for direct execution
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.validation import run_m_sweep


def main():
    results = run_m_sweep(lambda cfg: None, lambda cfg: None, {})
    for m, metrics in results.items():
        print(f"m={m}: {metrics}")


if __name__ == '__main__':
    main()
