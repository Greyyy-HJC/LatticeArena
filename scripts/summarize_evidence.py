"""Summarise evidence JSON files as a markdown table (todo F0)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from core.evidence import load_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    rows = []
    for path in args.paths:
        for entry in load_evidence(path):
            rows.append(entry)
    print('| task | test | claim | status | scope |')
    print('| --- | --- | --- | --- | --- |')
    for entry in rows:
        print(f"| {entry['task']} | {entry['test_id']} | {entry['claim']} | {entry['status']} | {entry['scope']} |")
    statuses = {}
    for entry in rows:
        statuses[entry['status']] = statuses.get(entry['status'], 0) + 1
    print()
    print(json.dumps(statuses))


if __name__ == '__main__':
    main()
