"""CLI for the residual-policy data contract audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import audit_dataset, load_manifest, save_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-name", default="data_center_general_depth_n5_sock"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = load_manifest(args.data_root, args.manifest, args.dataset_name)
    reports = audit_dataset(specs)
    output = save_audit(reports, args.output)
    n_errors = sum(len(report.errors) for report in reports)
    n_usable = sum(report.usable_for_residual for report in reports)
    print(f"wrote {output}")
    print(f"episodes={len(reports)} usable={n_usable} errors={n_errors}")
    for report in reports:
        if report.errors:
            print(f"{report.split}/{report.episode}: {'; '.join(report.errors)}")
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
