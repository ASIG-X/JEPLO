"""Utility to prune old checkpoints in the logs directory.

The script walks all run folders under the provided logs root, keeps the
numerically last checkpoint in each run, and deletes the rest after a single
user confirmation. A run is any directory that contains checkpoint-like files
with a trailing integer (for example, ``model_1234.pt``).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

CHECKPOINT_REGEX = re.compile(r"^(?P<stem>.+?)_(?P<step>\d+)\.(?P<ext>pt|pth|ckpt)$")


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    step: int


@dataclass(frozen=True)
class RunCleanupPlan:
    to_delete: List[Checkpoint]
    to_keep: Checkpoint


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove older checkpoints, keeping only the last one per run.")
    parser.add_argument(
        "logs_root",
        nargs="?",
        default=Path(__file__).resolve().parent / "logs",
        type=Path,
        help="Root folder that contains run subdirectories (default: ./logs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the confirmation prompt and delete immediately.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing any files.",
    )
    return parser.parse_args(argv)


def find_runs(logs_root: Path) -> Dict[Path, List[Checkpoint]]:
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root does not exist: {logs_root}")

    runs: Dict[Path, List[Checkpoint]] = {}

    for checkpoint_file in logs_root.rglob("*"):
        if not checkpoint_file.is_file():
            continue

        match = CHECKPOINT_REGEX.match(checkpoint_file.name)
        if match is None:
            continue

        run_dir = checkpoint_file.parent
        step = int(match.group("step"))
        runs.setdefault(run_dir, []).append(Checkpoint(path=checkpoint_file, step=step))

    return runs


def summarize_deletions(runs: Dict[Path, List[Checkpoint]]) -> Dict[Path, RunCleanupPlan]:
    plans: Dict[Path, RunCleanupPlan] = {}

    for run_dir, checkpoints in runs.items():
        if len(checkpoints) <= 1:
            continue

        checkpoints.sort(key=lambda item: item.step)
        keep = checkpoints[-1]
        to_delete = checkpoints[:-1]
        plans[run_dir] = RunCleanupPlan(to_delete=to_delete, to_keep=keep)

    return plans


def print_summary(plans: Dict[Path, RunCleanupPlan]) -> None:
    total_files = sum(len(plan.to_delete) for plan in plans.values())
    if total_files == 0:
        print("No checkpoints need to be removed; each run already keeps at most one file.")
        return

    print("The following checkpoints will be removed (keeping the largest step per run):")
    for run_dir in sorted(plans.keys()):
        plan = plans[run_dir]
        try:
            display_dir = run_dir.relative_to(Path.cwd())
        except ValueError:
            display_dir = run_dir
        print(f"  Run: {display_dir}")
        print(f"    Files to delete: {len(plan.to_delete)}")
        print(f"    Highest step kept: {plan.to_keep.step} ({plan.to_keep.path.name})")

    print(f"Total files to delete: {total_files}")


def request_confirmation(force: bool) -> bool:
    if force:
        return True

    reply = input("Proceed with deleting the files listed above? [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def delete_checkpoints(plans: Dict[Path, RunCleanupPlan], dry_run: bool) -> None:
    if dry_run:
        print("Dry run enabled; no files were deleted.")
        return

    for plan in plans.values():
        for checkpoint in plan.to_delete:
            checkpoint.path.unlink(missing_ok=True)

    print("Deletion complete.")


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    logs_root = args.logs_root.resolve()

    try:
        runs = find_runs(logs_root)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    plans = summarize_deletions(runs)

    if sum(len(plan.to_delete) for plan in plans.values()) == 0:
        print("Nothing to do.")
        return 0

    print_summary(plans)

    if not request_confirmation(args.force):
        print("Aborted by user.")
        return 0

    delete_checkpoints(plans, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
