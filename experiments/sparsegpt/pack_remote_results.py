import argparse
import tarfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack a SparseGPT result directory into a .tar.gz archive.")
    parser.add_argument("run_dir", help="Run directory to archive.")
    parser.add_argument("archive_path", nargs="?", default="", help="Optional output archive path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    archive_path = Path(args.archive_path).resolve() if args.archive_path else run_dir.with_suffix(".tar.gz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)

    print(f"[OK] archive={archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
