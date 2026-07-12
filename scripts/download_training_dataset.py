from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a versioned Kaggle dataset")
    parser.add_argument("handle")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(args.handle, output_dir=str(args.output))
    print(path, flush=True)


if __name__ == "__main__":
    main()
