#!/usr/bin/env python
import argparse, subprocess, sys, os, shutil
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Download a Kaggle dataset via CLI")
    p.add_argument("--dataset", required=True, help="e.g., harshalhonde/coinmarketcap-cryptocurrency-dataset-2023")
    p.add_argument("-o", "--output", default="data/raw", help="Output directory")
    args = p.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    if not shutil.which("kaggle"):
        print("ERROR: Kaggle CLI not found. Install with `pip install kaggle` and place kaggle.json in ~/.kaggle/")
        sys.exit(1)

    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
