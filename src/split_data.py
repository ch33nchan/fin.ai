import argparse
import json
import random
from pathlib import Path

def split_data(input_file: Path, output_dir: Path, train_split: float):
    """Splits a JSONL file into train and validation sets."""
    with input_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)

    split_idx = int(len(lines) * train_split)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with val_path.open("w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"Split {len(lines)} records into:")
    print(f"  - {len(train_lines)} training records at {train_path}")
    print(f"  - {len(val_lines)} validation records at {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Split a JSONL dataset into train and validation sets.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save train.jsonl and val.jsonl.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Proportion of data to use for training (0.0 to 1.0).")
    args = parser.parse_args()

    split_data(
        input_file=Path(args.input),
        output_dir=Path(args.output_dir),
        train_split=args.train_split,
    )

if __name__ == "__main__":
    main()
