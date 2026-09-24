"""Audit NIH ChestX-ray14 metadata, HDF5 coverage, and prepared splits.

Reports aggregate counts only; it never prints image names or patient IDs.
Run from the project root, for example:

    python scripts/audit_nih_data.py \
      --metadata data/raw/nih/Data_Entry_2017.csv \
      --hdf5 data/raw/nih/nih_images.h5 \
      --splits data/processed_s4ml

Use --views all to compare the HDF5 against every metadata row. The default
PA/AP scope matches the project's frontal-view training configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {"Image Index", "Patient ID", "View Position"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Full NIH Data_Entry_2017.csv (not a prefiltered subset)")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--splits", type=Path,
                        help="Directory containing train.csv, val.csv, and test.csv")
    parser.add_argument("--views", choices=("frontal", "all"), default="frontal",
                        help="Expected source view scope (default: PA/AP frontal views)")
    parser.add_argument("--verify-pixels", action="store_true",
                        help="Read every HDF5 image array to detect damaged compressed chunks")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        parser.error("h5py is required; install project dependencies with `pip install -r requirements.txt`")

    if not args.metadata.is_file():
        parser.error(f"metadata file not found: {args.metadata}")
    if not args.hdf5.is_file():
        parser.error(f"HDF5 file not found: {args.hdf5}")

    metadata = pd.read_csv(args.metadata)
    missing_columns = REQUIRED - set(metadata.columns)
    if missing_columns:
        parser.error(f"metadata is missing required columns: {sorted(missing_columns)}")

    print("NIH source metadata")
    print(f"  rows: {len(metadata):,}")
    print(f"  unique image names: {metadata['Image Index'].nunique():,}")
    print(f"  duplicate image rows: {metadata['Image Index'].duplicated().sum():,}")
    print(f"  unique patients: {metadata['Patient ID'].nunique():,}")
    print("  view counts:")
    for view, count in metadata["View Position"].fillna("(missing)").value_counts().sort_index().items():
        print(f"    {view}: {count:,}")

    expected_df = metadata
    if args.views == "frontal":
        expected_df = metadata[metadata["View Position"].isin(["PA", "AP"])]
    expected = set(expected_df["Image Index"].dropna().astype(str))

    try:
        with h5py.File(args.hdf5, "r") as h5:
            if "images" not in h5 or not isinstance(h5["images"], h5py.Group):
                parser.error("HDF5 has no 'images' group")
            image_group = h5["images"]
            actual = set(image_group.keys())
            bad_shapes = sum(
                1 for key in actual
                if getattr(image_group[key], "shape", None) != (256, 256)
            )
            unreadable_pixels = 0
            if args.verify_pixels:
                for key in actual:
                    try:
                        image_group[key][()]
                    except Exception:
                        unreadable_pixels += 1
    except (OSError, RuntimeError) as exc:
        parser.error(f"HDF5 cannot be read (possibly truncated/corrupt): {exc}")

    missing = expected - actual
    extra = actual - expected
    print(f"\nHDF5 images group: {len(actual):,} images")
    print(f"  expected metadata rows ({args.views} scope): {len(expected):,}")
    print(f"  expected images absent from HDF5: {len(missing):,}")
    print(f"  HDF5 keys not in expected metadata scope: {len(extra):,}")
    print(f"  datasets not shaped 256x256: {bad_shapes:,}")
    if args.verify_pixels:
        print(f"  image arrays that failed to read: {unreadable_pixels:,}")

    exit_code = 0
    if missing or extra or bad_shapes or (args.verify_pixels and unreadable_pixels):
        exit_code = 1

    if args.splits:
        frames: dict[str, pd.DataFrame] = {}
        for name in ("train", "val", "test"):
            path = args.splits / f"{name}.csv"
            if not path.is_file():
                parser.error(f"split file not found: {path}")
            frame = pd.read_csv(path)
            absent = REQUIRED - set(frame.columns)
            if absent:
                parser.error(f"{path} is missing required columns: {sorted(absent)}")
            frames[name] = frame

        print("\nPrepared splits")
        all_split_images: set[str] = set()
        patient_sets: dict[str, set[str]] = {}
        for name, frame in frames.items():
            names = set(frame["Image Index"].dropna().astype(str))
            patient_sets[name] = set(frame["Patient ID"].dropna().astype(str))
            absent_h5 = names - actual
            print(f"  {name}: {len(frame):,} rows, {frame['Patient ID'].nunique():,} patients, "
                  f"{len(absent_h5):,} images absent from HDF5, "
                  f"{frame['Image Index'].duplicated().sum():,} duplicate rows")
            if all_split_images & names:
                print("  ERROR: duplicate image appears in multiple splits")
                exit_code = 1
            all_split_images |= names
            if absent_h5 or frame["Image Index"].duplicated().any():
                exit_code = 1

        split_overlap = sum(
            len(patient_sets[a] & patient_sets[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        )
        omitted = expected - all_split_images
        outside_scope = all_split_images - expected
        print(f"  cross-split patient overlap pairs: {split_overlap:,}")
        print(f"  expected images omitted from all splits: {len(omitted):,}")
        print(f"  split images outside selected source scope: {len(outside_scope):,}")
        if split_overlap or omitted or outside_scope:
            exit_code = 1

    print("\nAUDIT: " + ("PASS" if exit_code == 0 else "CHECK REQUIRED"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
