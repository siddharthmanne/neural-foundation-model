#!/usr/bin/env python3
"""
Shard THINGS dataset images into .tar files.

Input structure:
    images/<class>/<class>_01s.jpg
    images/<class>/<class>_02s.jpg
    ...

Output structure:
    shards/shard_000.tar   (contains 000000001.jpg, 000000001.txt, 000000002.jpg, ...)
    shards/shard_001.tar   (contains 000000501.jpg, 000000501.txt, ...)
    ...

Each .txt file contains the original filename (e.g. "apple_01s.jpg").

Usage:
    python shard_things.py --images_dir ./images --output_dir ./shards --images_per_shard 500
"""

import argparse
import io
import tarfile
from pathlib import Path


def collect_images(images_dir: Path) -> list[Path]:
    """Collect all .jpg images from images/<class>/<class>_NNs.jpg structure."""
    images = []
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images.extend(sorted(class_dir.glob("*.jpg")))
    return images


def make_txt_entry(tar: tarfile.TarFile, arcname: str, content: str) -> None:
    """Add an in-memory text file to an open tar archive."""
    encoded = content.encode()
    info = tarfile.TarInfo(name=arcname)
    info.size = len(encoded)
    tar.addfile(info, io.BytesIO(encoded))


def shard_images(
    images_dir: Path,
    output_dir: Path,
    images_per_shard: int,
    dry_run: bool = False,
) -> None:
    images = collect_images(images_dir)
    total = len(images)

    if total == 0:
        print("No .jpg images found. Check your --images_dir path.")
        return

    num_shards = (total + images_per_shard - 1) // images_per_shard
    print(f"Found {total} images → {num_shards} shards of up to {images_per_shard} each.")

    if dry_run:
        print("\n[DRY RUN] No files will be written.\n")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx in range(num_shards):
        start = shard_idx * images_per_shard
        end = min(start + images_per_shard, total)
        shard_slice = images[start:end]

        tar_path = output_dir / f"shard_{shard_idx:03d}.tar"

        if dry_run:
            print(f"[shard_{shard_idx:03d}.tar]")
            for i, src in enumerate(shard_slice):
                img_num = start + i + 1
                print(f"  {src}  →  {img_num:09d}.jpg + {img_num:09d}.txt ({src.name})")
            continue

        with tarfile.open(tar_path, "w") as tar:
            for i, src_path in enumerate(shard_slice):
                img_num = start + i + 1
                base = f"{img_num:09d}"

                tar.add(src_path, arcname=f"{base}.jpg")
                make_txt_entry(tar, f"{base}.txt", src_path.name)

        print(f"  Wrote {tar_path}  ({len(shard_slice)} images)")

    if not dry_run:
        print(f"\nDone! Shards written to: {output_dir.resolve()}")
        print(f"  Tar files : shard_000.tar … shard_{num_shards - 1:03d}.tar")
        print(f"  Filenames : 000000001.jpg/txt … {total:09d}.jpg/txt")


def main():
    parser = argparse.ArgumentParser(
        description="Shard THINGS dataset images into .tar files."
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("images"),
        help="Root directory containing class subdirectories (default: ./images)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("shards"),
        help="Where shard .tar files will be created (default: ./shards)",
    )
    parser.add_argument(
        "--images_per_shard",
        type=int,
        default=500,
        help="Number of images per shard (default: 500)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned operations without writing any files",
    )

    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"Error: images directory not found: {args.images_dir.resolve()}")
        return

    shard_images(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        images_per_shard=args.images_per_shard,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()