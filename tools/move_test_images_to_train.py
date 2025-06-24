#!/usr/bin/env python3
"""
Script to randomly select 50 images from test set and move them to train set.
Ensures that img, gt, and trans files are moved together and remain consistent.
"""

import os
import random
import shutil
from pathlib import Path


def main():
    # Define paths
    base_path = Path("/home/ytao/Thesis/data/ijmond_data")
    test_path = base_path / "test"
    train_path = base_path / "train"

    # Create train directory if it doesn't exist
    train_path.mkdir(exist_ok=True)

    # Create subdirectories in train
    for subdir in ["img", "gt", "trans"]:
        (train_path / subdir).mkdir(exist_ok=True)

    # Get all image files from test/img directory (excluding annotations file)
    img_files = [f for f in os.listdir(test_path / "img") if f.endswith(".jpg") and not f.startswith("_annotations")]

    print(f"Found {len(img_files)} images in test set")

    # Randomly select 50 images
    if len(img_files) < 50:
        print(f"Warning: Only {len(img_files)} images available, selecting all of them")
        selected_files = img_files
    else:
        selected_files = random.sample(img_files, 50)

    print(f"Selected {len(selected_files)} images to move to train set")

    # Move corresponding files
    moved_count = 0
    failed_count = 0

    for img_file in selected_files:
        # Extract base name without extension
        base_name = os.path.splitext(img_file)[0]

        # Define source and destination paths
        files_to_move = [
            ("img", img_file, img_file),  # img: jpg -> jpg
            ("gt", f"{base_name}.png", f"{base_name}.png"),  # gt: png -> png
            ("trans", img_file, img_file),  # trans: jpg -> jpg
        ]

        # Check if all required files exist
        all_exist = True
        for subdir, src_name, _ in files_to_move:
            src_path = test_path / subdir / src_name
            if not src_path.exists():
                print(f"Warning: Missing file {src_path}")
                all_exist = False

        if not all_exist:
            print(f"Skipping {base_name} due to missing files")
            failed_count += 1
            continue

        # Move all files for this image
        try:
            for subdir, src_name, dst_name in files_to_move:
                src_path = test_path / subdir / src_name
                dst_path = train_path / subdir / dst_name

                # Move the file
                shutil.move(str(src_path), str(dst_path))
                print(f"Moved {src_path} -> {dst_path}")

            moved_count += 1
            print(f"Successfully moved image set {base_name} ({moved_count}/{len(selected_files)})")

        except Exception as e:
            print(f"Error moving files for {base_name}: {e}")
            failed_count += 1

    print(f"\nSummary:")
    print(f"Successfully moved: {moved_count} image sets")
    print(f"Failed to move: {failed_count} image sets")

    # Verify the moved files
    print(f"\nVerification:")
    for subdir in ["img", "gt", "trans"]:
        count = len([f for f in os.listdir(train_path / subdir) if f.endswith(".jpg") or f.endswith(".png")])
        print(f"Train {subdir} directory now contains: {count} files")


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(223)
    main()
