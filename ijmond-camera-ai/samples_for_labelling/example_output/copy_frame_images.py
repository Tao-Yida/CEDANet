"""
Copy frame images from the folder that is created by `samples_masks.py` to a new folder.
The new folder can be uploaded to Roboflow for labeling.
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def find_json_files(root_dir):
    """Find all frame_metadata.json files in the directory structure."""
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "frame_metadata.json":
                json_files.append(os.path.join(dirpath, filename))
    return json_files

def copy_png_files(json_files, destination_dir):
    """Copy PNG files specified in the JSON files to the destination directory."""
    os.makedirs(destination_dir, exist_ok=True)

    copied_files = 0
    errors = 0

    for json_file in json_files:
        try:
            # Get the directory containing the JSON file
            json_dir = os.path.dirname(json_file)
            base_dir = os.path.dirname(os.path.dirname(json_file))

            # Read the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract PNG file information
            png_filename = data.get('frame_file_name')
            png_filepath = data.get('frame_file_path')

            if not png_filename or not png_filepath:
                print(f"Warning: Missing filename or filepath in {json_file}")
                errors += 1
                continue

            # Construct source path (in same directory as the JSON)
            source_path = os.path.join(json_dir, png_filename)

            # Alternative source path (using frame_file_path)
            alt_source_path = os.path.join(base_dir, png_filepath, png_filename)
            alt_source_path2 = os.path.join(base_dir, png_filename)

            # Destination path
            dest_path = os.path.join(destination_dir, png_filename)

            # Try to copy the file from the primary or alternative locations
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_files += 1
                print(f"Copied: {png_filename}")
            elif os.path.exists(alt_source_path):
                shutil.copy2(alt_source_path, dest_path)
                copied_files += 1
                print(f"Copied from alt path: {png_filename}")
            elif os.path.exists(alt_source_path2):
                shutil.copy2(alt_source_path2, dest_path)
                copied_files += 1
                print(f"Copied from alt path 2: {png_filename}")
            else:
                print(f"Error: Could not find {png_filename} in expected locations")
                print(f"  - Tried: {source_path}")
                print(f"  - Tried: {alt_source_path}")
                print(f"  - Tried: {alt_source_path2}")
                errors += 1

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            errors += 1

    return copied_files, errors

def main():
    parser = argparse.ArgumentParser(description="Copy PNG files referenced in frame_metadata.json files")
    parser.add_argument("source_dir", help="Source directory containing the folder structure")
    parser.add_argument("destination_dir", help="Destination directory for the PNG files")
    args = parser.parse_args()

    print(f"Scanning for JSON files in {args.source_dir}...")
    json_files = find_json_files(args.source_dir)
    print(f"Found {len(json_files)} frame_metadata.json files")

    print(f"Copying PNG files to {args.destination_dir}...")
    copied, errors = copy_png_files(json_files, args.destination_dir)

    print(f"\nSummary:")
    print(f"  - {copied} files copied successfully")
    print(f"  - {errors} errors encountered")

if __name__ == "__main__":
    main()
