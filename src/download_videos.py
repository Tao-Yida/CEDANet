"""
This script downloads videos from the dataset and saves their labels with filtering options.

Usage:
python download_videos.py [criteria]

Available filtering criteria:
- all (default): Download all non-poor-quality videos
- high_confidence: Only high confidence samples (gold standard + strong agreement)
- positive_only: Only positive samples (smoke detected)
- negative_only: Only negative samples (no smoke)
- gold_standard: Only gold standard samples (researcher-verified)

Label state meanings:
47: Gold standard positive (researcher labeled positive, gold standard)
32: Gold standard negative (researcher labeled negative, gold standard)
23: Strong positive (two volunteers agree or researcher says smoke)
16: Strong negative (two volunteers agree or researcher says no smoke)
19: Weak positive (volunteers disagree, third volunteer says smoke)
20: Weak negative (volunteers disagree, third volunteer says no smoke)
5: Possibly positive (one volunteer says smoke)
4: Possibly negative (one volunteer says no smoke)
3: Disagreement (two volunteers disagree)
-1: No data/no discord
-2: Poor quality video (always excluded)

Examples:
python download_videos.py                    # Download all good quality videos
python download_videos.py high_confidence    # Download only high confidence samples
python download_videos.py positive_only      # Download only positive samples
python download_videos.py gold_standard      # Download only gold standard samples
"""

import sys
import urllib.request
import os
import json
import csv


def is_file_here(file_path):
    """
    Check if a file exists.

    Parameters
    ----------
    file_path : str
        The file path that we want to check.

    Returns
    -------
    bool
        If the file exists (True) or not (False).
    """
    return os.path.isfile(file_path)


def check_and_create_dir(dir_path):
    """
    Check and create a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        The dictionary path that we want to create.
    """
    if dir_path is None:
        return
    dir_name = os.path.dirname(dir_path)
    if dir_name != "" and not os.path.exists(dir_name):
        try:  # This is used to prevent race conditions during parallel computing
            os.makedirs(dir_name)
        except Exception as ex:
            print(ex)


def should_exclude_video(v):
    """
    Check if a video should be excluded based on label states.

    Parameters
    ----------
    v : dict
        The dictionary with keys and values in the video dataset JSON file.

    Returns
    -------
    bool
        True if video should be excluded, False otherwise.
    """
    # Exclude videos with poor quality (label_state or label_state_admin = -2)
    if v.get("label_state", 0) == -2 or v.get("label_state_admin", 0) == -2:
        return True
    return False


def get_video_url(v):
    """
    Get the video URL.

    Parameters
    ----------
    v : dict
        The dictionary with keys and values in the video dataset JSON file.

    Returns
    -------
    str
        The full URL of the video.
    """
    camera_names = ["hoogovens", "kooksfabriek_1", "kooksfabriek_2"]
    return v["url_root"] + camera_names[v["camera_id"]] + "/" + v["url_part"] + "/" + v["file_name"] + ".mp4"


def filter_by_label_criteria(data_dict, criteria="all"):
    """
    Filter videos based on label criteria.

    Parameters
    ----------
    data_dict : list
        List of video dictionaries.
    criteria : str
        Filtering criteria:
        - "all": Download all non-poor-quality videos (default)
        - "high_confidence": Only high confidence samples (gold standard + strong agreement)
        - "positive_only": Only positive samples (various confidence levels)
        - "negative_only": Only negative samples (various confidence levels)
        - "gold_standard": Only gold standard samples (47, 32)

    Returns
    -------
    list
        Filtered list of video dictionaries.
    """
    filtered_videos = []

    for v in data_dict:
        # Always exclude poor quality videos
        if should_exclude_video(v):
            continue

        ls = v.get("label_state", -1)
        lsa = v.get("label_state_admin", -1)

        if criteria == "all":
            filtered_videos.append(v)
        elif criteria == "high_confidence":
            # Gold standard + strong agreement
            if lsa in [47, 32, 23, 16]:
                filtered_videos.append(v)
        elif criteria == "positive_only":
            # All positive samples
            if lsa in [47, 23] or ls in [47, 23, 19, 5]:
                filtered_videos.append(v)
        elif criteria == "negative_only":
            # All negative samples
            if lsa in [32, 16] or ls in [32, 16, 20, 4]:
                filtered_videos.append(v)
        elif criteria == "gold_standard":
            # Only gold standard samples
            if lsa in [47, 32]:
                filtered_videos.append(v)

    return filtered_videos


def main(argv):
    # Parse command line arguments for filtering criteria
    criteria = "all"  # default
    if len(argv) > 1:
        criteria = argv[1]
        if criteria not in ["all", "high_confidence", "positive_only", "negative_only", "gold_standard"]:
            print(f"Invalid criteria: {criteria}")
            print("Valid options: all, high_confidence, positive_only, negative_only, gold_standard")
            return

    # Specify the path to the JSON file
    json_file_path = "src/samples_for_labelling/metadata_ijmond_jan_22_2024.json"

    # Specify the path that we want to store the videos and create it
    download_path = "data/ijmond_camera/videos/"
    check_and_create_dir(download_path)

    # Specify the path for labels file
    labels_file_path = "data/ijmond_camera/video_labels.csv"
    check_and_create_dir(labels_file_path)

    # Open the file and load its contents into a dictionary
    with open(json_file_path, "r") as json_file:
        data_dict = json.load(json_file)

    print(f"Total videos in original dataset: {len(data_dict)}")
    print(f"Using filtering criteria: {criteria}")

    # Apply filtering criteria
    filtered_videos = filter_by_label_criteria(data_dict, criteria)
    excluded_count = len(data_dict) - len(filtered_videos)

    print(f"Videos after filtering: {len(filtered_videos)}")
    print(f"Excluded videos: {excluded_count}")

    # Count poor quality exclusions separately
    poor_quality_count = sum(1 for v in data_dict if should_exclude_video(v))
    criteria_excluded = excluded_count - poor_quality_count

    print(f"  - Poor quality exclusions: {poor_quality_count}")
    print(f"  - Criteria-based exclusions: {criteria_excluded}")

    # Start to download files
    problem_videos = []
    downloaded_videos = []

    for v in filtered_videos:
        video_url = get_video_url(v)
        file_name = v["file_name"]
        file_path = download_path + file_name + ".mp4"

        if not is_file_here(file_path):
            print(f"Downloading video {file_name} (label_state: {v.get('label_state', -1)}, label_state_admin: {v.get('label_state_admin', -1)})")
            try:
                urllib.request.urlretrieve(video_url, file_path)
                downloaded_videos.append(v)
            except Exception as e:
                print(f"\tError downloading video {file_name}: {e}")
                problem_videos.append(v)
        else:
            print(f"Video {file_name} already exists, skipping download")
            downloaded_videos.append(v)

    # Save labels for all processed videos (including already existing ones)
    print(f"\nSaving labels for {len(filtered_videos)} videos to {labels_file_path}")

    # Create simplified labels file
    with open(labels_file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file_name", "label_state", "label_state_admin"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for v in filtered_videos:
            writer.writerow(
                {
                    "file_name": v["file_name"],
                    "label_state": v.get("label_state", -1),
                    "label_state_admin": v.get("label_state_admin", -1),
                }
            )

    # Print summary
    print(f"\n=== DOWNLOAD SUMMARY ===")
    print(f"Filtering criteria: {criteria}")
    print(f"Total videos in original dataset: {len(data_dict)}")
    print(f"Videos after filtering: {len(filtered_videos)}")
    print(f"Successfully downloaded/existing: {len(downloaded_videos)}")
    print(f"Failed downloads: {len(problem_videos)}")

    # Print label state distribution
    label_state_counts = {}
    label_state_admin_counts = {}

    for v in filtered_videos:
        ls = v.get("label_state", -1)
        lsa = v.get("label_state_admin", -1)

        label_state_counts[ls] = label_state_counts.get(ls, 0) + 1
        label_state_admin_counts[lsa] = label_state_admin_counts.get(lsa, 0) + 1

    print(f"\n=== LABEL STATE DISTRIBUTION ===")
    print("label_state distribution:")
    for state, count in sorted(label_state_counts.items()):
        print(f"  {state}: {count} videos")

    print("label_state_admin distribution:")
    for state, count in sorted(label_state_admin_counts.items()):
        print(f"  {state}: {count} videos")

    # Print errors
    if len(problem_videos) > 0:
        print(f"\n=== FAILED DOWNLOADS ===")
        print("The following videos were not downloaded due to errors:")
        for v in problem_videos:
            print(f"  {v['file_name']} (label_state: {v.get('label_state', -1)}, label_state_admin: {v.get('label_state_admin', -1)})")

    print(f"\nLabels saved to: {labels_file_path}")
    print("DONE")


if __name__ == "__main__":
    main(sys.argv)
