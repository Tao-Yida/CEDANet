"""
This file will loop all the subfolders in the "bbox_batch_1" folder,
and then output a json file that contain metadata for all segmentation masks.

The combined metadata will be used for the following application:
(for entering the metadata in to the database)
    - https://github.com/MultiX-Amsterdam/ijmond-camera-monitor

The "bbox_batch_1" folder should contain the output after running the `sample_masks.py` script.

Usage:
python get_all_metadata.py --root_folder bbox_batch_example --output_file combined_metadata.json
python get_all_metadata.py --root_folder bbox_batch_2 --output_file segmentation_dataset_1.json
"""

import os
import json
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, default="bbox_batch_example")
    parser.add_argument("--output_file", type=str, default="combined_metadata.json")
    opt = parser.parse_args()
    return opt

# Parse arguments
opt = args_parser()
root_folder = opt.root_folder # root folder where the data is stored
output_file = opt.output_file # output file name

# List to store combined data
combined_data = []

# Loop through the folder structure
for video_folder in os.listdir(root_folder):
    video_folder_path = os.path.join(root_folder, video_folder)

    # Load video.json file
    video_json_path = os.path.join(video_folder_path, "video.json")
    if not os.path.exists(video_json_path):
        continue
    with open(video_json_path, "r") as f:
        video_data = json.load(f)
    video_id = video_data.get("id")
    video_timestamp = video_data.get("start_time")
    number_of_frames = video_data.get("number_of_frames")

    # Loop through frame folders inside each video folder
    for frame_folder in os.listdir(video_folder_path):
        frame_folder_path = os.path.join(video_folder_path, frame_folder)

        # Check if the frame folder is a directory
        if not os.path.isdir(frame_folder_path):
            continue

        # Load frame metadata
        frame_json_path = os.path.join(frame_folder_path, "frame_metadata.json")
        if os.path.exists(frame_json_path):
            with open(frame_json_path, "r") as f:
                frame_metadata = json.load(f)
        frame_number = frame_metadata.get("frame_numer")
        frame_file_name = frame_metadata.get("frame_file_name")

        # Loop through mask folders inside each frame folder
        for mask_folder in os.listdir(frame_folder_path):
            mask_folder_path = os.path.join(frame_folder_path, mask_folder)

            # Check if the mask folder is a directory
            if not os.path.isdir(mask_folder_path):
                continue

            # Path to metadata.json file
            metadata_json_path = os.path.join(mask_folder_path, "metadata.json")

            # Check if metadata.json exists
            if os.path.exists(metadata_json_path):
                with open(metadata_json_path, "r") as f:
                    metadata = json.load(f)

                # Prepare the combined data entry
                # Note that the frame_timestamp here uses the one from the video
                # Notice that we have three levels here:
                # - The first level is the panorama, such as the original video on https://breathecam.multix.io/
                # - The second level is the video frame, which could be a frame of a video that is cropped from the panorama, or just the panorama itself
                # - The third level is the segmentation image, which could be an image that is cropped from the video frame, or just the video frame itself
                # The reason for such setup is for flexibility
                data_entry = {
                    "mask_file_name": metadata.get("mask_file_name"),  # file name of the segmentation mask
                    "crop_file_name": metadata.get("crop_file_name"), # file name of the segmentation image
                    "bbox_file_name": metadata.get("bbox_file_name"), # file name of the bounding box on top of the image
                    "mask_file_directory": mask_folder_path + "/", # directory to the segmentation mask and image files
                    "frame_timestamp": int(video_timestamp), # timestamp of the video frame
                    "video_id": video_id, # ID of the video on IJmondCAM https://ijmondcam.multix.io/
                    "boxes": metadata.get("boxes"), # the bounding box location relative to the video frame
                    "image_width": metadata.get("image_width"), # width of the video frame
                    "image_height": metadata.get("image_height"), # height of the video frame
                    "relative_boxes": metadata.get("relative_boxes"), # the bounding box location relative to the segmentation image
                    "cropped_width": metadata.get("cropped_width"), # width of the cropped image for segmentation
                    "cropped_height": metadata.get("cropped_height"), # height of the cropped image for segmentation
                    "frame_number": frame_number, # the frame number in the original video
                    "frame_file_name": frame_file_name, # file name of the video frame
                    "frame_file_directory": frame_folder_path + "/", # directory to the video frame
                    "x_image": metadata.get("x_image"), # x coordinate of the top left corner of the segmentation image relative to the video frame
                    "y_image": metadata.get("y_image"), # y coordinate of the top left corner of the segmentation image relative to the video frame
                    "number_of_frames": number_of_frames, # number of frames in the original video
                }

                # Append the entry to the list
                combined_data.append(data_entry)

# Save combined data to a JSON file
with open(output_file, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"Combined metadata has been written to {output_file}")
