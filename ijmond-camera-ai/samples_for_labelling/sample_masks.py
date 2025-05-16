from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from operator import itemgetter
from utils import create_sub_images, load_json, load_pickle
import cv2
import json
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--mask_folder", type=str, default="masks")
    parser.add_argument("--input_folder", type=str, default="frames")
    parser.add_argument("--output_folder", type=str, default="bbox_reduced")
    parser.add_argument("--store_bbox", type=bool, default=True)
    parser.add_argument("--feature_folder", type=str, default="data/features")
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--num_elements", type=int, default=1, help="Number of elements to select from each cluster")
    parser.add_argument("--video_metadata", type=str, default="metadata_ijmond_jan_22_2024.json")
    opt = parser.parse_args()
    return opt


def labels_selection(selected_frames, mask_folder_fullpath, input_folder_fullpath, output_folder, store_bbox, vid):
    """
    This function receives a list of selected frames and creates sub-images for each frame.
    It also stores the bounding boxes in a json file.
    Args:
        selected_frames: A list of selected frames.
        mask_folder_fullpath: The full path to the mask folder.
        input_folder_fullpath: The full path to the input folder.
        output_folder: The full path to the output folder.
        store_bbox: A boolean to store the bounding boxes.
        vid: The video name.
    Returns:
        num_sum_images: The number of sub-images.
    """
    num_sum_images = 0
    for file in selected_frames:
        # Note that the frame number starts from 1
        frame_num = int(file.split(".")[0].split("_")[-1])
        image_rgb = cv2.imread(os.path.join(input_folder_fullpath, vid, file.split(".")[0] + ".jpg"))
        sub_images, sub_images_with_bbox, bbox, sub_masks = create_sub_images(
                os.path.join(mask_folder_fullpath, vid, file),
                image_rgb,
                use_full_size=True)
        for i, sub_image in enumerate(sub_images):
            # Store the cropped frame
            bbox_folder_name = vid + "-" + str(frame_num) + "-" + str(i)
            output_vid_folder = os.path.join(output_folder, vid, str(frame_num), bbox_folder_name)
            if not os.path.exists(output_vid_folder):
                os.makedirs(output_vid_folder)
            crop_file_name = "crop.png"
            cv2.imwrite(os.path.join(output_vid_folder, crop_file_name), sub_image)
            # Store the cropped mask
            mask_file_name = "mask.png"
            cv2.imwrite(os.path.join(output_vid_folder, mask_file_name), sub_masks[i])
            if store_bbox:
                bbox_file_name = "crop_with_bbox.png"
                cv2.imwrite(os.path.join(output_vid_folder, bbox_file_name), sub_images_with_bbox[i])
            # Store the metadata
            with open(os.path.join(output_vid_folder, "metadata.json"), "w") as f:
                b = bbox[i]
                b["mask_file_path"] = os.path.join(vid, str(frame_num), bbox_folder_name)
                b["crop_file_name"] = crop_file_name
                b["mask_file_name"] = mask_file_name
                if store_bbox:
                    b["bbox_file_name"] = bbox_file_name
                json.dump(b, f)
        num_sum_images += len(sub_images)
        if len(sub_images) > 0:
            # Store the non-cropped frame and the metadata
            output_frame_folder = os.path.join(output_folder, vid, str(frame_num))
            if not os.path.exists(output_frame_folder):
                os.makedirs(output_frame_folder)
            frame_file_name = vid + "-" + str(frame_num) + ".png"
            cv2.imwrite(os.path.join(output_frame_folder, frame_file_name), image_rgb)
            frame_metadata = {
                "frame_numer": frame_num,
                "frame_file_name": frame_file_name,
                "frame_file_path": os.path.join(vid, str(frame_num))
            }
            with open(os.path.join(output_frame_folder, "frame_metadata.json"), "w") as f:
                json.dump(frame_metadata, f)

    if num_sum_images == 0:
        print(f"No sub-images were created for the video {vid}.")

    return num_sum_images


if __name__ == "__main__":
    opt = args_parser()
    data_folder = opt.data_folder
    mask_folder = opt.mask_folder
    input_folder = opt.input_folder
    output_folder = opt.output_folder
    store_bbox = opt.store_bbox
    mask_folder_fullpath = os.path.join(data_folder, mask_folder)
    input_folder_fullpath = os.path.join(data_folder, input_folder)
    feature_folder = opt.feature_folder
    num_clusters = opt.num_clusters
    num_elements = opt.num_elements
    total_samples = 0 # the number of boxes
    total_num_frames = 0 # the number of frames
    debug = False # the flag for debugging
    video_metadata = load_json(opt.video_metadata)
    video_names = ["_1jFnujWn50-0", "zl6ckY2YM8c-2", "zOt2vMuYLx4-0"]
    video_lookup = {v["file_name"]: v for v in video_metadata}

    for vid in os.listdir(feature_folder):
        # For every video, we will select the closest elements to the clusters' center
        vid_name = vid.split("_output")[0]
        if debug and vid_name not in video_names:
             continue
        print("="*20)
        print(vid_name)
        vid_path = os.path.join(feature_folder, vid)
        features = load_pickle(vid_path)

        # Prepare features
        features_values = [v for k, v in features.items()]
        features_values = np.array(features_values).reshape(len(features),-1)
        number_of_frames = features_values.shape[0]
        scaler = StandardScaler()
        features_values = scaler.fit_transform(features_values)

        # Cluster the features using KMeans
        model = KMeans(n_clusters=num_clusters, random_state=42)
        model.fit(features_values)
        cluster_centers = model.cluster_centers_
        total_selected_elements = []
        for i, center in enumerate(cluster_centers):
            distances = np.linalg.norm(features_values - center, axis=1)
            # Take the indices of the num_elements (e.g., 3) smallest distances
            idxes = np.argsort(distances)
            getter = itemgetter(*idxes[:num_elements])
            selected_elements = getter(list(features.keys()))
            if num_elements == 1:
                selected_elements = [selected_elements]
            total_selected_elements.extend(selected_elements)

        """
        # Cluster the features using DBSCAN
        model = DBSCAN(eps=70, min_samples=4)
        labels = model.fit_predict(features_values)
        # Get unique cluster labels (ignore noise label, which is -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        # Compute cluster centers and density for each cluster
        cluster_centers = {}
        cluster_density = {}
        for label in unique_labels:
            cluster_points = features_values[labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers[label] = center
            cluster_density[label] = cluster_points.shape[0]
        # If there are more clusters than num_clusters, select the num_clusters densest clusters
        print("Number of clusters: %d" % len(unique_labels))
        if len(unique_labels) > num_clusters:
            # Sort clusters by density in descending order and select top m labels
            sorted_labels = sorted(unique_labels, key=lambda x: cluster_density[x], reverse=True)
            selected_labels = sorted_labels[:num_clusters]
        else:
            selected_labels = list(unique_labels)
        # Select the closest feature to the cluster center
        total_selected_elements = []
        frame_file_names = list(features.keys())
        for label in selected_labels:
            cluster_idx = (labels == label)
            cluster_points = features_values[cluster_idx]
            cluster_keys = [frame for frame, flag in zip(frame_file_names, cluster_idx) if flag]
            center = cluster_centers[label]
            # Compute Euclidean distances from the cluster center
            distances = np.linalg.norm(cluster_points - center, axis=1)
            # Take the indices of the num_elements (e.g., 3) smallest distances
            idxes = np.argsort(distances)
            getter = itemgetter(*idxes[:num_elements])
            selected_elements = getter(cluster_keys)
            if num_elements == 1:
                selected_elements = [selected_elements]
            total_selected_elements.extend(selected_elements)
        """

        print(total_selected_elements)
        num_sum_images = labels_selection(
                total_selected_elements,
                mask_folder_fullpath,
                input_folder_fullpath,
                output_folder,
                store_bbox,
                vid_name)

        # Store the metadata of the video in a json file
        if num_sum_images != 0:
            with open(os.path.join(output_folder, vid_name, "video.json"), "w") as f:
                d = video_lookup[vid_name]
                d["number_of_frames"] = number_of_frames
                json.dump(d, f)

        total_samples += num_sum_images
        total_num_frames += len(total_selected_elements)

    print("="*30)
    print(f"Total samples: {total_samples}")
    print(f"Total number of frames: {total_num_frames}")
