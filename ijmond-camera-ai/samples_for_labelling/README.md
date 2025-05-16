# ijmond-camera-ai

### Table of Content
- [Download the videos](#download-videos)
- [Extract frames](#extract-frames)
- [Inference the BVM model](#inference-bvm)
- [Sample masks for labelling](#sample-masks)

## <a name="download-videos"></a>Download videos

The videos to be downloaded are identified by codes 23 and 47, and there should be a total of 481 videos. Each video contains 36 frames. The videos are stored in the `./data/videos` folder.
```sh
python download_videos.py
```

Structure of files system so far:
```sh
└── data # the root folder s
    └── videos # the folder that contains all the downloaded videos
        ├── _1jFnujWn50-0.mp4 # example of video
        └── ...
```

## <a name="extract-frames"></a>Extract frames

In order to process videos and extract the segmentation masks, the frames need to be extracted. This is done using the line below. `video_folder` is the folder containing all the videos, and `output_folder` is where the extracted frames will be stored.
```sh
python preprocessing.py --video_folder data/videos --output_folder data/frames
```

Structure of files system so far:
```sh
└── data # the root folder
    └── videos # the folder that contains all the downloaded videos
        ├── _1jFnujWn50-0.mp4 # example of video
        └── ...
    └── frames # the folder that contains the frames of the videos
        ├── _1jFnujWn50-0 # video folder
            ├── frame_0001.jpg # frame 1
            └── ...
        └── ... # video folders

```

## <a name="inference-bvm"></a>Inference the BVM model

The `trans_bvm` folder contains the [`BVM`](https://github.com/SiyuanYan1/Transmission-BVM/tree/main) model. To run the model, you need to specify the input directory `frames_folder` where the frames of each video are stored, and the output directory `output_folder` where all the masks and features for each frame of the video will be stored. The features will be used in the frame selection phase to select the most representative frames for labeling from each video. If you do not want to store the features, simply remove the `--mode_features` option from the command below.

At the end, two new folders will be created in the data folder: features, which will include the features of the frames of the videos, and masks, which will contain all the masks found by the BVM.
```sh
python trans_bvm/test.py --frames_folder data/frames --output_folder data --mode_features
```

Structure of files system so far:
```sh
└── data # the root folder
    └── videos # the folder that contains all the downloaded videos
        ├── _1jFnujWn50-0.mp4 # example of video
        └── ...
    └── frames # the folder that contains the frames of the videos
        ├── _1jFnujWn50-0 # video folder
            ├── frame_0001.jpg # frame 1
            └── ...
        └── ... # video folders
    └── features # the folder that contains all the feature files
        ├── _1jFnujWn50-0_output.pkl # example of feature file
        └── ...
    └── masks # the folder that contains the masks of the videos
        ├── _1jFnujWn50-0 # video folder
            ├── frame_0001.jpg # mask for frame 1
            └── ...
        └── ... # video folders
```

## <a name="sample-masks"></a>Sample masks for labelling

In order to select the most representative frames for labeling, a clustering algorithm was used. The algorithm employed is KMeans, with a default of 3 clusters, which can be adjusted through the arguments. The features for each frame were used for the clustering process. The selected frames are the top 1 closest to the center of each cluster, resulting in 3 frames in total. After selecting the frames, they can be cropped to a region close to the bounding box of the segmentation mask, or they can just be used directly without cropping. Currently we did not crop the frames further and leave it as an option for simplicity. Once the process is complete, all the data are stored in the `bbox` folder. You can change the data folder, output folder, number of clusters, and the number of elements that should be selected for each cluster.
```sh
python sample_masks.py --data_folder data --output_folder bbox --num_clusters 3 --num_elements 1
```

Structure of files system:
```sh
└── bbox # the root folder that stores all bounding boxes
    └── A9W8G55JucU3 # unique ID that represents the video, can be the video file name (without extension)
        ├── video.json # the video metadata
        └── 0 # frame number
            ├── frame_metadata.json # the frame metadata
            └── A9W8G55JucU3-0-0 # unique ID that represents the cropped region in the frame
                ├── mask.png # the grayscale mask (created by the segmentation model)
                ├── crop.png # the cropped image in the video frame
                └── metadata.json # storing the bounding box information
```

Structure in the `video.json` file below. Check the [video dataset documentation](https://github.com/MultiX-Amsterdam/ijmond-camera-monitor/tree/main/dataset/2024-01-22) for more information about the variables.
```json
{
    "camera_id": 2,
    "file_name": "_1jFnujWn50-0",
    "id": 656, # ID of the video in the IJmondCAM database
    "label_state": 23,
    "label_state_admin": 23,
    "start_time": 1686975462,
    "url_part": "_1jFnujWn50",
    "url_root": "https://ijmondcam.multix.io/videos/",
    "view_id": 0,
    "number_of_frames": 36 # total number of video frames
}
```

Structure in the `frame_metadata.json` file below.
```json
{
    "frame_numer": 21, # the frame number of the image in the original video
    "frame_file_name": "_1jFnujWn50-0-21.png", # file name of the video frame
    "frame_file_path": "_1jFnujWn50-0/21" # path to the video frame in the root folder
}
```

Structure in the `metadata.json` file below. Notice that we have three levels here:
- The first level is the panorama, such as the original video on [BreatheCAM](https://breathecam.multix.io/).
- The second level is the video frame, which could be a frame of a video that is cropped from the panorama, or just the panorama itself.
- The third level is the segmentation image, which could be an image that is cropped from the video frame, or just the video frame itself.
In our case, the cropped image is exactly the original video frame since we did not do any cropping. The reason of not cropping the video frame is due to consistency for labeling data. In the front-end labeling user interface, we need to show the video together with the segmentation image and mask.
```json
{
    "boxes": {
        "x": 804, # y coordinate of the top left corner of the box relative to the video frame
        "y": 828, # y coordinate of the top left corner of the box relative to the video frame
        "w": 72, # width of the box
        "h": 72 # height of the box
    }, # the bounding box location relative to the video frame
    "relative_boxes": {
        "x": 804, # x coordinate of the top left corner of the box relative to the segmentation image
        "y": 828, # y coordinate of the top left corner of the box relative to the segmentation image
        "w": 72, # width of the box
        "h": 72 # height of the box
    }, # the bounding box location relative to the segmentation image
    "image_width": 900, # width of the original video frame
    "image_height": 900, # height of the original video frame
    "cropped_width": 900, # width of the cropped image for segmentation
    "cropped_height": 900, # height of the cropped image for segmentation
    "x_image": 0, # x coordinate of the top left corner of the segmentation image relative to the video frame
    "y_image": 0, # y coordinate of the top left corner of the segmentation image relative to the video frame
    "mask_file_path": "_1jFnujWn50-0/21/_1jFnujWn50-0-21-0", # path to the mask file in the root folder
    "crop_file_name": "crop.png", # file name of the segmentation image
    "mask_file_name": "mask.png", # file name of the segmentation image
    "bbox_file_name": "crop_with_bbox.png" # file name of the bounding box on top of the image
}
```

See the `example_output` folder for examples. After producing the folder structure and the files, run the `get_all_metadata.py` file in the `example_output` folder to produce a file `combined_metadata.json`, which will be used to add the data to the IJmondCAM tool using the [`add_segment.py`](https://github.com/MultiX-Amsterdam/ijmond-camera-monitor/blob/main/back-end/www/add_segment.py) script. Below is the structure of the `combined_metadata.json` file:
```json
[
    {
        "mask_file_name": "mask.png",
        "crop_file_name": "crop.png",
        "bbox_file_name": "crop_with_bbox.png",
        "mask_file_directory": "bbox_batch_1/_1jFnujWn50-0/21/_1jFnujWn50-0-21-0/", # directory to the segmentation mask and image files (with the root folder name and a "/" at the end)
        "frame_timestamp": 1686975462, # timestamp of the video frame (this is copied from the video timestamp)
        "video_id": 656,
        "boxes": {
            "x": 804,
            "y": 828,
            "w": 72,
            "h": 72
        },
        "image_width": 900,
        "image_height": 900,
        "relative_boxes": {
            "x": 804,
            "y": 828,
            "w": 72,
            "h": 72
        },
        "cropped_width": 900,
        "cropped_height": 900,
        "frame_number": 21,
        "frame_file_name": "_1jFnujWn50-0-21.png",
        "frame_file_directory": "bbox_batch_1/_1jFnujWn50-0/21/", # directory to the video frame (with the root folder name and a "/" at the end)
        "x_image": 0,
        "y_image": 0,
        "number_of_frames": 36
    },
    ...
]
```