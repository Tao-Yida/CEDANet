import cv2
import os
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, default="data/videos")
    parser.add_argument('--output_folder', type=str, default="data/frames")
    opt = parser.parse_args()
    return opt

def extract_frames(video_path, output_folder):
    """
    Extract frames from a video file and save them as individual image files.
    Args:   
        video_path: path to the video file
        output_folder: path to the folder where the frames will be saved
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read and save frames
    frame_count = 0
    print(f"Extracting frames from {video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        # print(f"Frame {frame_count} extracted.")
    
    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = arg_parse()

    # Extract frames from each video
    for vid in os.listdir(opt.video_folder):
        extract_frames(os.path.join(opt.video_folder,vid), os.path.join(opt.output_folder, vid.split('.')[0]))
