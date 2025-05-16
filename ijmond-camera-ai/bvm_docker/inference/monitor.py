import os
import time
import threading
import queue
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import warnings
from datetime import datetime
import argparse
warnings.filterwarnings("ignore")

class ImageHandler(FileSystemEventHandler):
    def __init__(self, image_queue):
        self.image_queue = image_queue

    def on_created(self, event):
        # Check if the new file is an image file
        if not event.is_directory and event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f"New image detected: {event.src_path}")
            self.image_queue.put(event.src_path)

def process_image(image_queue, output_folder, pretrained_weights):
    while True:
        try:
            # Get an image path from the queue
            image_path = image_queue.get(timeout=1)
            print(f"Processing image: {image_path}")
            # Run the processing script and suppress warnings from subprocess
            subprocess.run([
                'python3', 'process_image.py',
                '--img_path', image_path,
                '--output_mask', os.path.join(output_folder, image_path.split('/')[-2]),
                '--pretrained_weights', pretrained_weights
            ], check=True, stderr=subprocess.DEVNULL)
            # Remove from the queue after processing
            image_queue.task_done()
        except queue.Empty:
            # No more images to process, can wait or check periodically
            time.sleep(1)
        except subprocess.CalledProcessError as e:
            print(f"Failed to process image {image_path}: {e}")

def enqueue_images_from_directory(directory_path, image_queue, output_folder):
    date_subdir = os.path.basename(directory_path)
    # Recursively add images to the queue from a directory
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                
                # Determine the expected output mask filename
                output_path = os.path.join(output_folder, date_subdir, filename)
                
                # Check if the mask file already exists
                if not os.path.exists(output_path):
                    print(f"Queueing image: {file_path}")
                    image_queue.put(file_path)
                else:
                    print(f"Skipping image {file_path}, mask already exists.")

def monitor_new_images(directory_path, image_queue):
    event_handler = ImageHandler(image_queue)
    observer = Observer()
    observer.schedule(event_handler, directory_path, recursive=True)
    observer.start()
    print(f"Monitoring directory: {directory_path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

def get_sorted_subdirectories(parent_directory):
    # Get a list of all subdirectories sorted by date
    subdirs = [
        os.path.join(parent_directory, d)
        for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d))
    ]
    # Sort directories by date (assuming format "YYYY-MM-DD")
    subdirs.sort(key=lambda date: datetime.strptime(os.path.basename(date), "%Y-%m-%d"))
    return subdirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default="./bvm_docker/data")
    parser.add_argument('--output_folder', type=str, default="./bvm_docker/outtttt")
    parser.add_argument('--pretrained_weights', type=str, default="./bvm_docker/models/baseline/Model_50_gen.pth")
    opt = parser.parse_args()

    root_directory = opt.img_folder #"/tmpdata"
    output_folder = opt.output_folder #"./bvm_docker/masks"
    pretrained_weights = opt.pretrained_weights #"models/ucnet_trans3_baseline/Model_50_gen.pth"

    # Create a queue to hold image paths
    image_queue = queue.Queue()

    # Get sorted subdirectories and process existing images
    sorted_subdirs = get_sorted_subdirectories(root_directory)
    for subdir in sorted_subdirs:
        print(f"Processing images in directory: {subdir}")
        enqueue_images_from_directory(subdir, image_queue, output_folder)

    # Start a thread to process images from the queue
    processing_thread = threading.Thread(target=process_image, args=(image_queue, output_folder, pretrained_weights))
    processing_thread.daemon = True
    processing_thread.start()

    # Monitor the most recent subdirectory for new images
    if sorted_subdirs:
        most_recent_dir = sorted_subdirs[-1]
        monitor_thread = threading.Thread(target=monitor_new_images, args=(most_recent_dir, image_queue))
        monitor_thread.daemon = True
        monitor_thread.start()

    # Monitor the root directory for new date subdirectories
    try:
        while True:
            # Check for new subdirectories and update monitoring
            new_subdirs = get_sorted_subdirectories(root_directory)
            if len(new_subdirs) != len(sorted_subdirs):
                added_subdirs = set(new_subdirs) - set(sorted_subdirs)
                for new_subdir in sorted(added_subdirs):
                    print(f"NEW subdirectory detected: {new_subdir}")
                    enqueue_images_from_directory(new_subdir, image_queue, output_folder)
                    # Update monitoring to the new most recent subdirectory
                    if new_subdir > most_recent_dir:
                        most_recent_dir = new_subdir
                        # Stop previous monitor thread and start a new one for the new most recent directory
                        monitor_thread.join(timeout=1)  # Wait for the current monitoring thread to stop
                        monitor_thread = threading.Thread(target=monitor_new_images, args=(most_recent_dir, image_queue))
                        monitor_thread.daemon = True
                        monitor_thread.start()
                sorted_subdirs = new_subdirs
            time.sleep(5)  # Check for new subdirectories every 5 seconds
    except KeyboardInterrupt:
        print("Exiting...")
