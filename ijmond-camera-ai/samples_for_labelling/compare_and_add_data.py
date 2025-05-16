'''
This script is used to create new training instanses to extend the training set.
The new instances are created by selecting the instances that have not already selected.
After the selection, the new instances are labelled using the oravle evaluator and then added to the training set.
'''
import os

def traverse_folder(folder):
    '''
    Traverse the folder and return the list of images
    '''
    images = []
    
    for video in os.listdir(folder):
        for frame in os.listdir(os.path.join(folder, video)):
            if frame.endswith(".json"):
                continue
            images.append(os.path.join(video, frame))
    return images

if __name__ == "__main__":
    images_folder_A = "/data/BVM/clean_code/samples_for_labelling/bbox" # Data folder that inlcudes a big set of images
    images_folder_B = "/data/BVM/clean_code/samples_for_labelling/bbox_reduced" # Images that have been used in the training set before
    images_folder_A_B = "/data/BVM/clean_code/samples_for_labelling/bbox_extra" # A - B: keep the data that has not been used in image set that is stored (and used) in "bbox_reduced"
    new_images_folder = "/data/BVM/clean_code/samples_for_labelling/bbox_extra"

    frames_A = traverse_folder(images_folder_A)
    frames_B = traverse_folder(images_folder_B)
    frames_A_B = list(set(frames_A) - set(frames_B)) # Take only the frames that are not in the used set
    print(len(frames_A_B), len(frames_A), len(frames_B))

    # Copy all the data from frames_A_B to the new folder
    os.makedirs(new_images_folder, exist_ok=True)
    for frame in frames_A_B:
        os.makedirs(os.path.join(new_images_folder, frame.split("/")[0]), exist_ok=True)
        os.system(f"cp -r {os.path.join(images_folder_A, frame)} {os.path.join(new_images_folder, frame.split("/")[0])}")
    
    print("Done!")