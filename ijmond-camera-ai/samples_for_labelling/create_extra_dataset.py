import os

def find_instances(blacklisted_small_data):
    instances = []
    for blacklisted_instance in blacklisted_small_data:
        frame = blacklisted_instance.split(",")[0].split("/")[-3]
        video = blacklisted_instance.split(",")[0].split("/")[-4]
        path_ = f"{video}/{frame}"
        instances.append(path_)
    return instances

if __name__=="__main__":
    blacklist_small = "/data/BVM/DATA/masks_selection/despoina/whitelisted_instances.list"
    blacklist_extended = "/data/BVM/DATA/masks_selection/despoina_extra_annotations/whitelisted_instances.list"
    output_folder = "/data/BVM/DATA/data/ijmond_data_extended/train_stream"

    with open(blacklist_small, "r") as f:
        blacklisted_small_data = f.readlines()
    
    with open(blacklist_extended, "r") as f:
        blacklisted_extended_data = f.readlines()
    
    small_list = find_instances(blacklisted_small_data)
    extended_list = find_instances(blacklisted_extended_data)
    rest = list(set(extended_list) - set(small_list))

    # _1jFnujWn50-0-5-1_mask.png
    # _1jFnujWn50-0-5-1_crop.png
    input_folder = "/data/BVM/clean_code/samples_for_labelling/bbox"
    for instance in blacklisted_extended_data:
        frame = instance.split(",")[0].split("/")[-3]
        video = instance.split(",")[0].split("/")[-4]
        subframe = instance.split(",")[0].split("/")[-2]
        path_ = f"{video}/{frame}"
        if path_ in rest:
            instance_crop_name = os.path.join(output_folder, "img", instance.split(",")[0].split("/")[-2] + "_crop.png")
            instance_mask_name = os.path.join(output_folder, "gt", instance.split(",")[0].split("/")[-2] + "_mask.png")
            os.system(f"cp -r {input_folder}/{video}/{frame}/{subframe}/crop.png {instance_crop_name}")
            os.system(f"cp -r {input_folder}/{video}/{frame}/{subframe}/mask.png {instance_mask_name}")
            
    print("Done!")