import cv2
import os

data = "data/SMOKE5K_Dataset/SMOKE5K/weak_supervision/scribble"


def check_scribble():
    """
    检查数据集中所有的涂鸦标注是否正确
    """
    scribble_files = [f for f in os.listdir(data) if f.endswith(".png")]
    for scribble_file in scribble_files:
        scribble_path = os.path.join(data, scribble_file)
        scribble_img = cv2.imread(scribble_path, cv2.IMREAD_GRAYSCALE)

        if scribble_img is None:
            print(f"Error reading {scribble_file}")
            continue

        # 检查涂鸦标注是否为二值图像
        unique_values = set(scribble_img.flatten())
        if unique_values.issubset({0, 255}):
            print(f"{scribble_file} is a valid scribble annotation.")
        else:
            print(f"{scribble_file} contains invalid values: {unique_values}.")


if __name__ == "__main__":
    check_scribble()
