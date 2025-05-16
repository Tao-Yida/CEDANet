import numpy as np
import os
from PIL import Image

# mask_dir = "./data/ijmond_data/masks"
# gt_dir = "./data/ijmond_data/gt"  # 建议改成 gt_binary/ 以便区分
# os.makedirs(gt_dir, exist_ok=True)

# converted_files = []

# for fname in os.listdir(mask_dir):
#     if fname.endswith(".npy"):
#         mask = np.load(os.path.join(mask_dir, fname))  # 0 / 1 / 2
#         mask[mask == 2] = 1  # ☆ 合并低浓烟→1

#         mask_img = mask.astype(np.uint8) * 255  # 0 or 255
#         if mask_img.ndim == 3 and mask_img.shape[-1] == 1:
#             mask_img = mask_img.squeeze(-1)

#         img = Image.fromarray(mask_img, mode="L")  # 'L' = 单通道灰度
#         img.save(os.path.join(gt_dir, f"{os.path.splitext(fname)[0]}.png"))
#         converted_files.append(fname)

# print(f"Binary masks saved to {gt_dir} with {len(converted_files)} files.")

mask_dir = "./data/ijmond_data/masks"
gt_dir = "./data/ijmond_data/gt_3cls"  # 建议换名以示区分
os.makedirs(gt_dir, exist_ok=True)

converted_files = []

# 映射表：原值→灰度值
to_gray = {0: 0, 1: 255, 2: 127}  # 先映射 0/1/2 → 0/255/127

for fname in os.listdir(mask_dir):
    if fname.endswith(".npy"):
        mask = np.load(os.path.join(mask_dir, fname))  # 0 / 1 / 2
        mask_img = np.vectorize(to_gray.get)(mask).astype(np.uint8)

        if mask_img.ndim == 3 and mask_img.shape[-1] == 1:
            mask_img = mask_img.squeeze(-1)

        img = Image.fromarray(mask_img, mode="L")
        img.save(os.path.join(gt_dir, f"{os.path.splitext(fname)[0]}.png"))
        converted_files.append(fname)

print(f"Three-class masks saved to {gt_dir} – {len(converted_files)} files.")
