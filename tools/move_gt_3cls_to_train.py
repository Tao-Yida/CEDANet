import os
import shutil

# 路径设置
test_gt_dir = "data/ijmond_data/test/gt"
test_gt3cls_dir = "data/ijmond_data/test/gt_3cls"
train_gt3cls_dir = "data/ijmond_data/train/gt_3cls"


# 只保留图片文件
def filter_images(file_list):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return [f for f in file_list if f.lower().endswith(exts)]


os.makedirs(train_gt3cls_dir, exist_ok=True)

# 获取文件名集合
gt_names = set(filter_images(os.listdir(test_gt_dir)))
gt3cls_names = set(filter_images(os.listdir(test_gt3cls_dir)))

# 找出只在gt_3cls中、但不在gt中的图片
to_move = gt3cls_names - gt_names

print(f"需要移动的文件数: {len(to_move)}")
for fname in to_move:
    src = os.path.join(test_gt3cls_dir, fname)
    dst = os.path.join(train_gt3cls_dir, fname)
    print(f"移动: {src} -> {dst}")
    shutil.move(src, dst)
