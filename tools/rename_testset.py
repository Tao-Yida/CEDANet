import os

# 需要重命名的文件夹
folders = [
    "data/ijmond_data/test/gt",
    "data/ijmond_data/test/gt_3cls",
    "data/ijmond_data/test/masks",
    "data/ijmond_data/test/img",
]


def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".npy"))


for folder in folders:
    files = sorted([f for f in os.listdir(folder) if is_image_file(f)])
    # img 文件夹要排除 json 文件
    if folder.endswith("img"):
        files = [f for f in files if not f.lower().endswith(".json")]
    for idx, fname in enumerate(files, 1):
        ext = os.path.splitext(fname)[1]
        new_name = f"img{idx}{ext}"
        src = os.path.join(folder, fname)
        dst = os.path.join(folder, new_name)
        if src != dst:
            os.rename(src, dst)
        print(f"{src} -> {dst}")
