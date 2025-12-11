import os
import shutil
import random
from pathlib import Path

# ------------------------ 配置路径 ------------------------
labels_all = "/ultralytics/data/labels_all"      # 所有原始标签文件 (.txt)
labels_root = "/ultralytics/data/labels"        # 划分后的标签输出 (train/val)
images_all = "/ultralytics/data/images_all"     # 所有原始图片
output_images = "/ultralytics/data/images"      # 输出图片 (train/val)
unlabel_img_dir = "/ultralytics/data/unlabel_img"  # 新增：未标注图片存放路径

img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
split_ratio = 0.9
random_seed = 42
# ----------------------------------------------------------

def split_labels():
    """从 labels_all 中随机划分 train/val 标签文件"""
    all_labels_dir = Path(labels_all)
    train_labels_dir = Path(labels_root) / "train"
    val_labels_dir = Path(labels_root) / "val"

    if not all_labels_dir.exists():
        raise FileNotFoundError(f"标签源目录不存在: {all_labels_dir}")

    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    label_files = [f for f in all_labels_dir.iterdir() if f.is_file() and f.suffix == '.txt']
    if len(label_files) == 0:
        raise ValueError(f"在 {all_labels_dir} 中未找到任何 .txt 标签文件")

    random.seed(random_seed)
    random.shuffle(label_files)

    split_idx = int(len(label_files) * split_ratio)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]

    count_train = 0
    for src_file in train_files:
        dst_file = train_labels_dir / src_file.name
        shutil.copy(src_file, dst_file)
        count_train += 1

    count_val = 0
    for src_file in val_files:
        dst_file = val_labels_dir / src_file.name
        shutil.copy(src_file, dst_file)
        count_val += 1

    print(f"[标签划分完成] train: {count_train}, val: {count_val}")
    print(f"               比例: {len(train_files)}:{len(val_files)} ≈ {split_ratio}:{1-split_ratio}")


def find_and_copy_images(label_subdir, image_output_dir):
    """根据标签文件名复制对应的图片到指定目录"""
    os.makedirs(image_output_dir, exist_ok=True)

    label_path = Path(labels_root) / label_subdir
    if not label_path.exists():
        print(f"[警告] 标签路径不存在: {label_path}")
        return

    count = 0
    for label_file in label_path.glob("*.txt"):
        image_name_without_ext = label_file.stem

        found = False
        src_image_path = None
        for ext in img_extensions:
            candidate = Path(images_all) / (image_name_without_ext + ext)
            if candidate.exists():
                src_image_path = candidate
                found = True
                break

        if found:
            dst_image_path = Path(output_images) / image_output_dir / (image_name_without_ext + src_image_path.suffix)
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_image_path, dst_image_path)
            count += 1
        else:
            print(f"[未找到] 图片缺失: {image_name_without_ext}")

    print(f"[图片复制] 已复制 {count} 张图片到 {image_output_dir}")


# 👇 新增函数：复制无标签图片
def copy_unlabeled_images():
    """
    找出 images_all 中所有没有对应 .txt 标签文件的图片，
    每 200 张复制到一个独立子目录中，如 unlabel_img/part_001/, part_002/ ...
    """
    print(f"\n[步骤3/3] 正在查找并复制无标签图片到 {unlabel_img_dir}（每200张一分组）...")

    # --- 1. 获取所有已标注图片的 stem 名称 ---
    labeled_stems = set()
    labels_all_path = Path(labels_all)

    if labels_all_path.exists():
        for txt_file in labels_all_path.glob("*.txt"):
            labeled_stems.add(txt_file.stem.lower())  # 统一转小写，避免大小写冲突
    else:
        print(f"[警告] labels_all 目录不存在: {labels_all_path}")
        print("         将视为所有图片都无标签")
    
    # --- 2. 收集所有图片路径，并去重 ---
    all_image_paths = []
    images_all_path = Path(images_all)

    if not images_all_path.exists():
        raise FileNotFoundError(f"图片源目录不存在: {images_all_path}")

    for ext in img_extensions:
        # 匹配大小写扩展名（如 .JPG, .jpeg 等）
        all_image_paths.extend(images_all_path.glob(f"*{ext}"))
        all_image_paths.extend(images_all_path.glob(f"*{ext.upper()}"))

    # 去重 + 按文件名排序（保证可复现）
    all_image_paths = sorted(set(all_image_paths))

    # --- 3. 筛选出无标签图片 ---
    unlabeled_images = []
    for img_path in all_image_paths:
        if img_path.stem.lower() not in labeled_stems:
            unlabeled_images.append(img_path)

    if len(unlabeled_images) == 0:
        print("[提示] 未找到无标签图片，跳过复制")
        return

    # --- 4. 创建输出目录，并按每组200张分批复制 ---
    unlabel_output = Path(unlabel_img_dir)
    unlabel_output.mkdir(parents=True, exist_ok=True)

    batch_size = 200
    total_copied = 0

    for i, img_path in enumerate(unlabeled_images):
        # 计算当前属于哪个分组文件夹
        part_idx = i // batch_size  # 0,1,2,...
        subfolder_name = f"part_{part_idx+1:03d}"  # → part_001, part_002...
        part_dir = unlabel_output / subfolder_name
        part_dir.mkdir(exist_ok=True)  # 确保子目录存在

        # 目标路径保持原文件名
        dst_path = part_dir / img_path.name

        try:
            shutil.copy(img_path, dst_path)
            total_copied += 1
        except Exception as e:
            print(f"[错误] 复制失败: {img_path.name} -> {e}")

    print(f"[无标签图片] 共找到 {len(unlabeled_images)} 张无标签图片")
    print(f"             已分批复制到: {unlabel_img_dir}/part_XXX/ （每 {batch_size} 张一组）")
    print(f"             总共创建了 { (len(unlabeled_images) - 1) // batch_size + 1 } 个子文件夹")


# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    # print("[步骤1/3] 正在划分标签文件 (train:val = 9:1)...")
    # split_labels()

    # print("\n[步骤2/3] 正在根据标签复制对应图片...")
    # subsets = ["train", "val"]
    # for subset in subsets:
    #     find_and_copy_images(subset, subset)

    # 👇 调用新功能
    copy_unlabeled_images()

    print("\n✅ 数据准备全部完成！")
    print(f"   标签路径: {labels_root}")
    print(f"   图片路径: {output_images}")
    print(f"   无标签图片: {unlabel_img_dir}")
