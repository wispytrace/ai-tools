import os
from pathlib import Path

# ====================== 配置参数 ======================
LABELS_DIR = "/ultralytics/data/labels"           # 包含 .txt 文件的根目录（支持子目录如 train/val）
OLD_TO_NEW = {1 : 5, 2: 5, 3: 5, 4:5}         # 字典形式：旧类别 → 新类别
DRY_RUN = False                          # True: 只预览不修改；False: 实际写入
BACKUP = False                            # 是否在修改前备份原文件（.bak）
# =====================================================


def replace_class_in_file(file_path, mapping):
    """读取单个 label 文件，替换类别，并返回修改行数"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    changed = 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
            if cls_id in mapping:
                parts[0] = str(mapping[cls_id])
                changed += 1
            new_lines.append(' '.join(parts) + '\n')
            print(f"[处理] {file_path}: {line.strip()} -> {' '.join(parts)}")
        except ValueError:
            print(f"[警告] 跳过无效行（非数字类别）: {line.strip()} in {file_path}")

    if changed > 0 and not DRY_RUN:
        # 备份
        if BACKUP:
            backup_path = str(file_path) + ".bak"
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
            else:
                print(f"[提示] 备份已存在: {backup_path}")
        # 写入新内容
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    return changed


def main():
    labels_path = Path(LABELS_DIR)
    if not labels_path.exists():
        print(f"[错误] 标签目录不存在: {LABELS_DIR}")
        return

    total_files = 0
    total_changes = 0

    print(f"开始处理标签目录: {labels_path}")
    print(f"替换映射: {OLD_TO_NEW}")
    print(f"试运行模式 (DRY_RUN): {DRY_RUN}")
    print("-" * 60)

    for txt_file in labels_path.rglob("*.txt"):  # 递归查找所有 .txt
        changes = replace_class_in_file(txt_file, OLD_TO_NEW)
        if changes > 0:
            total_changes += changes
            total_files += 1
            status = "[模拟]" if DRY_RUN else "[已修改]"
            print(f"{status} {txt_file} -> 修改了 {changes} 行")

    print("-" * 60)
    print(f"✅ 完成！共处理 {total_files} 个文件，总计修改 {total_changes} 处类别。")
    if DRY_RUN:
        print(f"💡 提示：设置 DRY_RUN = False 以应用实际修改。")


if __name__ == "__main__":
    main()
