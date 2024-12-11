import os
import shutil
import random

def copy_random_images(src_folder, dst_folder, percentage=0.5):
    """
    随机选择 src_folder 文件夹中的一定比例的图片文件并复制到 dst_folder 文件夹中。

    参数:
    src_folder (str): 源文件夹路径。
    dst_folder (str): 目标文件夹路径。
    percentage (float): 要复制的图片比例（默认为 0.5，即 50%）。
    """
    # 如果目标文件夹不存在，创建它
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 获取源文件夹中的所有文件列表
    all_files = os.listdir(src_folder)

    # 筛选出图片文件（假设是以常见图片扩展名结尾的文件）
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 计算要复制的图片数量
    num_to_copy = int(len(image_files) * percentage)

    # 随机选择指定数量的图片文件
    selected_files = random.sample(image_files, num_to_copy)

    # 复制文件到目标文件夹
    for file_name in selected_files:
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {file_name}")

    print(f"共复制了 {len(selected_files)} 张图片到 {dst_folder}")

# 示例使用
src_folder = r"E:\code\WSSS-Tissue-main\WSSS-Tissue-main\datasets\BCSS-WSSS\train"  # 替换为源文件夹路径
dst_folder = r"E:\code\WSSS-Tissue-main\WSSS-Tissue-main\datasets\BCSS-WSSS\train_50p_diff"  # 替换为目标文件夹路径
copy_random_images(src_folder, dst_folder)
