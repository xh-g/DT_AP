import os
import glob


def create_empty_txt_for_images(folder):
    """为指定文件夹中的每个图片文件创建同名空白 txt 文件"""
    # 这里只按当前数据使用的 png 图像匹配
    pattern = os.path.join(folder, "*.png")
    image_files = glob.glob(pattern)

    if not image_files:
        print(f"在 {folder} 中未找到 png 图像文件")
        return

    for image_path in image_files:
        base, _ = os.path.splitext(image_path)
        txt_path = base + ".txt"

        # 如果不存在同名 txt，则创建空白文件
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8"):
                pass
            print(f"创建空白 txt 文件: {os.path.basename(txt_path)}")
        else:
            print(f"已存在，跳过: {os.path.basename(txt_path)}")


if __name__ == "__main__":
    # 根据你目前的工程路径设置 data_clean 目录
    folder = "./data_clean"
    create_empty_txt_for_images(folder)
