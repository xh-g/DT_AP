# resize操作
import cv2
import os
import glob
import numpy as np

def resize_and_center_image(image, target_size=(120, 120)):
    """
    将图像resize到目标大小，并保持内容在中心位置
    """
    # 获取原始图像尺寸
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例（保持宽高比）
    scale = min(target_w / w, target_h / h)
    
    # 计算缩放后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    result = np.full((target_h, target_w, 3), [254, 254, 254], dtype=np.uint8)
    
    # 计算在中心位置的坐标
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # 将缩放后的图像放置在中心
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result

def process_and_save_images(folder_path, output_folder, target_size=(120, 120)):
    """
    处理文件夹中的所有图像并保存到指定输出文件夹
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
    
    print(f"找到 {len(image_files)} 张图像")
    
    processed_count = 0
    
    for i, image_path in enumerate(image_files):
        # 读取图像
        img = cv2.imread(image_path)
        
        if img is not None:
            print(f"\n处理图像 {i+1}: {os.path.basename(image_path)}")
            print(f"原始形状: {img.shape}")
            
            # 处理图像
            processed_img = resize_and_center_image(img, target_size)
            
            print(f"处理后形状: {processed_img.shape}")
            
            # 保存处理后的图像（保持原文件名）
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, processed_img)
            print(f"已保存: {output_path}")
            
            processed_count += 1
        else:
            print(f"无法读取图像: {image_path}")
    
    return processed_count

# 使用示例
if __name__ == "__main__":
    # 输入和输出路径
    input_folder = './data'
    output_folder = './data_resized'
    target_size = (120, 120)
    
    # 处理并保存所有图像
    processed_count = process_and_save_images(input_folder, output_folder, target_size)
    
    print(f"\n处理完成！共成功处理了 {processed_count} 张图像")
    print(f"所有图像已保存到: {output_folder}")
    print(f"所有图像现在都是 {target_size} 大小，目标物体位于中心位置")