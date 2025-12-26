"""
七巧板图像去噪脚本

功能：
1. 读取原始图像（可能包含噪声、阴影、水印等）
2. 通过颜色量化和聚类去除噪声
3. 保存干净的图像

去噪策略：
- K-means颜色聚类，将相似颜色合并为主要颜色
- 形态学操作去除孤立噪点
- 颜色映射到最近的主要颜色
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from pathlib import Path
import os


class ImageDenoiser:
    def __init__(self, n_colors=8, morph_kernel_size=3):
        """
        初始化图像去噪器
        
        Args:
            n_colors: 期望的主要颜色数量（7块组件+背景）
            morph_kernel_size: 形态学操作的核大小
        """
        self.n_colors = n_colors
        self.morph_kernel_size = morph_kernel_size
        
    def load_image(self, image_path):
        """加载图像"""
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 如果有alpha通道，转换为RGB
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            # 将alpha通道与白色背景混合
            alpha = img_array[:, :, 3:4] / 255.0
            rgb = img_array[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        
        return img_array
    
    def color_quantization(self, image):
        """
        颜色量化：使用K-means将图像颜色减少到n_colors种
        
        Args:
            image: RGB图像数组
            
        Returns:
            quantized_image: 量化后的图像
            color_centers: 主要颜色中心
        """
        h, w, c = image.shape
        
        # Reshape图像为(像素数, 3)
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(pixels)
        color_centers = kmeans.cluster_centers_.astype(np.uint8)
        
        # 将每个像素替换为其簇中心的颜色
        quantized_pixels = color_centers[labels]
        quantized_image = quantized_pixels.reshape(h, w, c)
        
        return quantized_image, color_centers
    
    def remove_small_noise(self, image, min_region_size=50):
        """
        去除小的孤立噪点区域
        
        Args:
            image: RGB图像
            min_region_size: 最小保留区域大小（像素数）
            
        Returns:
            cleaned_image: 清理后的图像
        """
        h, w, c = image.shape
        cleaned = image.copy()
        
        # 获取所有唯一颜色
        pixels = image.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        # 对每种颜色创建mask
        for color in unique_colors:
            # 创建当前颜色的mask
            mask = np.all(image == color, axis=2).astype(np.uint8)
            
            # 标记连通区域
            labeled_mask = self.label_connected_components(mask)
            
            # 计算每个区域的大小
            unique_labels = np.unique(labeled_mask)
            for label in unique_labels:
                if label == 0:  # 跳过背景
                    continue
                    
                region_size = np.sum(labeled_mask == label)
                
                # 如果区域太小，认为是噪声，替换为最常见的邻近颜色
                if region_size < min_region_size:
                    region_coords = np.where(labeled_mask == label)
                    cleaned = self.replace_noise_region(cleaned, region_coords, image)
        
        return cleaned
    
    def label_connected_components(self, binary_mask):
        """
        简单的连通区域标记（4连通）
        
        Args:
            binary_mask: 二值mask
            
        Returns:
            labeled: 标记后的mask
        """
        labeled = np.zeros_like(binary_mask, dtype=np.int32)
        current_label = 0
        h, w = binary_mask.shape
        
        def flood_fill(y, x, label):
            """泛洪填充"""
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if labeled[cy, cx] != 0 or binary_mask[cy, cx] == 0:
                    continue
                    
                labeled[cy, cx] = label
                
                # 4连通
                stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
        
        # 遍历所有像素
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] == 1 and labeled[i, j] == 0:
                    current_label += 1
                    flood_fill(i, j, current_label)
        
        return labeled
    
    def replace_noise_region(self, image, region_coords, original_image):
        """
        用周围最常见的颜色替换噪声区域
        
        Args:
            image: 当前图像
            region_coords: 噪声区域坐标
            original_image: 原始图像
            
        Returns:
            image: 更新后的图像
        """
        ys, xs = region_coords
        
        # 获取周围像素的颜色
        neighbors = []
        for y, x in zip(ys, xs):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                        if not (ny in ys and nx in xs):  # 不在噪声区域内
                            neighbors.append(tuple(image[ny, nx]))
        
        if neighbors:
            # 找到最常见的邻近颜色
            from collections import Counter
            most_common_color = Counter(neighbors).most_common(1)[0][0]
            image[ys, xs] = most_common_color
        
        return image
    
    def median_filter(self, image, kernel_size=3):
        """
        中值滤波（简化版，用于平滑）
        
        Args:
            image: RGB图像
            kernel_size: 滤波核大小
            
        Returns:
            filtered: 滤波后的图像
        """
        h, w, c = image.shape
        pad = kernel_size // 2
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        filtered = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    window = padded[i:i+kernel_size, j:j+kernel_size, k]
                    filtered[i, j, k] = np.median(window)
        
        return filtered
    
    def denoise(self, image_path, apply_median=False):
        """
        执行完整的去噪流程
        
        Args:
            image_path: 图像路径
            apply_median: 是否应用中值滤波
            
        Returns:
            denoised_image: 去噪后的图像
        """
        # 1. 加载图像
        image = self.load_image(image_path)
        print(f"  原始图像形状: {image.shape}")
        
        # 2. 颜色量化
        quantized, color_centers = self.color_quantization(image)
        print(f"  量化为 {len(color_centers)} 种主要颜色")
        
        # 3. 去除小噪点（可选，可能比较慢）
        # cleaned = self.remove_small_noise(quantized, min_region_size=20)
        cleaned = quantized  # 如果速度慢可以跳过此步
        
        # 4. 中值滤波（可选）
        if apply_median:
            cleaned = self.median_filter(cleaned, kernel_size=3)
            print(f"  应用中值滤波")
        
        return cleaned


def process_single_image(image_path, output_path, n_colors=8):
    """
    处理单张图像
    
    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        n_colors: 期望的主要颜色数量
    """
    print(f"\n处理: {os.path.basename(image_path)}")
    
    denoiser = ImageDenoiser(n_colors=n_colors)
    denoised = denoiser.denoise(image_path, apply_median=False)
    
    # 保存图像
    output_image = Image.fromarray(denoised)
    output_image.save(output_path)
    print(f"  已保存: {os.path.basename(output_path)}")
    
    return denoised


def process_all_images(input_folder, output_folder, n_colors=8):
    """
    处理文件夹中的所有图像
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        n_colors: 期望的主要颜色数量（默认8：7块组件+背景）
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 创建输出文件夹
    output_path.mkdir(exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg'))
    
    if len(image_files) == 0:
        print(f"在 {input_folder} 中未找到图像文件")
        return
    
    print("=" * 70)
    print(f"七巧板图像去噪")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"找到 {len(image_files)} 张图像")
    print("=" * 70)
    
    success_count = 0
    
    for img_file in sorted(image_files):
        try:
            output_file = output_path / f"{img_file.stem}_clean.png"
            process_single_image(str(img_file), str(output_file), n_colors=n_colors)
            success_count += 1
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"处理完成！成功处理 {success_count}/{len(image_files)} 张图像")
    print(f"去噪后的图像已保存到: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    # 输入和输出文件夹
    input_folder = "./data_resized"       # 保存截的图的文件夹路径
    output_folder = "./data_clean"        # 去噪后图像的保存路径
    
    # 处理所有图像
    # n_colors: 期望的主要颜色数量
    # 如果去噪效果不好，可以尝试调整这个参数（7-10之间）
    process_all_images(input_folder, output_folder, n_colors=8)
