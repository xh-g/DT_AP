"""
七巧板组件中心坐标提取 - 简化版
按面积大小编号（#1=最大，#7=最小）
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TangramExtractor:
    def __init__(self, image_path, bg_threshold=220):
        self.image_path = image_path
        self.bg_threshold = bg_threshold
        self.image = None
        self.width = None
        self.height = None
    
    def load_image(self):
        """加载图像"""
        img = Image.open(self.image_path)
        self.image = np.array(img)
        self.height, self.width = self.image.shape[:2]
    
    def filter_background(self):
        """过滤背景"""
        grayscale = np.mean(self.image, axis=2)
        foreground_mask = grayscale < self.bg_threshold
        return foreground_mask
    
    def find_components_by_color(self, foreground_mask):
        """按颜色分组，然后对每种颜色做连通区域分析"""
        foreground_pixels = self.image[foreground_mask]
        foreground_coords = np.argwhere(foreground_mask)
        
        # 找到所有唯一颜色
        unique_colors = np.unique(foreground_pixels.reshape(-1, 3), axis=0)
        print(f"检测到 {len(unique_colors)} 种唯一颜色")
        
        all_components = []
        
        # 对每种颜色进行连通区域分析
        for color in unique_colors:
            color_mask = np.all(self.image == color, axis=2)
            color_coords = np.argwhere(color_mask)
            
            if len(color_coords) < 50:  # 过滤噪点
                continue
            
            # BFS找连通区域
            visited = set()
            
            for start_y, start_x in color_coords:
                if (start_y, start_x) in visited:
                    continue
                
                # BFS
                component = []
                queue = deque([(start_y, start_x)])
                visited.add((start_y, start_x))
                
                while queue:
                    y, x = queue.popleft()
                    component.append((y, x))
                    
                    # 检查8邻域
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.height and 0 <= nx < self.width and
                                (ny, nx) not in visited and color_mask[ny, nx]):
                                visited.add((ny, nx))
                                queue.append((ny, nx))
                
                if len(component) > 50:
                    all_components.append(component)
        
        print(f"成功识别 {len(all_components)} 个连通组件")
        return all_components
    
    def identify_shape(self, coords_array):
        """使用轮廓分析和顶点数判断形状，对旋转更鲁棒"""
        # 计算包围盒与基础比率（与旧逻辑保持兼容）
        y_min, y_max = coords_array[:, 0].min(), coords_array[:, 0].max()
        x_min, x_max = coords_array[:, 1].min(), coords_array[:, 1].max()

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1

        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        bbox_area = bbox_width * bbox_height
        fill_ratio = len(coords_array) / bbox_area if bbox_area > 0 else 0.0

        # 将该连通域映射到局部二值图，寻找轮廓
        h, w = bbox_height, bbox_width
        if h <= 0 or w <= 0:
            return '其他', fill_ratio, aspect_ratio

        binary = np.zeros((h, w), dtype=np.uint8)
        yy = coords_array[:, 0] - y_min
        xx = coords_array[:, 1] - x_min
        binary[yy, xx] = 255
        # 形态学闭运算，减少锯齿和小孔，稳定顶点数
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return '其他', fill_ratio, aspect_ratio

        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)

        # 先用最小包络三角形进行面积比检测，强鲁棒三角形判定
        hull_area = cv2.contourArea(hull)
        try:
            enc_tri_area, enc_tri = cv2.minEnclosingTriangle(hull)
        except Exception:
            enc_tri_area, enc_tri = (0.0, None)
        if enc_tri_area and enc_tri_area > 0:
            tri_fill = hull_area / enc_tri_area
            if tri_fill > 0.80:
                return '三角形', fill_ratio, aspect_ratio

        peri = cv2.arcLength(hull, True)
        approx = None
        for f in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
            eps = f * peri
            approx = cv2.approxPolyDP(hull, eps, True)
            if len(approx) <= 4:
                break
        verts = len(approx)

        # 三角形：3个顶点
        if verts == 3:
            return '三角形', fill_ratio, aspect_ratio

        # 顶点数>=4：先尝试三角形拟合（应对锯齿导致的多点近似）
        if verts >= 4:
            hull_area = cv2.contourArea(hull)
            tri = None
            for f in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]:
                tri_approx = cv2.approxPolyDP(hull, f * peri, True)
                if len(tri_approx) == 3:
                    tri = tri_approx
                    break
            if tri is not None:
                tri_area = cv2.contourArea(tri)
                if hull_area > 0 and (tri_area / max(hull_area, 1e-6)) > 0.80:
                    return '三角形', fill_ratio, aspect_ratio

            # 进一步启发式：顶点近共线或存在极短边，视为三角形（多余顶点）
            pts_all = approx.reshape(-1, 2).astype(np.float32)
            # 边长
            edge_lengths = []
            for i in range(len(pts_all)):
                p1 = pts_all[i]
                p2 = pts_all[(i + 1) % len(pts_all)]
                edge_lengths.append(np.linalg.norm(p2 - p1))
            max_e = max(edge_lengths) if edge_lengths else 1.0
            min_e = min(edge_lengths) if edge_lengths else 0.0
            perim_e = sum(edge_lengths) if edge_lengths else 1.0
            # 内角余弦与角度
            def cos_angle(a, b, c):
                v1 = a - b
                v2 = c - b
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 == 0 or n2 == 0:
                    return 1.0
                return np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles_deg = []
            for i in range(len(pts_all)):
                a = pts_all[(i - 1) % len(pts_all)]
                b = pts_all[i]
                c = pts_all[(i + 1) % len(pts_all)]
                ca = cos_angle(a, b, c)
                angles_deg.append(np.degrees(np.arccos(ca)))
            # 判定条件：存在大角近共线或极短边（多余顶点）
            near_collinear = any(ang > 160.0 for ang in angles_deg)
            very_short_edge = (min_e > 0 and (min_e / max(max_e, 1e-6) < 0.35 or min_e / perim_e < 0.08))
            if near_collinear or very_short_edge:
                return '三角形', fill_ratio, aspect_ratio

        # 四边形：进一步区分正方形和平行四边形
        if verts == 4:
            pts = approx.reshape(-1, 2)

            # 计算连续边向量与夹角余弦，判断是否近似直角
            def cos_angle(a, b, c):
                v1 = a - b
                v2 = c - b
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 == 0 or n2 == 0:
                    return 1.0
                return np.dot(v1, v2) / (n1 * n2)

            cos_list = []
            for i in range(4):
                a = pts[(i - 1) % 4].astype(np.float32)
                b = pts[i].astype(np.float32)
                c = pts[(i + 1) % 4].astype(np.float32)
                cos_list.append(cos_angle(a, b, c))

            right_angles = all(abs(c) < 0.25 for c in cos_list)  # 约等于90°

            # 边长比判断是否接近等边（用于区分正方形 vs 长方形/其他）
            edges = []
            for i in range(4):
                p1 = pts[i].astype(np.float32)
                p2 = pts[(i + 1) % 4].astype(np.float32)
                edges.append(np.linalg.norm(p2 - p1))
            max_len = max(edges)
            min_len = min(edges) if min(edges) > 0 else 1.0
            side_ratio = max_len / min_len

            if right_angles and side_ratio < 1.15:
                return '正方形', fill_ratio, aspect_ratio
            else:
                return '平行四边形', fill_ratio, aspect_ratio

        # 顶点数不在预期范围，回退：根据凸包顶点数粗略判断
        hull = cv2.convexHull(cnt)
        hull_peri = cv2.arcLength(hull, True)
        hull_approx = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
        if len(hull_approx) == 3:
            return '三角形', fill_ratio, aspect_ratio
        elif len(hull_approx) == 4:
            return '平行四边形', fill_ratio, aspect_ratio
        else:
            return '其他', fill_ratio, aspect_ratio
    
    def calculate_component_info(self, components):
        """计算每个组件的信息"""
        component_info = []
        
        for coords in components:
            coords_array = np.array(coords)
            
            # 计算中心（像素坐标）
            center_y = np.mean(coords_array[:, 0])
            center_x = np.mean(coords_array[:, 1])
            
            # 归一化坐标
            x_norm, y_norm = self.normalize_coordinates(center_x, center_y)
            
            # 面积
            area = len(coords)
            
            # 识别形状
            shape_type, fill_ratio, aspect_ratio = self.identify_shape(coords_array)
            
            component_info.append({
                'coords': coords,
                'center_pixel': (center_x, center_y),
                'center_normalized': (x_norm, y_norm),
                'area': area,
                'shape_type': shape_type,
                'fill_ratio': fill_ratio,
                'aspect_ratio': aspect_ratio
            })
        
        return component_info
    
    def normalize_coordinates(self, center_x, center_y):
        """归一化坐标到[-1, 1]，原点在图像中心，x轴向右，y轴向上"""
        img_center_x = self.width / 2
        img_center_y = self.height / 2
        
        x = center_x - img_center_x
        y = -(center_y - img_center_y)  # y轴向上为正
        
        norm_factor = max(self.width, self.height) / 2
        x_norm = x / norm_factor
        y_norm = y / norm_factor
        
        return x_norm, y_norm
    
    def extract_and_save(self, save_folder):
        """提取坐标并保存"""
        import os
        
        # 加载和处理
        self.load_image()
        print("正在过滤背景...")
        foreground_mask = self.filter_background()
        components = self.find_components_by_color(foreground_mask)
        component_info = self.calculate_component_info(components)
        
        # 按面积排序（从大到小）
        component_info.sort(key=lambda x: x['area'], reverse=True)
        
        # 取前7个
        top_7 = component_info[:7]
        
        # 生成结果数组
        numbered_coords = np.array([c['center_normalized'] for c in top_7])
        zeros_extra = np.zeros((numbered_coords.shape[0], 2))
        numbered_coords = np.hstack([numbered_coords, zeros_extra])
        
        # 保存路径
        os.makedirs(save_folder, exist_ok=True)
        base_name = os.path.basename(self.image_path).replace('.png', '').replace('_clean', '')
        
        # 保存txt文件
        txt_path = os.path.join(save_folder, f"{base_name}_info.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"图像: {os.path.basename(self.image_path)}\n")
            f.write(f"图像尺寸: {self.width} x {self.height}\n")
            f.write(f"编号规则: 按面积从大到小排序 (#1=最大面积, #7=最小面积)\n")
            f.write("="*80 + "\n\n")
            
            f.write("组件详细信息:\n")
            f.write("-"*80 + "\n")
            
            for i, info in enumerate(top_7):
                f.write(f"\n编号 {i+1}\n")
                f.write(f"  形状: {info['shape_type']}\n")
                f.write(f"  面积: {info['area']} 像素\n")
                f.write(f"  填充率: {info['fill_ratio']:.3f}\n")
                f.write(f"  长宽比: {info['aspect_ratio']:.3f}\n")
                x, y = info['center_normalized']
                f.write(f"  归一化坐标: ({x:+.6f}, {y:+.6f})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("归一化坐标数组 (7x4):\n")
            f.write("-"*80 + "\n")
            for i in range(7):
                x, y, z1, z2 = numbered_coords[i]
                f.write(f"{i+1}:  [{x:+.6f}, {y:+.6f}, {int(z1)}, {int(z2)}]\n")
        
        print(f"详细信息已保存: {txt_path}")
        
        # 保存可视化图像
        vis_path = os.path.join(save_folder, f"{base_name}_result.png")
        self.visualize(component_info, top_7, vis_path)
        
        return numbered_coords
    
    def visualize(self, all_components, top_7, save_path):
        """可视化结果"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.image)
        
        # 绘制前7个组件的中心点
        for i, info in enumerate(top_7):
            cx, cy = info['center_pixel']
            ax.plot(cx, cy, 'r+', markersize=20, markeredgewidth=3)
            label = f"#{i+1}\n{info['shape_type']}\n面积:{info['area']}"
            ax.text(cx, cy-8, label,
                   color='red', fontsize=10, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9))
        
        # 绘制图像中心
        ax.plot(self.width/2, self.height/2, 'bx', markersize=25, markeredgewidth=3)
        ax.text(self.width/2, self.height/2-10, '图像中心',
               color='blue', fontsize=12, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='cyan', alpha=0.8))
        
        ax.set_title('七巧板组件识别（按面积排序）', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化图像已保存: {save_path}")
        plt.close()


def process_all_images(input_folder, output_folder):
    """批量处理所有图像"""
    import os
    import glob
    
    #image_files = glob.glob(os.path.join(input_folder, "*_clean.png"))
    image_files = glob.glob(os.path.join(input_folder, "*.png"))
    
    if len(image_files) == 0:
        print(f"错误：在 {input_folder} 中未找到 *_clean.png 文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图像")
    print("="*80)
    
    for i, image_path in enumerate(image_files):
        print(f"\n处理 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        print("-"*80)
        
        try:
            extractor = TangramExtractor(image_path)
            extractor.extract_and_save(output_folder)
        except Exception as e:
            print(f"错误: {e}")
            continue
    
    print("\n" + "="*80)
    print(f"全部完成！")
    print(f"结果保存在: {output_folder}")
    print("="*80)


if __name__ == "__main__":
    # input_folder = "./data_clean"
    # output_folder = "./results"
    input_folder = "./xiugai"
    output_folder = "./results-xiugai"
    
    print("\n" + "="*80)
    print("七巧板组件坐标提取 - 批量处理")
    print("="*80)
    
    process_all_images(input_folder, output_folder)
