import cv2
import numpy as np
from PIL import Image
import os
from huggingface_hub import hf_hub_download

def expand_polygon(cnt, offset):
    """
    对多边形进行向外偏移 (Polygon Buffering)，恢复因腐蚀/边缘检测损失的面积。
    保持顶点数量不变，避免圆角化。
    """
    # 1. 确保是多边形 (approx)
    # 使用较小的 epsilon 以保留更多细节，或者直接使用传入的 cnt 如果它已经是 approx
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # 如果顶点太多或太少，可能不是标准的七巧板形状，保守起见只做微小处理或原样返回
    if len(approx) < 3:
        return cnt
        
    points = approx.reshape(-1, 2).astype(np.float32)
    num_points = len(points)
    original_area = cv2.contourArea(approx)
    
    def compute_expanded(current_offset):
        new_points = []
        for i in range(num_points):
            p_prev = points[i-1]
            p_curr = points[i]
            p_next = points[(i+1) % num_points]
            
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            
            # 处理重合点
            if l1 < 1e-5: v1 = np.array([1.0, 0.0])
            else: v1 /= l1
            
            if l2 < 1e-5: v2 = np.array([1.0, 0.0])
            else: v2 /= l2
            
            # 计算法向量 (假设某种手性，后续通过面积校验)
            # 向量 (x, y) 的法向量 (-y, x)
            n1 = np.array([-v1[1], v1[0]])
            n2 = np.array([-v2[1], v2[0]])
            
            # 角平分线方向
            bisector = n1 + n2
            len_bis = np.linalg.norm(bisector)
            
            if len_bis < 1e-3:
                # 共线或180度折返，直接沿法线移动
                shift = n1 * current_offset
            else:
                bisector /= len_bis
                # 投影长度: offset / cos(theta/2)
                # cos(theta/2) = dot(n1, bisector)
                cos_half = np.dot(n1, bisector)
                # 避免尖角过度延伸 (Miter Limit)
                if abs(cos_half) < 0.1: # 角度非常尖
                    shift = bisector * current_offset * 2 # 限制最大延伸
                else:
                    shift = bisector * (current_offset / cos_half)
            
            new_points.append(p_curr + shift)
        return np.array(new_points, dtype=np.int32).reshape(-1, 1, 2)

    # 尝试正向偏移
    poly_plus = compute_expanded(offset)
    area_plus = cv2.contourArea(poly_plus)
    
    # 如果面积变大了，说明方向正确
    if area_plus > original_area:
        return poly_plus
    else:
        # 否则反向偏移
        return compute_expanded(-offset)

def get_shape_and_angle(cnt):
    # 1. 强制拟合为 3 或 4 边形
    peri = cv2.arcLength(cnt, True)
    approx = None
    # 动态调整 epsilon 直到得到 3 或 4 个顶点
    for eps_factor in np.linspace(0.01, 0.15, 15):
        approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
        if len(approx) in [3, 4]:
            break
    
    if approx is None or len(approx) not in [3, 4]:
        return "unknown", 0.0

    pts = approx.reshape(-1, 2)
    
    if len(pts) == 3:
        # --- 三角形 ---
        # 0度参考：直角在左下，斜边在右上
        
        # 1. 找直角顶点：对边最长的那个顶点
        dists = []
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i+1)%3]
            d = np.linalg.norm(p1 - p2)
            dists.append(d)
        
        # 最长边索引 (斜边)
        max_idx = np.argmax(dists) 
        # 直角顶点索引
        ra_idx = (max_idx + 2) % 3
        ra_pt = pts[ra_idx]
        
        # 斜边中点
        p_hyp_1 = pts[max_idx]
        p_hyp_2 = pts[(max_idx+1)%3]
        mid_pt = (p_hyp_1 + p_hyp_2) / 2.0
        
        # 向量：直角顶点 -> 斜边中点
        vec = mid_pt - ra_pt
        angle_rad = np.arctan2(vec[1], vec[0])
        angle_deg = np.degrees(angle_rad)
        
        # 0度时，直角在左下，斜边在右上，向量指向右上 (-45度)
        # 旋转角度 = 当前角度 - 参考角度
        rotation = angle_deg - (-45)
        
        # 归一化到 -180 ~ 180
        rotation = (rotation + 180) % 360 - 180
        return "triangle", rotation

    elif len(pts) == 4:
        # --- 四边形 (正方形 或 平行四边形) ---
        
        # 计算边长
        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i+1)%4]
            d = np.linalg.norm(p1 - p2)
            edges.append(d)
        
        edges.sort()
        # 判断长宽比 (正方形接近1，平行四边形接近 1.414)
        ratio = edges[-1] / edges[0] if edges[0] > 0 else 0
        
        if ratio < 1.2:
            # --- 正方形 ---
            # 0度参考：边水平垂直
            # 具有 90 度旋转不变性
            
            # 取第一条边的角度
            p1 = pts[0]
            p2 = pts[1]
            vec = p2 - p1
            angle_deg = np.degrees(np.arctan2(vec[1], vec[0]))
            
            # 模 90 度
            rotation = angle_deg % 90
            
            # 归一化到 [-45, 45]
            if rotation > 45:
                rotation -= 90
                
            return "square", rotation
        else:
            # --- 平行四边形 ---
            # 0度参考：长边水平
            # 具有 180 度旋转不变性
            
            # 找长边向量
            max_len = 0
            long_vec = None
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i+1)%4]
                d = np.linalg.norm(p1 - p2)
                if d > max_len:
                    max_len = d
                    long_vec = p2 - p1
            
            angle_rad = np.arctan2(long_vec[1], long_vec[0])
            angle_deg = np.degrees(angle_rad)
            
            # 模 180 度
            rotation = angle_deg % 180
            
            # 归一化到 [-90, 90]
            if rotation > 90: 
                rotation -= 180
            
            return "parallelogram", rotation

def analyze_tangram_canny(image_path, visualize=True):
    # 1. 读取图像 (增强版)
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在: {image_path}")
        # 尝试解析符号链接
        if os.path.islink(image_path):
            target = os.readlink(image_path)
            print(f"  -> 符号链接指向: {target}")
            resolved = os.path.realpath(image_path)
            print(f"  -> 绝对路径解析为: {resolved}")
            if not os.path.exists(resolved):
                print("  -> 目标文件确实不存在。可能是 Git LFS 未拉取或缓存损坏。")
        return None, None

    # 检查是否是 LFS 指针
    try:
        with open(image_path, 'rb') as f:
            header = f.read(100)
            if b'version https://git-lfs.github.com/spec/v1' in header:
                print("错误: 这是一个 Git LFS 指针文件，不是真正的图像。请安装 git-lfs 并运行 'git lfs pull'。")
                return None, None
    except:
        pass

    img = cv2.imread(image_path)
    if img is None: 
        print(f"错误: 无法读取图像 {image_path}。可能是格式不支持或文件损坏。")
        return None, None

    # 2. 预处理
    # 稍微模糊一点点，去除可能的噪点，有助于 Canny 提取平滑线条
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 3. Canny 边缘检测 (关键步骤)
    # 50 和 150 是低高阈值，针对这种清晰的几何图效果很好
    edges = cv2.Canny(gray, 50, 150)
    
    # Debug: 保存一下看看边缘是否提取出来了
    # cv2.imwrite('debug_edges.png', edges)

    # 4. 膨胀边缘 (关键步骤)
    # 既然原本的缝隙太细，我们就人为地把边缘线“加粗”
    # 这样可以保证积木之间绝对断开
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # 5. 取反并寻找轮廓
    # Canny 得到的是白线黑底，我们需要白块黑底来找连通域
    # 取反后：背景和缝隙变成黑色，积木变成白色
    binary_mask = cv2.bitwise_not(thick_edges)
    
    # 寻找轮廓
# 1. 按照面积从大到小排序
    # key=cv2.contourArea 意味着按面积大小排
    # reverse=True 意味着降序（最大的在最前面）
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    valid_components = []
    image_area = img.shape[0] * img.shape[1]
    
    print(f"初步检测到 {len(contours)} 个轮廓")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # --- 过滤器 1: 剔除背景框 ---
        # 如果面积超过全图的 90%，那它肯定是背景，不是积木
        if area > image_area * 0.25:
            print(f"轮廓 {i} 被剔除: 面积太大 (背景框)")
            continue
            
        # --- 过滤器 2: 剔除微小噪点 ---
        if area < 500: # 根据你的分辨率调整
            print(f"轮廓 {i} 被剔除: 面积太小 (噪点)")
            # 因为已经按面积降序排列，后面的肯定更小，直接退出循环
            break

        # --- 过滤器 3: 几何形状约束 (核心优化) ---
        # 七巧板都是多边形（三角形3条边，四边形4条边）
        # 我们用 approxPolyDP 来拟合轮廓，看它有几个顶点
        epsilon = 0.04 * cv2.arcLength(cnt, True) # 精度系数，越小越贴合
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        
        # 正常的七巧板，顶点数应该是 3 或 4
        # 考虑到边缘圆角或膨胀变形，我们放宽到 3~6 之间
        if vertices < 3 or vertices > 8:
            print(f"轮廓 {i} 被剔除: 顶点数为 {vertices} (形状不规则)")
            continue

        # --- 过滤器 4: 颜色/亮度约束 (新增) ---
        # 计算轮廓内的平均亮度，区分黑色积木和白色背景/空隙
        # 优化：只在 ROI 区域计算，避免创建全图 mask
        x, y, w, h = cv2.boundingRect(cnt)
        roi_gray = gray[y:y+h, x:x+w]
        roi_mask = np.zeros((h, w), np.uint8)
        
        # 将轮廓坐标平移到 ROI 坐标系
        cnt_shifted = cnt - [x, y]
        cv2.drawContours(roi_mask, [cnt_shifted], -1, 255, -1)
        
        # 计算mask区域内的平均亮度
        mean_val = cv2.mean(roi_gray, mask=roi_mask)[0]
        
        # 假设积木是黑色的（亮度低），背景是白色的（亮度高）
        # 如果平均亮度 > 150，则认为是背景
        if mean_val > 150: 
            print(f"轮廓 {i} 被剔除: 亮度过高 ({mean_val:.1f}) (可能是背景/空隙)")
            continue
            
        # 通过所有测试，加入结果
        # --- 面积恢复 (新增) ---
        # 之前为了分割积木，我们膨胀了边缘 (iterations=2, kernel=3x3)，导致积木本身被腐蚀了约 2-3 像素。
        # 现在我们需要把这个面积补回来，同时保持尖角。
        restored_cnt = expand_polygon(cnt, offset=2.5)
        valid_components.append(restored_cnt)
        
        # --- 强行截断 (Top-K) ---
        # 如果我们已经找到了完美的 7 个，后面的就不要了
        # 这能防止把一些稍微大一点的噪点误判进来
        if len(valid_components) == 7:
            print("已收集满 7 个组件，停止搜索。")
            break

    if visualize:
        # 绘制结果
        output_img = img.copy()
        print(f"\n--- 最终检测结果 ({len(valid_components)} 个组件) ---")
        print(f"{'ID':<5} {'类型':<15} {'中心坐标(x,y)':<20} {'旋转角度':<10}")
        print("-" * 60)

        for i, cnt in enumerate(valid_components):
            # 标注重心
            M = cv2.moments(cnt)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            # 计算形状和角度
            shape_type, rotation = get_shape_and_angle(cnt)
            
            print(f"{i:<5} {shape_type:<15} ({cx}, {cy}){'':<10} {rotation:.1f}°")

            # 画轮廓
            cv2.drawContours(output_img, [cnt], 0, (0, 255, 0), 2)
            
            # 画多边形拟合线
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(output_img, [approx], 0, (0, 0, 255), 2)
            
            # 绘制中心点
            cv2.circle(output_img, (cx, cy), 5, (255, 0, 0), -1)
            
            # 绘制文字信息
            text = f"{shape_type[:3]} {rotation:.0f}"
            cv2.putText(output_img, text, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(output_img, text, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # print(f"最终确认: {len(valid_components)} 个组件")
        cv2.imshow("Final Result", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return valid_components, img

if __name__ == "__main__":
    # 运行
    # link_path = "./hub/datasets--lil-lab--kilogram/snapshots/ef45dfa1d58a83fc375cd025288d4e7f6ff9a5ed/training/images/val-color/page1-82_0.png"
    # real_path = os.path.realpath(link_path)
    # analyze_tangram_canny(real_path)

    real_path = hf_hub_download(repo_id="lil-lab/kilogram", filename="training/images/train-black/page1-82.png", repo_type="dataset")
    analyze_tangram_canny(real_path)