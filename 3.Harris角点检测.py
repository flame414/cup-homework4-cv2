import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(img, blockSize=3, ksize=3, k=0.04, threshold=0.05):
    """
    Harris角点检测
    """
    blockSize = blockSize if blockSize % 2 == 1 else blockSize + 1
    blockSize = max(1, blockSize)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    
    # 1. 高斯滤波
    img_blur = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # 2. 计算梯度
    Ix = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # 3. 计算梯度平方与乘积
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # 4. 高斯滤波平滑
    Ix2_blur = cv2.GaussianBlur(Ix2, (blockSize, blockSize), 1)
    Iy2_blur = cv2.GaussianBlur(Iy2, (blockSize, blockSize), 1)
    Ixy_blur = cv2.GaussianBlur(Ixy, (blockSize, blockSize), 1)
    
    # 5、6. 构造二阶矩矩阵并计算响应值R
    det_M = Ix2_blur * Iy2_blur - Ixy_blur ** 2
    trace_M = Ix2_blur + Iy2_blur
    R = det_M - k * (trace_M ** 2)
    
    # 7.阈值筛选候选角点
    R_max = np.max(R)
    corner_mask = R > threshold * R_max  # 相对阈值
    
    # 8.非极大值抑制
    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(R, kernel)
    nms_mask = (R == local_max) & corner_mask  # 同时满足高响应+局部最大
    
    # 获取最终角点坐标
    corners = np.argwhere(nms_mask)
    
    # 可视化：绘制角点
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y, x in corners:
        cv2.circle(img_color, (x, y), 2, (0, 0, 255), -1)  
    
    return img_color, corners

if __name__ == "__main__":
    # 加载图像
    img = cv2.imread("test3.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("图像加载失败，请检查路径！")
    
    # 调用修复后的函数
    harris_img, corners = harris_corner_detection(img, blockSize=3, k=0.04, threshold=0.08)
    
    # 对比OpenCV官方函数
    img_official = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dst = cv2.cornerHarris(img, blockSize=3, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img_official[dst > 0.08 * dst.max()] = [0, 0, 255] 
    
    # 可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img, cmap="gray"), plt.title("original image"), plt.axis("off")
    plt.subplot(132), plt.imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB)), plt.title("hand-on Harris"), plt.axis("off")
    plt.subplot(133), plt.imshow(cv2.cvtColor(img_official, cv2.COLOR_BGR2RGB)), plt.title("OpenCV Harris"), plt.axis("off")
    plt.savefig("harris_fixed_result.png", bbox_inches="tight")
    plt.show()
    
    print(f"检测到的角点数量：{len(corners)}")
    # 分析
    print("分析：增大窗口会使算法忽略小尺度纹理的角点，更关注大结构的角点特征，角点数量随窗口增大而减少")
