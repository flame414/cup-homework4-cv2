import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def canny_edge_detection(image, sigma=1, low_thresh=30, high_thresh=90):
    """Canny边缘检测"""
    # 1.高斯滤波去噪
    blur = cv2.GaussianBlur(image, (5, 5), sigma)
    
    # 2.计算梯度幅值和方向（Sobel）
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi  
    
    # 3.非极大值抑制（NMS）
    h, w = image.shape
    nms = np.zeros_like(grad_mag)
    for y in range(1, h-1):
        for x in range(1, w-1):
            angle = grad_dir[y, x]
            # 归一化角度到0-180
            if angle < 0:
                angle += 180
            # 四个方向：0°, 45°, 90°, 135°
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [grad_mag[y, x-1], grad_mag[y, x+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [grad_mag[y-1, x+1], grad_mag[y+1, x-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [grad_mag[y-1, x], grad_mag[y+1, x]]
            else:
                neighbors = [grad_mag[y-1, x-1], grad_mag[y+1, x+1]]
            # 非极大值抑制
            if grad_mag[y, x] >= max(neighbors):
                nms[y, x] = grad_mag[y, x]
    
    # 4.双阈值检测与边缘连接
    nms = (nms / nms.max()) * 255
    strong_edges = (nms >= high_thresh)
    weak_edges = (nms >= low_thresh) & (nms < high_thresh)
    # 边缘连接
    edges = np.zeros_like(nms, dtype=np.uint8)
    edges[strong_edges] = 255
    # 8邻域检测弱边缘是否连接强边缘
    kernel = np.ones((3, 3), dtype=np.uint8)
    strong_neighbors = cv2.dilate(strong_edges.astype(np.uint8), kernel, iterations=1)
    edges[weak_edges & (strong_neighbors > 0)] = 255
    
    return blur, grad_mag, nms, edges

# 读取图片
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, "test2.jpg")
img = cv2.imread(img_path, 0)
if img is None:
    raise FileNotFoundError("请确保test.jpg在当前目录下！")

# 手动Canny检测
blur, grad_mag, nms, edges = canny_edge_detection(img)
# OpenCV自带Canny对比
cv2_canny = cv2.Canny(img, 30, 90)

# 可视化各环节结果
plt.figure(figsize=(20, 5))
plt.subplot(151), plt.imshow(img, cmap='gray'), plt.title('original'), plt.axis('off')
plt.subplot(152), plt.imshow(blur, cmap='gray'), plt.title('1.gaussian filter'), plt.axis('off')
plt.subplot(153), plt.imshow(grad_mag, cmap='gray'), plt.title('2.gradient magnitude'), plt.axis('off')
plt.subplot(154), plt.imshow(nms, cmap='gray'), plt.title('3.non-maximum suppression'), plt.axis('off')
plt.subplot(155), plt.imshow(edges, cmap='gray'), plt.title('4.Double-threshold edge detection'), plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'canny_steps.png'))

# 对比手动与OpenCV结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(edges, cmap='gray'), plt.title('hand Canny edge'), plt.axis('off')
plt.subplot(122), plt.imshow(cv2_canny, cmap='gray'), plt.title('OpenCV Canny edge'), plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'canny_comparison.png'))
plt.show()

print("Canny边缘检测完成")
