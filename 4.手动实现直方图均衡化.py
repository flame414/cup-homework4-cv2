import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

def manual_hist_equalization(image):
    """手动实现灰度图直方图均衡化"""
    h, w = image.shape
    # 1.计算原始直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    # 2.计算累积分布函数
    cdf = hist.cumsum()
    # 3.归一化CDF
    cdf_norm = cdf / cdf.max() * 255
    cdf_norm = cdf_norm.astype(np.uint8)
    # 4.映射像素值
    equalized = cdf_norm[image]
    # 计算均衡化后的直方图
    hist_eq, _ = np.histogram(equalized.flatten(), 256, [0, 256])
    return equalized, hist, hist_eq

# 读取图片
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, "test4.jpg")
img = cv2.imread(img_path, 0)
if img is None:
    raise FileNotFoundError("请确保test4.jpg在当前目录下！")

# 手动均衡化
img_eq, hist_ori, hist_eq = manual_hist_equalization(img)
# OpenCV均衡化对比
cv2_eq = cv2.equalizeHist(img)

# 可视化结果
plt.figure(figsize=(15, 10))
# 原始图像与直方图
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray'), plt.title('original image'), plt.axis('off')
plt.subplot(2, 3, 4), plt.plot(hist_ori), plt.title('original histogram'), plt.xlim([0, 256])
# 手动均衡化结果
plt.subplot(2, 3, 2), plt.imshow(img_eq, cmap='gray'), plt.title('manual histogram equalization'), plt.axis('off')
plt.subplot(2, 3, 5), plt.plot(hist_eq), plt.title('equalized histogram'), plt.xlim([0, 256])
# OpenCV均衡化结果
plt.subplot(2, 3, 3), plt.imshow(cv2_eq, cmap='gray'), plt.title('OpenCV histogram equalization'), plt.axis('off')
plt.subplot(2, 3, 6), plt.plot(np.histogram(cv2_eq.flatten(), 256, [0, 256])[0]), plt.title('OpenCV equalized hiatogram'), plt.xlim([0, 256])
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'hist_eq_result.png'))
plt.show()

print("直方图均衡化完成")
