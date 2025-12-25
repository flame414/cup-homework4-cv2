import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def sobel_filter(image):
    """手动实现Sobel梯度算子"""
    # Sobel核（x和y方向）
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    h, w = image.shape
    pad_size = 1
    # 零填充
    padded = np.pad(image, pad_size, mode='constant')
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)
    
    # 卷积计算梯度
    for y in range(h):
        for x in range(w):
            grad_x[y, x] = np.sum(padded[y:y+3, x:x+3] * sobel_x)
            grad_y[y, x] = np.sum(padded[y:y+3, x:x+3] * sobel_y)
    
    # 梯度幅值
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # 归一化到0-255
    grad_mag = (grad_mag / grad_mag.max()) * 255
    grad_mag = grad_mag.astype(np.uint8)
    return grad_x, grad_y, grad_mag

# 读取图片
current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, "test1.jpg")
img = cv2.imread(img_path, 0)  # 灰度图
if img is None:
    raise FileNotFoundError("请确保test1.jpg在当前目录下！")

# Sobel滤波
grad_x, grad_y, grad_mag = sobel_filter(img)

# 可视化
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('original'), plt.axis('off')
plt.subplot(132), plt.imshow(grad_mag, cmap='gray'), plt.title('Sobel gradient magnitude'), plt.axis('off')
plt.subplot(133), plt.imshow(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3), cmap='gray'), plt.title('OpenCV Sobel X'), plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'sobel_result.png'))
plt.show()

print("Sobel梯度算子滤波完成")
