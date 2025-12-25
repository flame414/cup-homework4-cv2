项目概述
本项目完成数字图像处理的四道核心实验：梯度算子实现与滤波、Canny 边缘检测、Harris 角点检测、直方图均衡化。所有算法均手动实现核心逻辑（允许调用 OpenCV 基础库函数），并通过可视化展示各环节处理结果，同时分析关键参数对算法效果的影响。
实验题目与实现说明
题目 1：梯度算子实现与图像滤波

    目标：实现 Sobel 梯度算子，对图像进行 x/y 方向梯度滤波并可视化结果。
    核心步骤：
        图像灰度化与高斯平滑预处理；
        实现 Sobel 算子的卷积核（3×3），分别计算 x、y 方向梯度；
        融合 x/y 梯度得到整体梯度图，可视化单方向梯度与融合梯度结果。
    输出：原始图、x 方向梯度图、y 方向梯度图、整体梯度图。

题目 2：手动实现 Canny 边缘检测

    目标：严格遵循 Canny 算法流程实现边缘检测，展示各环节结果。
    核心步骤：
        高斯滤波去噪；
        Sobel 算子计算梯度幅值与方向；
        非极大值抑制（NMS）细化边缘；
        双阈值法筛选强 / 弱边缘并完成边缘连接。
    输出：高斯滤波图、梯度幅值图、NMS 处理图、最终边缘检测图。

题目 3：手动实现 Harris 角点检测

    目标：实现 Harris 角点检测算法，分析窗口参数对检测结果的影响。
    核心步骤：
        小方差高斯滤波预处理图像；
        计算 x/y 方向梯度及梯度平方、乘积项；
        高斯平滑梯度统计量，构造二阶矩矩阵M；
        计算 Harris 响应值R，通过阈值筛选候选角点；
        非极大值抑制（NMS）获取最终角点，分析blockSize参数对检测效果的影响。
    输出：角点检测结果图、不同窗口大小的检测效果对比分析。

题目 4：手动实现直方图均衡化

    目标：实现灰度图像的直方图均衡化，提升图像对比度。
    核心步骤：
        计算原始图像的灰度直方图；
        求解累积分布函数（CDF）并完成灰度值映射；
        生成均衡化后的图像，计算均衡化后的直方图。
    输出：原始图像、均衡化图像、原始直方图、均衡化直方图。

环境依赖

    Python 3.10.19
    OpenCV-Python (cv2)
    NumPy
    Matplotlib

安装命令：
bash

pip install opencv-python numpy matplotlib

代码结构
plaintext

├── gradient_operator.py  # 题目1：梯度算子实现与滤波
├── canny_edge_detection.py  # 题目2：Canny边缘检测
├── harris_corner_detection.py  # 题目3：Harris角点检测
├── histogram_equalization.py  # 题目4：直方图均衡化
├── images/  # 存放测试图像（如test.jpg、chessboard.png）
├── results/  # 存放各实验的可视化结果图
└── README.md  # 项目说明文档

运行说明

    将测试图像放入images文件夹，替换代码中对应的图像路径；
    分别运行各实验脚本，示例：
    bash

    python gradient_operator.py
    python canny_edge_detection.py
    python harris_corner_detection.py
    python histogram_equalization.py

    运行后在results文件夹查看可视化结果，控制台会输出关键参数分析与统计信息。

关键参数说明
  实验	                     核心参数	                                   作用与推荐值
梯度算子	         ksize	                              Sobel 核尺寸，推荐 3（奇数）
Canny 边缘检测	   sigma/low_thresh/high_thresh       	高斯滤波标准差（0.5~1）、双阈值（0.1/0.3）
Harris 角点检测 	 blockSize/k/threshold              	邻域大小（3/5/7）、响应系数（0.04）、筛选阈值（0.08）
直方图均衡化        无	                                无参数，基于图像灰度分布自动计算


实验结论

    梯度算子是边缘与角点检测的基础，Sobel 算子通过卷积能有效提取图像灰度变化趋势；
    Canny 边缘检测的非极大值抑制与双阈值法可显著提升边缘的准确性与连续性；
    Harris 角点检测的blockSize参数直接影响角点检测的密度，窗口越大检测角点越少但鲁棒性越强；
    直方图均衡化通过重新分配灰度值，能有效改善低对比度图像的视觉效果。
