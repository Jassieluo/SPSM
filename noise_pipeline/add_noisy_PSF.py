import numpy as np
import cv2
from scipy import signal

def make_gaussian_psf(kernel_size, sigma):
    """
    构建归一化的高斯点扩散函数（PSF）。
    参数:
        kernel_size: 核大小，奇数
        sigma: 高斯标准差，控制模糊强度
    """
    # 创建坐标网格
    x = np.arange(0, kernel_size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = kernel_size // 2
    
    # 高斯公式
    # 使用高斯核作为空间模糊的受控近似。
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    # 归一化，确保能量守恒
    return g / g.sum()

def add_atmospheric_blur(image, sigma=1.0, kernel_size=15):
    """
    通过高斯 PSF 模拟空间模糊。
    参数:
        sigma: 高斯标准差，值越大模糊越严重。
    """
    # 1. 构建PSF
    psf = make_gaussian_psf(kernel_size, sigma)
    
    # 2. 卷积操作
    # 使用OpenCV的filter2D函数，depth=-1表示输出深度与输入相同
    # 边界处理使用反射边界。
    blurred_image = cv2.filter2D(image, -1, psf, borderType=cv2.BORDER_REFLECT)
    
    return blurred_image
