import numpy as np
import cv2

def add_sensor_noise(image, gaussian_sigma=10, apply_poisson=True, photon_scale=100.0, rng=None):
    """
    模拟传感器噪声：高斯读出噪声与信号相关的泊松噪声。
    参数:
        image: 输入图像 [0, 255]
        gaussian_sigma: 高斯噪声的标准差
        apply_poisson: 是否应用泊松噪声
    """
    if rng is None:
        rng = np.random.default_rng()

    image_float = image.astype(np.float64)
    noisy_image = image_float.copy()
    
    # 1. 泊松计数强度为 image / 255 * photon_scale。
    if apply_poisson:
        # 为了模拟光子计数过程，我们将像素值视为期望值
        # 注意：OpenCV的饱和处理
        # 使用Numpy的poisson函数
        # 这里需要对每个像素进行采样
        # scale用于调整模拟的“光子数”量级，值越小噪声越明显
        scale_factor = float(photon_scale)
        if scale_factor <= 0:
            raise ValueError("photon_scale must be positive")
        noisy_image = rng.poisson(noisy_image / 255.0 * scale_factor) / scale_factor * 255.0
        
    # 2. 高斯读出噪声。
    # 加性噪声，模拟读出噪声
    gaussian_noise = rng.normal(0, gaussian_sigma, image.shape)
    noisy_image = noisy_image + gaussian_noise
    
    # 3. 截断与类型转换
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)
