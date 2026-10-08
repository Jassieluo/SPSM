import numpy as np
import cv2
from noise import pnoise2


def generate_cloud_texture(shape, scale=100, octaves=4, persistence=0.5, rng=None):
    """
    生成基于 Perlin 噪声的云层纹理遮罩。
    """
    h, w = shape
    if rng is None:
        rng = np.random.default_rng()
    texture = np.zeros((h, w), dtype=np.float64)
    # 每张图片使用不同但可复现的空间偏移，避免所有样本共享同一云纹理。
    base_x = int(rng.integers(0, 100000))
    base_y = int(rng.integers(0, 100000))
    # 多层Perlin噪声混合，增加细节
    for o in range(octaves):
        freq = 2 ** o
        amp = persistence ** o
        for i in range(h):
            for j in range(w):
                texture[i][j] += amp * pnoise2(
                    (i + base_x) / (scale / freq),
                    (j + base_y) / (scale / freq),
                    octaves=1,
                    base=0,
                )
    # 归一化到 [0, 1]
    lo, hi = float(np.min(texture)), float(np.max(texture))
    if hi - lo < 1e-12:
        return np.zeros_like(texture)
    texture = (texture - lo) / (hi - lo)
    return texture


def add_clouds(image, density=0.5, scale=100, rng=None):
    """
    在遥感图像上叠加云层。
    参数:
        density: 云层密度/不透明度
    """
    h, w = image.shape[:2]
    cloud_mask = generate_cloud_texture(
        (h, w), scale=scale, octaves=6, persistence=0.55, rng=rng
    )
    # 柔和阈值处理，sigmoid函数模拟云体边缘
    import scipy.special
    cloud_mask = scipy.special.expit((cloud_mask - 0.5) * 8)  # sigmoid拉伸
    # 高斯模糊柔化边缘
    cloud_mask = cv2.GaussianBlur(cloud_mask, (0, 0), sigmaX=3, sigmaY=3)
    # 调整云的不透明度
    cloud_mask = cloud_mask * density
    # 云色微蓝灰而非纯白
    # OpenCV images are BGR; this is the BGR equivalent of a pale blue-gray.
    cloud_color = np.array([1.0, 0.96, 0.92])
    image_float = image.astype(np.float64) / 255.0
    cloud_mask_3d = np.stack([cloud_mask]*3, axis=-1)
    cloudy_image = image_float * (1 - cloud_mask_3d) + cloud_mask_3d * cloud_color
    return np.clip(cloudy_image * 255, 0, 255).astype(np.uint8)
