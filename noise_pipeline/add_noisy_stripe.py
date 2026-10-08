import numpy as np


def add_stripe_noise(
    image,
    intensity=50,
    frequency=20,
    angle=0,
    noise_type="periodic",
    density=0.5,
    width_range=(1, 3),
    rng=None,
):
    """
    模拟遥感图像中的条纹噪声。
    参数:
        image: 输入图像
        intensity: 条纹强度
        frequency: 周期条纹的像素周期；随机条纹模式下不使用该参数
        angle: 条纹方向角度
        noise_type: 'periodic' (周期性) 或 'random' (随机条纹)
        density: 随机条纹在扫描方向上的覆盖比例
        width_range: 随机条纹宽度范围（含下界，不含上界）
        rng: 可选的 NumPy 随机数生成器
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    
    # 创建坐标网格
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    X, Y = np.meshgrid(x, y)
    
    # 角度旋转计算
    # 条纹方向通常与扫描方向垂直，通过调整angle模拟不同方向
    theta = np.radians(angle)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    
    # 生成条纹模式
    if noise_type == "periodic":
        # 周期性正弦条纹：frequency 表示像素周期。
        stripe_mask = intensity * np.sin(2 * np.pi * X_rot / frequency)
    elif noise_type == "random":
        # 在旋转后的坐标上生成随机条带，使 angle 对随机模式同样生效。
        stripe_mask = np.zeros((h, w), dtype=np.float64)
        u_min, u_max = float(X_rot.min()), float(X_rot.max())
        target_coverage = float(np.clip(density, 0.0, 1.0))
        mean_width = max(float(np.mean(width_range)), 1.0)
        mean_gap = max(mean_width * (1.0 - target_coverage) / max(target_coverage, 1e-6), 1.0)
        position = u_min
        while position < u_max:
            width = int(rng.integers(width_range[0], width_range[1]))
            width = max(width, 1)
            bias = float(rng.uniform(-intensity, intensity))
            band = (X_rot >= position) & (X_rot < position + width)
            stripe_mask[band] = bias + rng.normal(0.0, 2.0, size=int(band.sum()))
            jitter = float(rng.uniform(0.75, 1.25))
            position += width + max(mean_gap * jitter, 1.0)
    else:
        raise ValueError("noise_type must be 'periodic' or 'random'")
    
    # 叠加条纹噪声 (加性模型)
    if image.ndim == 3 and image.shape[2] == 3:
        stripe_mask_3d = np.stack([stripe_mask] * 3, axis=-1)
    else:
        stripe_mask_3d = stripe_mask
    noisy_image = image.astype(np.float64) + stripe_mask_3d
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)
