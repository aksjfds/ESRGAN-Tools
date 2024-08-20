import cv2
import numpy as np


# Blur
def create_sinc_kernel(size, cutoff):
    """创建一个Sinc滤波器核"""
    # 创建一个网格
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)

    # 计算径向距离
    r = np.sqrt(x**2 + y**2)

    # 计算 sinc 函数
    kernel = np.sinc(2 * cutoff * r)

    # 归一化核使得其所有元素之和为1
    kernel /= np.sum(kernel)

    return kernel


def Blur(img, method="Sinc", ksize=0, param=0, kernel=None):
    if method == "Gauss":
        if ksize == 0:
            ksize = 7
        return cv2.GaussianBlur(img, (ksize, ksize), param)
    else:
        if ksize == 0:
            ksize = 19
        if param == 0:
            param = 0.1
        if not kernel:
            kernel = create_sinc_kernel(
                ksize, param)   # 如果sinc核不随机的话, 就传进来kernel
        return cv2.filter2D(img, -1, kernel)


# Resize
interpolationFlags = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR]
methods = ["up", "down", "keep"]

def Resize(img):
    method = methods[np.random.randint(0, 3)]
    scale = 1
    match method:
        case "up":
            scale = np.random.uniform(1, 1.5)
        case "down":
            scale = np.random.uniform(0.5, 1)

    interpolation = interpolationFlags[np.random.randint(0, 3)]
    return cv2.resize(
        img,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=interpolation)


# Noise
def Noise(img, noiseType="Gauss", isGray=True, mean=0, std=10, lam=7.):
    if noiseType == "Gauss":
        noise = np.random.normal(mean, std, img.shape)
        if isGray:
            noise = noise[:, :, 0]
            noise = np.stack([noise] * 3, axis=-1)
        noisy_image = img + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    else:
        noise = np.random.poisson(lam, img.shape)
        noisy_image = img + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image


# JPEG
def JPEGCompression(img, compression_quality=20):
    result, encoded_image = cv2.imencode(
        '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])

    return cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
