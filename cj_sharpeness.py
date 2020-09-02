from scipy import signal  # for convolutions
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import cj_csc as csc
import cj_rgbimage as rgbimage


def soft_coring(RGB, slope, tau_threshold, gamma_speed):
    # Usage(用途)：用于锐化掩模锐化过程。
    # slope(斜率)：斜率越大，锐化越重。
    # tau_threshold：图像不锐化的阈值。tau_threshold的值越低，越多的频率被锐化。
    # gamma_speed：控制收敛速度，斜率越小，图像越锐化，这可能是一个很好的调谐器。

    return slope * RGB * (1. - np.exp(-((np.abs(RGB / tau_threshold)) ** gamma_speed)))


def gaussian(kernel_size, sigma):
    temp = np.floor(np.float32(kernel_size) / 2.)  # 向下取整
    x, y = np.meshgrid(np.linspace(-temp[0], temp[0], kernel_size[0]), np.linspace(-temp[1], temp[1], kernel_size[1]))
    # example: if kernel_size = [5, 3], then:
    # x: array([[-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.]])
    # y: array([[-1., -1., -1., -1., -1.],
    #           [ 0.,  0.,  0.,  0.,  0.],
    #           [ 1.,  1.,  1.,  1.,  1.]])
    # Gaussian equation
    temp = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    # make kernel sum equal to 1
    return temp / np.sum(temp)


def sharpen_gaussian(RGB, gaussian_kernel_size=[5, 5], gaussian_sigma=2.0, slope=1.5, tau_threshold=0.02,
                     gamma_speed=4., clip_range=[0, 255]):
    # Objective: sharpen image
    #   gaussian_kernel_size: 高斯模糊滤波核的大小
    #
    #   gaussian_sigma:  sigma 越大，锐化越强
    #
    #   slope(斜率):  斜率越大，锐化越强
    #
    #   tau_threshold:   图像不锐化的阈值。tau_threshold的值越低，越多的频率被锐化。
    #
    #   gamma_speed:   控制收敛速度，斜率越小，图像越锐化，这可能是一个很好的调谐器。

    print("----------------------------------------------------")
    print("Running sharpening by unsharp masking...")

    # create gaussian kernel
    gaussian_kernel = gaussian(gaussian_kernel_size, gaussian_sigma)

    if np.ndim(RGB) > 2:
        image_blur = np.empty(np.shape(RGB), dtype=np.float32)
        for i in range(0, np.shape(RGB)[2]):
            image_blur[:, :, i] = signal.convolve2d(RGB[:, :, i], gaussian_kernel, mode="same", boundary="symm")
    else:
        image_blur = signal.convolve2d(RGB, gaussian_kernel, mode="same", boundary="symm")

    # the high frequency component image
    image_high_pass = RGB - image_blur

    tau_threshold = tau_threshold * clip_range[1]

    return np.clip(RGB + soft_coring(image_high_pass, slope, tau_threshold, gamma_speed), clip_range[0], clip_range[1])


def sharpen_convolove(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    return dst


def sharpen_bilateralFilter(RGB):
    d = 5  # kernel size
    sigmaColor = 20  # color domain sigma
    sigmaSpace = 20  # space domain sigma
    weight = 3
    # weight_ratio=0.3
    h, w, c = RGB.shape
    ycc = csc.rgb2ycbcr(RGB, w, h)
    ycc_out = ycc
    y = ycc[:, :, 0]
    cb = ycc[:, :, 1]
    cr = ycc[:, :, 2]
    y_bilateral_filtered = cv.bilateralFilter(y.astype(np.float32), d, sigmaColor, sigmaSpace)
    detail = ycc[:, :, 0] - y_bilateral_filtered
    y_out = y_bilateral_filtered + weight * detail
    y_out = np.clip(y_out, 0, 255)
    # y_out = (1-weight_ratio)*y_bilateral_filtered + weight_ratio * detail
    ycc_out[:, :, 0] = y_out
    rgb_out = csc.ycbcr2rgb(ycc_out, w, h)
    return rgb_out


if __name__ == "__main__":
    maxvalue = 255
    LUT_SIZE = 17
    pattern = "GRBG"
    image = plt.imread('crop_bgr_0.jpg')
    if np.max(image) <= 1:
        image = image * maxvalue
    # new_image=sharpen_bilateralFilter(image)
    new_image = sharpen_gaussian(image)
    rgbimage.rgb_image_show_color(image, maxvalue=255, color="color", compress_ratio=1)
    rgbimage.rgb_image_show_color(new_image, maxvalue=255, color="color", compress_ratio=1)
