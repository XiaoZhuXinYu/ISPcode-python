# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import cj_bmpimage


def gaussian(x, sigma):
    # return np.exp(-(x ** 2) / (2 * (sigma ** 2))) / (2 * np.pi * (sigma ** 2))
    return np.exp(-(x ** 2) / (2 * (sigma ** 2)))  # 最终要进行归一化处理，底部系数可不参与计算，提高处理速度


def bilateral_filter(image, diameter, sigmaColor, sigmaSpace):

    img_height, img_width = image.shape
    new_image = np.zeros(image.shape)

    # 计算灰度值模板系数表
    weight_gray = np.zeros(256)  # 存放灰度差值的平方
    for i in range(256):
        weight_gray[i] = gaussian(i, sigmaColor)

    # 计算空间模板
    weight_space = np.zeros(diameter * diameter)  # 存放模板系数
    radius = diameter // 2
    maxk = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r_square = i * i + j * j
            weight_space[maxk] = gaussian(r_square, sigmaSpace)
            maxk = maxk + 1

    # 所有像素进行滤波
    for row in range(img_height):
        for col in range(img_width):
            wp_total = 0
            filtered_image = 0
            maxk = 0
            # 每个窗口的所有像素
            for k in range(diameter):
                for l in range(diameter):
                    # 窗口像素的绝对距离
                    n_x = row - (diameter // 2 - k)
                    n_y = col - (diameter // 2 - l)

                    if n_x < 0 or n_x >= img_height:  # row 的取值为0- height-1，所以这边判断需要=
                        n_x = row
                    if n_y < 0 or n_y >= img_width:
                        n_y = col
                    data = image[n_x][n_y]
                    pixel = image[row][col]

                    # 值域的权重
                    gi = weight_gray[np.abs(int(data) - int(pixel))]

                    # 空间域的权重
                    gs = weight_space[maxk]  # 边界处值域权重为0，空域不做特殊处理
                    wp = gi * gs
                    filtered_image = filtered_image + (image[n_x][n_y] * wp)
                    wp_total = wp_total + wp
                    maxk = maxk + 1
            filtered_image = filtered_image // wp_total  # 整除，比如得到2.0而不是2，所以后面要转成int型
            new_image[row][col] = int(filtered_image)
    return new_image


if __name__ == "__main__":
    print('This is main of module')
    file_name1 = "../pic/lena_gray_noised.bmp"

    image = cj_bmpimage.read_bmpimage(file_name1, 512, 512, dtype="uint8")
    filtered_image1 = bilateral_filter(image, 7, 10, 10)
    cj_bmpimage.show_bmpimage(filtered_image1, 512, 512, sensorbit=8, compress_ratio=1)


