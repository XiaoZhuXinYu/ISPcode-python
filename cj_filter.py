import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from scipy import signal


# 坐标系转换,把左上角的坐标系转成中心点的坐标系
def spilt(a):
    if a/2 == 0:
        x1 = x2 = a/2
    else:
        x1 = math.floor(a/2)
        x2 = a - x1
    return -x1, x2  # 当a为5时，返回-2,3, 在进行for循环时，刚好包含-2到2,3不包括在内，这样就刚好对称了


# 高斯滤波核的生成,跟元素距离中心元素的距离有关
def gaussian_box(a, b, sigma=10):
    sum1 = 0
    box = []
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    for i in range(x1, x2):
        for j in range(y1, y2):
            t = i*i + j*j
            re = math.e ** (-t/(2*sigma*sigma))  # 由于后续要进行归一化，所以高斯滤波的系数可以忽略
            sum1 = sum1 + re  # sum是为了把所有权重加起来，再做归一化
            box.append(re)
    box = np.array(box)
    box = box / sum1  # 归一化操作
    box.shape = (a, b)
    return box


# 自己实现的filter
def self_filter(img, fil):
    # opencv 实现方式
    # res = cv2.filter2D(img, -1, fil)
    # return rs

    # 卷积核翻转
    fil = np.fliplr(fil)  # 横向翻转
    fil = np.flipud(fil)  # 上下翻转

    # 结果图片的空间分配
    image_size = np.shape(img)
    res = np.zeros(image_size)

    # 对三个颜色通道做卷积
    res[:, :, 0] = signal.convolve(img[:, :, 0], fil, mode="same")  # 使用signal的卷积函数
    res[:, :, 1] = signal.convolve(img[:, :, 1], fil, mode="same")  # 使用signal的卷积函数
    res[:, :, 2] = signal.convolve(img[:, :, 2], fil, mode="same")  # 使用signal的卷积函数
    # res = cv2.filter2D(img, -1, fil)
    return res


if __name__ == "__main__":
    img1 = plt.imread("kodim19.png")  # 在这里读取图片
    plt.imshow(img1)  # 显示读取的图片
    plt.show()
   
    # 5X5窗口大小的高斯滤波
    fil2 = gaussian_box(3, 3, 1)
    # fil2 = gaussian_box(5, 5)
    res2 = self_filter(img1, fil2)
    plt.figure()
    plt.imshow(res2)  # 显示卷积后的图片
    # plt.imsave("res2.jpg",res2)
    plt.show()
