import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from scipy import signal


def read_bmpimage(image1, width, height, dtype):
    image = np.fromfile(image1, dtype)  # 相比于open，该函数可以指定数据类型
    image = image[1078:]
    image.shape = [height, width]

    testimage = np.zeros(image.shape)
    # testimage = image  # 直接进行赋值，下面的for循环后，得不到想要的结果，可能和若拷贝有关
    # 采图工具采集到的图电脑上显示是正常的，但是数据确是和正常bmp图像上下对称的。
    # 也就是说两张图像的数据是关于x轴对称，但是电脑上显示的图却是一样的。特别记下。
    # 关于bmp图像的数据格式，我也没有过多研究，这边知道有这个问题即可，不必深究。
    for i in range(height):
        testimage[i] = image[height-1-i]

    return testimage


def show_bmpimage(image, width, height, sensorbit, color="color", compress_ratio=1):
    if sensorbit == 8:
        image = image / 255  # 8bit sensor 所以是除255，为了和下面函数中 vmax=1进行配合
    elif sensorbit == 10:
        image = image / 1023  # 10bit sensor 所以是除1023，为了和下面函数中 vmax=1进行配合
    elif sensorbit == 12:
        image = image / 4095  # 12bit sensor 所以是除4095，为了和下面函数中 vmax=1进行配合
    else:
        image = image / 4095  # 12bit sensor 所以是除4095，为了和下面函数中 vmax=1进行配合

    x = width / (compress_ratio * 100)
    y = height / (compress_ratio * 100)
    plt.figure(num='test', figsize=(x, y))
    if color == "gray":
        plt.imshow(image, cmap='gray', interpolation='bicubic', vmax=1.0)
    else:
        plt.imshow(image, interpolation='bicubic', vmax=1.0)

    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    # print('show gray image')


def rgb_separation(image):
    image = image.astype(np.float)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return R, G, B


if __name__ == "__main__":
    print('This is main of module')
    # file_name1 = "../pic/GAIN7EXP2.0_0.bmp"
    file_name1 = "../pic/pic_14.bmp"

    image = read_bmpimage(file_name1, 640, 480, dtype="uint8")
    show_bmpimage(image, 640, 480, sensorbit=8, compress_ratio=1)



