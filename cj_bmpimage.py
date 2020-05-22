import numpy as np
import math
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


def bmp_image_show_gray(image, width, height, compress_ratio=1):
    x = width / (compress_ratio * 100)
    y = height / (compress_ratio * 100)
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(image, cmap='gray', interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    print('show gray image')


def test_read_bmpimage(image1, width, height, dtype, sensorbit, compress_ratio=1):
    image = np.fromfile(image1, dtype)  # 相比于open，该函数可以指定数据类型
    image = image[1078:]
    image.shape = [height, width]
    testimage = image
    print(image)
    if sensorbit == 8:
        testimage = testimage / 255  # 8bit sensor 所以是除255，为了和下面函数中 vmax=1进行配合

    bmp_image_show_gray(testimage, width, height, compress_ratio=1)


def test_show_bmp_histogram(image1, dtype, width, height, start_x, start_y, len_x, len_y, step_x, step_y):
    image = np.fromfile(image1, dtype)  # 相比于open，该函数可以指定数据类型
    image = image[1078:]
    image.shape = [height, width]
    testimage = image[start_y:(len_y + start_y):step_y, start_x:(len_x + start_x):step_x]
    # array_bins = np.arange(0, 256, 1)
    # array_bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256])
    array_bins = np.array([0, 50, 100, 150, 200, 256])
    # array_bins = np.array([0, 32, 64, 96, 128, 160, 192, 224, 256])
    testimage_flatten = testimage.flatten()  # 将二维数组转成一维数组
    n = plt.hist(testimage_flatten, bins=array_bins)  # 第一个参数必须是一个一维数组
    print("testimage_flatten.mean:", testimage_flatten.mean())
    plt.title("histogram")
    plt.show()


def test_show_bf3a03_gamma():
    x = np.array([0, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 255])
    xnew = np.linspace(0, 255, 2551)
    x______ = np.array([0, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 255])  # x轴
    y_moren = np.array([0, 9, 17, 37, 74, 103, 126, 145, 163, 179, 192, 203, 214, 232, 246, 258])  # 默认
    y__qxll = np.array([0, 6, 14, 29, 61, 90, 115, 138, 161, 181, 199, 214, 226, 242, 254, 260])  # 清新亮丽
    y__test = np.array([0, 6, 14, 29, 61, 90, 115, 138, 161, 185, 205, 220, 232, 248, 258, 262])  # 测试
    y_dz = np.array([0, 9, 21, 39, 68, 94, 114, 131, 145, 158, 170, 181, 190, 208, 224, 238])  # 低噪
    y_gbgdh = np.array([0, 6, 17, 37, 69, 91, 107, 122, 137, 151, 161, 172, 181, 199, 215, 227])  # 过曝过度好

    f = interpolate.interp1d(x, x, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="no gamma")

    f = interpolate.interp1d(x, y_moren, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="default")

    f = interpolate.interp1d(x, y__qxll, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="test1")

    f = interpolate.interp1d(x, y__test, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="test2")

    f = interpolate.interp1d(x, y_dz, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="test3")

    f = interpolate.interp1d(x, y_gbgdh, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="test4")

    plt.title("gamma")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    print('This is main of module')
    file_name1 = "../pic/GAIN7EXP2.0_92.bmp"
    # file_name1 = "000001.bmp"

    # test_read_bmpimage(file_name1, 640, 480,  dtype="uint8", sensorbit=8, compress_ratio=1)
    test_show_bmp_histogram(file_name1, dtype="uint8", width=640, height=480, start_x=160, start_y=120, len_x=320,
                            len_y=240, step_x=2, step_y=2)

    # test_show_bf3a03_gamma()
