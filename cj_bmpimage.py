import numpy as np
import math
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


def read_bmpimage(image1, width, height, dtype):
    image = np.fromfile(image1, dtype)  # 相比于open，该函数可以指定数据类型
    image = image[1078:]
    image.shape = [height, width]

    testimage = np.zeros(image.shape)
    # testimage = image  # 直接进行赋值，下面的for循环后，得不到想要的结果，可能和若拷贝有关

    for i in range(height):
        testimage[i] = image[height-1-i]

    return testimage


def show_bmpimage(image, width, height, sensorbit, compress_ratio=1):
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
    plt.imshow(image, cmap='gray', interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    print('show gray image')


def test_show_bmp_histogram(image1, dtype, width, height, start_x, start_y, len_x, len_y, step_x, step_y, num):
    image = np.fromfile(image1, dtype)  # 相比于open，该函数可以指定数据类型
    image = image[1078:]
    image.shape = [height, width]
    testimage = image[start_y:(len_y + start_y):step_y, start_x:(len_x + start_x):step_x]
    # array_bins = np.arange(0, 256, 255 / num)  # 等差数列数组支持任意个数组元素
    array_bins = np.array([0, 40, 100, 170, 250, 256])  # 特殊数组单独添加测试
    testimage_flatten = testimage.flatten()  # 将二维数组转成一维数组
    n = plt.hist(testimage_flatten, bins=array_bins)  # 第一个参数必须是一个一维数组
    np.set_printoptions(precision=1, suppress=True)  # 设置输出小数点位数 取消科学计数法
    print("平均灰度:", format(testimage_flatten.mean(), '.1f'))  # 上一句设置小数点位数对它无效，进行另外设置
    print("直方图统计个数:", n[0])
    percent = np.zeros(num)
    for i in range(0, num):
        percent[i] = n[0][i] / n[0].sum()
    print("直方图统计百分比:", percent * 100)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 添加这两句支持plt中文显示

    plt.xlabel('灰度值')
    plt.ylabel('个数')
    plt.title('histogram')
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
    file_name1 = "../pic/GAIN7EXP2.0_99.bmp"
    # file_name1 = "000001.bmp"

    # image = read_bmpimage(file_name1, 640, 480, dtype="uint8")
    # show_bmpimage(image, 640, 480, sensorbit=8, compress_ratio=1)

    test_show_bmp_histogram(file_name1, dtype="uint8", width=640, height=480, start_x=160, start_y=120, len_x=320,
                            len_y=240, step_x=2, step_y=2, num=5)

    # test_show_bf3a03_gamma()
