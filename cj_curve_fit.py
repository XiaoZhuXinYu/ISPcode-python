import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq


def Fun_curve_fit(x, a1, a2, a3):  # 定义拟合函数 采用curve_fit，x必须是第一个参数
    return a1 * x ** 2 + a2 * x + a3


def Fun_leastsq(p, x):  # 定义拟合函数形式
    a1, a2, a3 = p
    return a1 * x ** 2 + a2 * x + a3


def error(p, x, y):  # 拟合残差
    return Fun_leastsq(p, x) - y


# 画样本点
def test_draw_samplepoint():
    Xi = np.array(
        [8.3, 10.8, 13.1, 17.5, 27.5, 37.9, 46.5, 52.6, 56.2, 57.9, 59.9, 61.1, 62.8, 64.2, 65.5, 66.9, 68, 68.9, 70,
         70.7, 71.4, 71.5, 70.7, 67.7, 66.7, 65])
    xdata1 = np.array(
        [6.6, 9.2, 11.3, 14.4, 23.6, 32.1, 40.8, 46.4, 49.8, 52.4, 55.2, 57.6, 60, 62, 63.6, 65.1, 66.3, 66.5, 65.3,
         63.2, 62.1, 61.5, 61.3, 60.8, 60.5, 60.1])
    xdata2 = np.array(
        [5.8, 7.1, 10.5, 11.8, 12.8, 13.5, 13.5, 14.3, 14.8, 15.5, 16.3, 16.5, 16.7, 16.7, 16.9, 16.9, 16.9, 17.1, 18.8,
         20.5, 21.3, 21.9, 22.1, 22, 23, 23.3])
    xdata3 = np.array(
        [36.4, 52.6, 50.3, 31.7, 31.3, 27.6, 31.8, 30.7, 26.9, 21, 19, 15.2, 14.9, 12.9, 10.9, 10.1, 9.4, 8.5, 8.8, 9.3,
         10.2, 10.9, 11.1, 11.5, 11.4, 11.4])
    xdata4 = np.array(
        [51, 31.1, 28, 42.1, 32.2, 26.8, 13.9, 8.6, 8.4, 11, 9.5, 10.6, 8.4, 8.4, 8.5, 7.9, 7.3, 7.8, 7.1, 7, 6.3, 5.6,
         5.4, 5.7, 5, 5.2])
    Yi = np.array(
        [161, 140.4, 133, 138.1, 118.4, 103, 85.8, 75.6, 70, 66.8, 62.4, 60, 56.6, 54.6, 53.1, 51.5, 50.5, 50.3, 49.6,
         49.5, 49.1, 48.6, 48.4, 49.2, 48.6, 49.2])
    Zi = np.array(
        [0x70, 0x80, 0xa0, 0xe0, 0x100, 0x120, 0x130, 0x150, 0x180, 0x1c0, 0x1f0, 0x230, 0x260, 0x2a0, 0x2e0, 0x320,
         0x360, 0x3b0, 0x3f0, 0x440, 0x480, 0x4c0, 0x510, 0x570, 0x5b0, 0x610])
    # plt.scatter(Xi, Yi, color="red", label="Sample Point", linewidth=1)
    plt.scatter(xdata3+xdata4, Yi, color="red", label="Sample Point", linewidth=1)
    plt.show()


def test_curvefit():
    # xdata = np.array([0, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 255])  # x轴
    # ydata = np.array([0, 6, 14, 29, 61, 90, 115, 138, 161, 181, 199, 214, 226, 242, 254, 260])  # 清新亮丽

    # xdata = np.array(
    #     [8.3, 10.8, 13.1, 17.5, 27.5, 37.9, 46.5, 52.6, 56.2, 57.9, 59.9, 61.1, 62.8, 64.2, 65.5, 66.9, 68, 68.9, 70,
    #      70.7, 71.4, 71.5, 70.7, 67.7, 66.7, 65])
    xdata = np.array(
        [6.6, 9.2, 11.3, 14.4, 23.6, 32.1, 40.8, 46.4, 49.8, 52.4, 55.2, 57.6, 60, 62, 63.6, 65.1, 66.3, 66.5, 65.3,
         63.2, 62.1, 61.5, 61.3, 60.8, 60.5, 60.1])
    xdata2 = np.array(
        [5.8, 7.1, 10.5, 11.8, 12.8, 13.5, 13.5, 14.3, 14.8, 15.5, 16.3, 16.5, 16.7, 16.7, 16.9, 16.9, 16.9, 17.1, 18.8,
         20.5, 21.3, 21.9, 22.1, 22, 23, 23.3])
    xdata3 = np.array(
        [36.4, 52.6, 50.3, 31.7, 31.3, 27.6, 31.8, 30.7, 26.9, 21, 19, 15.2, 14.9, 12.9, 10.9, 10.1, 9.4, 8.5, 8.8, 9.3,
         10.2, 10.9, 11.1, 11.5, 11.4, 11.4])
    xdata4 = np.array(
        [51, 31.1, 28, 42.1, 32.2, 26.8, 13.9, 8.6, 8.4, 11, 9.5, 10.6, 8.4, 8.4, 8.5, 7.9, 7.3, 7.8, 7.1, 7, 6.3, 5.6,
         5.4, 5.7, 5, 5.2])
    ydata = np.array(
        [161, 140.4, 133, 138.1, 118.4, 103, 85.8, 75.6, 70, 66.8, 62.4, 60, 56.6, 54.6, 53.1, 51.5, 50.5, 50.3, 49.6,
         49.5, 49.1, 48.6, 48.4, 49.2, 48.6, 49.2])
    xdata = xdata3 + xdata4;
    # ydata = np.array(
    #     [0x70, 0x80, 0xa0, 0xe0, 0x100, 0x120, 0x130, 0x150, 0x180, 0x1c0, 0x1f0, 0x230, 0x260, 0x2a0, 0x2e0, 0x320,
    #      0x360, 0x3b0, 0x3f0, 0x440, 0x480, 0x4c0, 0x510, 0x570, 0x5b0, 0x610])

    p0 = [0.1, -0.01, 100]  # 拟合的初始参数设置，可以任意设置 最小二乘法使用

    # 下面是曲线拟合的三种方式 polyfit, curve_fit 和 leastsq 实际选用一种即可

    para = np.polyfit(xdata3+xdata4, ydata, 2)  # 该函数主要做多项式拟合，如果是指数，对数 推荐使用后面两种方式
    y_fitted = para[0] * (xdata ** 2) + para[1] * xdata + para[2]
    # y_fitted = para[0] * xdata + para[1]

    # para, pcov = curve_fit(Fun_curve_fit, xdata, ydata)
    # y_fitted = Fun_curve_fit(xdata, para[0], para[1], para[2])  # 画出拟合后的曲线

    # para = leastsq(error, p0, args=(xdata, ydata))  # 最小二乘法进行拟合
    # y_fitted = Fun_leastsq(para[0], xdata)  # 画出拟合后的曲线

    plt.figure
    plt.plot(xdata, ydata, 'r', label='Original curve')
    plt.plot(xdata, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()
    print(para)


if __name__ == '__main__':
    test_curvefit()
    # test_draw_samplepoint()
