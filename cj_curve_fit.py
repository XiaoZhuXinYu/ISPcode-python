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


def main():
    xdata = np.array([0, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 255])  # x轴
    ydata = np.array([0, 6, 14, 29, 61, 90, 115, 138, 161, 181, 199, 214, 226, 242, 254, 260])  # 清新亮丽
    p0 = [0.1, -0.01, 100]  # 拟合的初始参数设置，可以任意设置 最小二乘法使用

    # 下面是曲线拟合的三种方式 polyfit, curve_fit 和 leastsq 实际选用一种即可

    para = np.polyfit(xdata, ydata, 2)  # 该函数主要做多项式拟合，如果是指数，对数 推荐使用后面两种方式
    y_fitted = para[0] * (xdata ** 2) + para[1] * xdata + para[2]

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
    main()
