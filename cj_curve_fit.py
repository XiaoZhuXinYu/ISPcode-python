import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys


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
        [0.36, 0.66, 0.25, 0.70, 0.19, 0.54, 0.42, 0.35, 0.50, 0.19, 0.28, 0.49, 0.43, 0.50, 0.23, 0.29, 0.38, 0.45,
         0.33, 0.22, 0.47, 0.28, 0.24, 0.48, 0.37, 0.32, 0.26, 0.47, 0.28])
    Yi = np.array(
        [0.59, 0.80, 0.45, 0.70, 0.35, 0.89, 0.71, 0.61, 0.84, 0.39, 0.56, 0.84, 0.72, 0.84, 0.53, 0.62, 0.82, 0.74,
         0.65, 0.22, 0.77, 0.45, 0.24, 0.85, 0.37, 0.32, 0.26, 0.97, 0.28])

    xdata5 = np.array(
        [0.36, 0.25, 0.19, 0.54, 0.42, 0.35, 0.50, 0.19, 0.28, 0.49, 0.43, 0.50, 0.23, 0.29, 0.38, 0.45,
         0.33, 0.47, 0.48, 0.47])
    Yi5 = np.array(
        [0.59, 0.45, 0.35, 0.89, 0.71, 0.61, 0.84, 0.39, 0.56, 0.84, 0.72, 0.84, 0.53, 0.62, 0.82, 0.74,
         0.65, 0.77, 0.85, 0.97])

    xdata6 = np.array(
        [0.33, 0.57, 0.56, 0.23, 0.51, 0.77, 0.24, 0.78, 0.66, 0.20, 0.43, 0.22, 0.76, 0.20, 0.34])
    Yi6 = np.array(
        [0.42, 0.67, 0.62, 0.36, 0.53, 0.77, 0.28, 0.78, 0.66, 0.30, 0.43, 0.25, 0.76, 0.26, 0.42])

    xdata7 = np.array(
        [0.36, 0.66, 0.25, 0.70, 0.19, 0.54, 0.42, 0.35, 0.50, 0.19, 0.28, 0.49, 0.43, 0.50, 0.23, 0.29, 0.38, 0.45,
         0.33, 0.22, 0.47, 0.28, 0.24, 0.48, 0.37, 0.32, 0.26, 0.47, 0.28, 0.33, 0.57, 0.56, 0.23, 0.51, 0.77, 0.24,
         0.78, 0.66, 0.20, 0.43, 0.22, 0.76, 0.20, 0.34])
    Yi7 = np.array(
        [0.59, 0.80, 0.45, 0.70, 0.35, 0.89, 0.71, 0.61, 0.84, 0.39, 0.56, 0.84, 0.72, 0.84, 0.53, 0.62, 0.82, 0.74,
         0.65, 0.22, 0.77, 0.45, 0.24, 0.85, 0.37, 0.32, 0.26, 0.97, 0.28, 0.42, 0.67, 0.62, 0.36, 0.53, 0.77, 0.28,
         0.78, 0.66, 0.30, 0.43, 0.25, 0.76, 0.26, 0.42])

    Zi = np.array(
        [0x70, 0x80, 0xa0, 0xe0, 0x100, 0x120, 0x130, 0x150, 0x180, 0x1c0, 0x1f0, 0x230, 0x260, 0x2a0, 0x2e0, 0x320,
         0x360, 0x3b0, 0x3f0, 0x440, 0x480, 0x4c0, 0x510, 0x570, 0x5b0, 0x610])
    # plt.scatter(Xi, Yi, color="red", label="Sample Point", linewidth=1)
    plt.scatter(Yi7, xdata7, color="red", label="Sample Point", linewidth=1)
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

    xdata5 = np.array(
        [0.36, 0.25, 0.19, 0.54, 0.42, 0.35, 0.50, 0.19, 0.28, 0.49, 0.43, 0.50, 0.23, 0.29, 0.38, 0.45,
         0.33, 0.47, 0.48, 0.47])
    Yi5 = np.array(
        [0.59, 0.45, 0.35, 0.89, 0.71, 0.61, 0.84, 0.39, 0.56, 0.84, 0.72, 0.84, 0.53, 0.62, 0.82, 0.74,
         0.65, 0.77, 0.85, 0.97])

    xdata6 = np.array(
        [0.33, 0.57, 0.56, 0.23, 0.51, 0.77, 0.24, 0.78, 0.66, 0.20, 0.43, 0.22, 0.76, 0.20, 0.34])
    Yi6 = np.array(
        [0.42, 0.67, 0.62, 0.36, 0.53, 0.77, 0.28, 0.78, 0.66, 0.30, 0.43, 0.25, 0.76, 0.26, 0.42])

    xdata7 = np.array(
        [0.36, 0.25, 0.19, 0.54, 0.42, 0.35, 0.50, 0.19, 0.28, 0.49, 0.43, 0.50, 0.23, 0.29, 0.38, 0.45,
         0.33, 0.47, 0.48, 0.47, 0.33, 0.57, 0.56, 0.23, 0.51, 0.77, 0.24, 0.78, 0.66, 0.20, 0.43, 0.22, 0.76, 0.20,
         0.34])
    Yi7 = np.array(
        [0.59, 0.45, 0.35, 0.89, 0.71, 0.61, 0.84, 0.39, 0.56, 0.84, 0.72, 0.84, 0.53, 0.62, 0.82, 0.74,
         0.65, 0.77, 0.85, 0.97, 0.42, 0.67, 0.62, 0.36, 0.53, 0.77, 0.28, 0.78, 0.66, 0.30, 0.43, 0.25, 0.76, 0.26,
         0.42])
    # 下面是曲线拟合的三种方式 polyfit, curve_fit 和 leastsq 实际选用一种即可

    para = np.polyfit(Yi5, xdata5, 1)  # 该函数主要做多项式拟合，如果是指数，对数 推荐使用后面两种方式
    # y_fitted = para[0] * (xdata ** 2) + para[1] * xdata + para[2]
    y_fitted = para[0] * Yi5 + para[1]

    # para, pcov = curve_fit(Fun_curve_fit, xdata, ydata)
    # y_fitted = Fun_curve_fit(xdata, para[0], para[1], para[2])  # 画出拟合后的曲线

    # p0 = [0.1, -0.01, 100]  # 拟合的初始参数设置，可以任意设置 最小二乘法使用
    # para = leastsq(error, p0, args=(xdata, ydata))  # 最小二乘法进行拟合
    # y_fitted = Fun_leastsq(para[0], xdata)  # 画出拟合后的曲线

    plt.figure()
    # plt.plot(xdata, ydata, 'r', label='Original curve')
    plt.scatter(Yi5, xdata5, color="red", label="Sample Point", linewidth=1)
    plt.plot(Yi5, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()
    print(para)


# 下面都是gamma拟合函数
def gamma_cr(gamma):
    x = np.arange(0, 256)
    x = x.astype(float)
    y = x ** (1 / gamma)
    print(np.max(y))
    plt.plot(x, y, 'r', label='gamma curve')
    plt.legend()
    plt.show()


def logarithmic_cr(q=1, k=1):  # 课程里的三种组合 (1, 1) (10, 50) (15, 100)
    x = np.arange(1, 256)
    x = x.astype(np.float)
    q = 1 if q < 1 else q
    k = 1 if k < 1 else k
    L_max = np.max(x)
    L_d = np.log10(1 + x * q) / np.log10(1 + L_max * k)
    y = x * L_d / x
    plt.plot(x, y, 'r', label='logarithmic curve')
    plt.legend()
    plt.show()


def sigmod_cr_curve(b, c):  # 课程里的三种组合 (0.7, 4) (0.7, 8) (4, 100)
    x1 = np.arange(0, 256)
    # x1 = x1.astype(int)  # 如果是int类型下面4个值，为什么会打印下面负数
    x1 = x1.astype(float)
    y1 = ((x1 ** b) / ((c ** b) + (x1 ** b)))
    index = np.where(y1 < 0)
    print(np.max(y1))
    print(index,
          y1[index])  # (array([213, 214, 215, 255], dtype=int64),) [-0.96336507 -0.99979975 -1.03815631 -2.00450715]
    plt.plot(x1, y1, 'r', label='r sigmod')
    plt.legend()
    plt.show()


# 指数
def exponential_cr(q=1, k=1):
    L = np.arange(0, 256)
    L = L.astype(np.float)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L_average = np.exp(np.average(np.log(L + sys.float_info.epsilon)))
    L_max = np.max(L)
    L_d0 = 1 - np.exp(-(L * q) / (L_average * k))
    L_d1 = (L / L_max) ** (q / k)

    plt.plot(L, L_d0, 'r', label='r exponential')
    plt.plot(L, L_d1, 'g', label='r exponential1')
    plt.legend()
    plt.show()


def tonemapping_operator_aces():
    RGB = np.arange(0, 256)
    RGB = RGB.astype(np.float)

    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    Y = (RGB * (A * RGB + B)) / (RGB * (C * RGB + D) + E)

    plt.plot(RGB, Y, 'r', label='r sigmod')
    plt.legend()
    plt.show()


def tonemapping_operator_filmic(shoulder_strength=0.22, linear_strength=0.3, linear_angle=0.1, toe_strength=0.2,
                                toe_numerator=0.01, toe_denominator=0.3, exposure_bias=2, linear_whitepoint=11.2):
    RGB = np.arange(0, 256)
    RGB = RGB.astype(np.float)

    A = shoulder_strength
    B = linear_strength
    C = linear_angle
    D = toe_strength
    E = toe_numerator
    F = toe_denominator

    def f(x, A, B, C, D, E, F):
        return ((
                        (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F)

    RGB = f(RGB * exposure_bias, A, B, C, D, E, F)
    Y = RGB * (1 / f(linear_whitepoint, A, B, C, D, E, F))
    print(f(linear_whitepoint, A, B, C, D, E, F))
    plt.plot(RGB, Y, 'r', label='r sigmod')
    plt.legend()
    plt.show()


def LTM1(new_scale=30):
    Y = np.arange(1, 256)
    d = 120  # kernel size
    sigmaColor = 0.8  # color domain sigma
    sigmaSpace = 100  # ((width**2+height**2)**0.5)*0.02  # space domain sigma
    y = Y.astype(np.float)
    log_y = np.log10(y)
    log_base = cv.bilateralFilter(log_y.astype(np.float32), d, sigmaColor, sigmaSpace)
    detail_log_lum = log_y - log_base

    new_base = np.log10(new_scale)
    min_log_base = np.maximum(np.min(log_base), 1)  # 为了放缩比准确
    new_base2 = (np.max(log_base) - min_log_base)
    base_scale = new_base / new_base2

    large_scale2_reduced = log_base * base_scale
    log_absolute_scale = np.max(log_base) * base_scale
    out_log_lum = detail_log_lum + large_scale2_reduced - log_absolute_scale
    out_log_lum2 = 10 ** out_log_lum

    plt.plot(Y, out_log_lum2, 'r', label='r ltm')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_curvefit()
    # test_draw_samplepoint()
    # gamma_cr(gamma=3.4)
    # logarithmic_cr(q=15, k=150)
    # sigmod_cr_curve(4, 143)
    # tonemapping_operator_aces()
    # tonemapping_operator_filmic()
    # exponential_cr(q=5, k=10)
    # LTM1(new_scale=30)
