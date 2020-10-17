import numpy as np
import cv2 as cv
from scipy import interpolate
from matplotlib import pyplot as plt
import sys
import cj_rgbimage as rgbimage
import os


def degamma_srgb(self, clip_range=[0, 1023]):
    # bring data in range 0 to 1
    data = np.clip(self.data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.asarray(data)
    mask = data > 0.04045

    # basically, if data[x, y, c] > 0.04045, data[x, y, c] = ( (data[x, y, c] + 0.055) / 1.055 ) ^ 2.4
    #            else, data[x, y, c] = data[x, y, c] / 12.92
    data[mask] += 0.055
    data[mask] /= 1.055
    data[mask] **= 2.4

    data[np.invert(mask)] /= 12.92

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def gamma_srgb(data, clip_range=[0, 1023]):
    # bring data in range 0 to 1
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.asarray(data)
    mask = data > 0.0031308

    # basically, if data[x, y, c] > 0.0031308, data[x, y, c] = 1.055 * ( var_R(i, j) ^ ( 1 / 2.4 ) ) - 0.055
    #            else, data[x, y, c] = data[x, y, c] * 12.92
    data[mask] **= 0.4167
    data[mask] *= 1.055
    data[mask] -= 0.055

    data[np.invert(mask)] *= 12.92

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def RGB_luminance(RGB):
    Y = 0.299 * RGB[:, :, 0] + 0.5877 * RGB[:, :, 1] + 0.114 * RGB[:, :, 2]
    return Y


# sys.float_info.epsilon是机器可以区分出的两个浮点数的最小区别
# a 输入的是图像的亮度值
def log_average(a, epsilon=sys.float_info.epsilon):
    a = a.astype(np.float)
    average = np.exp(np.average(np.log(a + epsilon)))
    print("average = ", average)
    return average


def tonemapping_operator_Schlick1994(RGB, p=1):
    RGB = RGB.astype(np.float)

    L = RGB_luminance(RGB)
    L_max = np.max(L)
    L_d = (p * L) / (p * L - L + L_max)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_Tumblin1999(RGB, L_da=20, C_max=100, L_max=100):
    RGB = RGB.astype(np.float)

    L_w = RGB_luminance(RGB)

    def f(x):
        return np.where(x > 100, 2.655,
                        1.855 + 0.4 * np.log10(x + 2.3 * 10 ** -5))

    L_wa = np.exp(np.mean(np.log(L_w + 2.3 * 10 ** -5)))
    g_d = f(L_da)
    g_w = f(L_wa)
    g_wd = g_w / (1.855 + 0.4 * np.log(L_da))

    mL_wa = np.sqrt(C_max) ** (g_wd - 1)

    L_d = mL_wa * L_da * (L_w / L_wa) ** (g_w / g_d)

    RGB = RGB * L_d[..., np.newaxis] / L_w[..., np.newaxis]
    RGB = RGB / L_max

    return RGB


def tonemapping_operator_Reinhard2004(RGB, f=0, m=0.3, a=0, c=0):
    RGB = RGB.astype(np.float)

    C_av = np.array((np.average(RGB[..., 0]), np.average(RGB[..., 1]),
                     np.average(RGB[..., 2])))

    L = RGB_luminance(RGB)

    L_lav = log_average(L)
    L_min, L_max = np.min(L), np.max(L)

    f = np.exp(-f)

    m = (m if m > 0 else (0.3 + 0.7 * (
            (np.log(L_max) - L_lav) / (np.log(L_max) - np.log(L_min)) ** 1.4)))

    I_l = (c * RGB + (1 - c)) * L[..., np.newaxis]
    I_g = c * C_av + (1 - c) * L_lav
    I_a = a * I_l + (1 - a) * I_g

    RGB = RGB / (RGB + (f * I_a) ** m)

    return RGB


# 下面都是曲线
def gamma_cr(x, gamma):
    x = x.astype(np.float)
    shape1 = x.shape
    x = x.flatten()
    y = x ** (1 / gamma)
    y = y * (255.0 / np.max(y))
    y = y.astype(np.uint8)
    y.shape = shape1
    return y


def logarithmic_cr(x, q=1, k=1):
    x = x.astype(np.float)
    q = 1 if q < 1 else q
    k = 1 if k < 1 else k
    L_max = np.max(x)
    # L_max = 255  # 可以和上一种求 L_max 的方法进行比较
    L_d = np.log10(1 + x * q) / np.log10(1 + L_max * k)
    y = x * L_d / x
    return y


def exponential_cr(RGB, q=1, k=1):
    RGB = RGB.astype(np.float)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = RGB_luminance(RGB)
    L_a = log_average(L)
    L_d = 1 - np.exp(-(L * q) / (L_a * k))

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


# x表示输入图像的亮度
def sigmod_cr(x, b, c):
    x = x.astype(np.float)
    shape1 = x.shape
    x = x.flatten()
    # x1 = x1.astype(int)  # 如果是int类型会出现负值，具体见 sigmod_cr_curve(b, c)
    y = ((x ** b) / ((c ** b) + (x ** b)))
    y = y * 255
    y = y.astype(np.uint8)
    y.shape = shape1
    return y


def LTM(RGB, maxvalue=2 ** 14, new_scale=30):
    d = 120  # kernel size
    sigmaColor = 0.8  # color domain sigma
    sigmaSpace = 100  # ((width**2+height**2)**0.5)*0.02  # space domain sigma
    y = 20 * RGB[:, :, 0] / 61 + 40 * RGB[:, :, 1] / 61 + 1 * RGB[:, :, 2] / 61

    chroma = np.empty_like(RGB)
    chroma[:, :, 0] = RGB[:, :, 0] / y
    chroma[:, :, 1] = RGB[:, :, 1] / y
    chroma[:, :, 2] = RGB[:, :, 2] / y
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

    chroma[:, :, 0] = chroma[:, :, 0] * out_log_lum2
    chroma[:, :, 1] = chroma[:, :, 1] * out_log_lum2
    chroma[:, :, 2] = chroma[:, :, 2] * out_log_lum2
    chroma = chroma - np.min(chroma)
    chroma = np.clip(chroma, 0, 1)
    # 不做的话,色彩会偏色.
    chroma = chroma ** (1 / 2.2)

    return chroma


def LTM1(Y, maxvalue=2 ** 14, d=17, sigmaColor=0.8, sigmaSpace=16, new_scale=30):
    y = Y.astype(np.float)
    y = y / 255 * (maxvalue - 1) + 1
    log_y = np.log10(y)
    log_base = cv.bilateralFilter(log_y.astype(np.float32), d, sigmaColor, sigmaSpace)
    log_detail = log_y - log_base

    new_base = np.log10(new_scale)
    min_log_base = np.maximum(np.min(log_base), 1)  # 为了放缩比准确
    new_base2 = (np.max(log_base) - min_log_base)
    compressionfactor = new_base / new_base2

    large_scale2_reduced = log_base * compressionfactor
    log_absolute_scale = np.max(log_base) * compressionfactor
    out_log_lum = log_detail + large_scale2_reduced - log_absolute_scale
    out_log_lum2 = 10 ** out_log_lum
    out_log_lum2 = out_log_lum2 ** (1 / 2.2)
    out_log_lum2 = out_log_lum2 * 255
    out_log_lum2 = out_log_lum2.astype(np.uint8)
    return out_log_lum2


def test_show_bf3a03_gamma():
    x = np.array([0, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 255])
    xnew = np.linspace(0, 255, 2551)
    x______ = np.array([0, 4, 4, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32])  # x轴
    y_moren = np.array([0, 9, 17, 37, 74, 103, 126, 145, 163, 179, 192, 203, 214, 232, 246, 258])  # 默认
    y__qxll = np.array([0, 6, 14, 29, 61, 90, 115, 138, 161, 181, 199, 214, 226, 242, 254, 260])  # 清新亮丽
    y__test = np.array([0, 22, 40, 68, 100, 120, 125, 130, 136, 142, 147, 151, 156, 172, 212, 255])  # 测试
    y__test1 = np.array([0, 1, 2, 5, 12, 22, 34, 52, 76, 104, 140, 180, 206, 232, 246, 257])  # 测试
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
    plt.plot(xnew, ynew, label="qxll")

    f = interpolate.interp1d(x, y__test, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="test1")

    f = interpolate.interp1d(x, y__test1, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="test2")

    f = interpolate.interp1d(x, y_dz, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="dz")

    f = interpolate.interp1d(x, y_gbgdh, kind="quadratic")
    ynew = f(xnew)
    plt.plot(xnew, ynew, label="gbgdh")

    plt.title("gamma")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    print('This is main of module')
    # test_show_bf3a03_gamma()

    x = np.arange(256)
    image_gray = cv.imread("../pic/pic_0.bmp", cv.IMREAD_GRAYSCALE)
    # gtmfile = sigmod_cr(image_gray, 4, 100)
    # ltmfile = LTM1(image_gray, maxvalue=2 ** 14, d=7, sigmaColor=0.8, sigmaSpace=10, new_scale=7)
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image_gray)

    # equ = cv.equalizeHist(image_gray)
    # gray = np.full((480, 640), 128)
    # print(gray)
    cv.imwrite("../pic/pic_clahe.bmp", cl1)
    # rgbimage.rgb_image_show_color(equ, maxvalue=255, color="gray", compress_ratio=1)

    # for root, dirs, files in os.walk("../pic/525/ok2"):
    #
    #     # root 表示当前正在访问的文件夹路径
    #     # dirs 表示该文件夹下的子目录名list
    #     # files 表示该文件夹下的文件list
    #
    #     # 遍历文件
    #     for f in files:
    #         filename = os.path.join(root, f)
    #
    #         filename1 = "../pic/median/ok2" + filename[14:]  # 14不是固定数字，和系统路径的长度有关
    #         print(filename, filename1)
    #         # filename1 = "../pic/gamma-gamma3.4/" + filename[13:]
    #         image_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)  # 测试时文件夹中不要有txt文件
    #
    #         tmp_gray = np.zeros(image_gray.shape)
    #         # tmp_gray[50:430, 50:590] = image_gray[50:430, 50:590]
    #         tmp_gray = image_gray
    #         # gtmfile = sigmod_cr(tmp_gray, 4, 143)
    #         # image_gray[50:430, 50:590] = gtmfile[50:430, 50:590]
    #         # image_gray = gtmfile
    #
    #         # gammafile = gamma_cr(image_gray, gamma=3.4)
    #         # equ = cv.equalizeHist(image_gray)
    #         # clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    #         # cl1 = clahe.apply(image_gray)
    #
    #         cv.medianBlur(image_gray, 3, tmp_gray)  # 中值滤波
    #         cv.imwrite(filename1, tmp_gray)
