import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


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
    test_show_bf3a03_gamma()
