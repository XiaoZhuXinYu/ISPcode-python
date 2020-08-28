import numpy as np
import cj_gamma as gamma


def ycbcr_compress(image):
    image[:, :, 0] = image[:, :, 0] * 219 / 255 + 16
    image[:, :, 0] = np.clip(image[:, :, 0], 16, 235)
    image[:, :, 1] = image[:, :, 1] * 224 / 255 + 16
    image[:, :, 1] = np.clip(image[:, :, 1], 16, 240)
    image[:, :, 2] = image[:, :, 2] * 224 / 255 + 16
    image[:, :, 2] = np.clip(image[:, :, 2], 16, 240)
    return image


def ycbcr_decompress(image):
    image[:, :, 0] = (image[:, :, 0] - 16) * 255 / 219
    image[:, :, 1] = (image[:, :, 1] - 16) * 255 / 224
    image[:, :, 2] = (image[:, :, 2] - 16) * 255 / 224
    image = np.clip(image, 0, 255)
    return image


# 0~255 的 Ycbcr转换
def ycbcr2rgb(image, width, height):
    rgb_img = np.zeros(shape=(height, width, 3))
    rgb_img[:, :, 0] = image[:, :, 0] + 1.402 * (image[:, :, 2] - 128)  # R= Y+1.402*(Cr-128)
    rgb_img[:, :, 1] = image[:, :, 0] - 0.344136 * (image[:, :, 1] - 128) - 0.714136 * (
            image[:, :, 2] - 128)  # G=Y-0.344136*(Cb-128)-0.714136*(Cr-128)
    rgb_img[:, :, 2] = image[:, :, 0] + 1.772 * (image[:, :, 1] - 128)  # B=Y+1.772*(Cb-128)
    rgb_img = np.clip(rgb_img, 0, 255)
    return rgb_img


# 0~255 的 Ycbcr转换
def rgb2ycbcr(image, width, height):
    ycbcr_img = np.zeros(shape=(height, width, 3))
    ycbcr_img[:, :, 0] = 0.299 * image[:, :, 0] + 0.5877 * image[:, :, 1] + 0.114 * image[:, :, 2]
    ycbcr_img[:, :, 1] = 128 - 0.168736 * image[:, :, 0] - 0.331264 * image[:, :, 1] + 0.5 * image[:, :, 2]
    ycbcr_img[:, :, 2] = 128 + 0.5 * image[:, :, 0] - 0.418688 * image[:, :, 1] - 0.081312 * image[:, :, 2]
    ycbcr_img = np.clip(ycbcr_img, 0, 255)
    return ycbcr_img


def labf(t):
    # 本算法中传入的 t 是一个二维矩阵
    d = t ** (1 / 3)
    index = np.where(t <= 0.008856)  # 0.008856 约等于 (6/29) 的三次方
    d[index] = 7.787 * t[index] + 4 / 29  # 7.787 约等于 (1/3) * (29/6) * (29/6)
    return d


def rgb2lab(X):
    a = np.array([
        [3.40479, -1.537150, -0.498535],
        [-0.969256, 1.875992, 0.041556],
        [0.055648, -0.204043, 1.057311]])
    # ai = np.array([
    #     [0.38627512, 0.33488427, 0.1689713],
    #     [0.19917304, 0.70345694, 0.06626421],
    #     [0.01810671, 0.11812969, 0.94969014]])  # 该矩阵和下面算出来的是一样的
    ai = np.linalg.inv(a)  # 求矩阵a的逆矩阵

    h, w, c = X.shape  # X是含有RGB三个分量的数据
    R = X[:, :, 0]
    G = X[:, :, 1]
    B = X[:, :, 2]
    planed_R = R.flatten()  # 将二维矩阵转成1维矩阵
    planed_G = G.flatten()
    planed_B = B.flatten()
    planed_image = np.zeros((c, h * w))  # 注意这里 planed_B 是一个二维数组
    planed_image[0, :] = planed_R  # 将 planed_R 赋值给 planed_image 的第0行
    planed_image[1, :] = planed_G
    planed_image[2, :] = planed_B
    planed_lab = np.dot(ai, planed_image)  # 相当于两个矩阵相乘 将rgb空间转到xyz空间
    planed_1 = planed_lab[0, :]
    planed_2 = planed_lab[1, :]
    planed_3 = planed_lab[2, :]
    L1 = np.reshape(planed_1, (h, w))
    L2 = np.reshape(planed_2, (h, w))
    L3 = np.reshape(planed_3, (h, w))
    result_lab = np.zeros((h, w, c))
    # color  space conversion  into LAB
    result_lab[:, :, 0] = 116 * labf(L2 / 255) - 16
    result_lab[:, :, 1] = 500 * (labf(L1 / 255) - labf(L2 / 255))
    result_lab[:, :, 2] = 200 * (labf(L2 / 255) - labf(L3 / 255))

    return result_lab


def rgb2xyz(data, color_space="srgb", clip_range=[0, 65535]):
    # input rgb in range clip_range
    # output xyz is in range 0 to 1

    if color_space == "srgb":
        # degamma / linearization
        data = gamma.degamma_srgb(data, clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])

        # matrix multiplication`
        output = np.empty(np.shape(data), dtype=np.float32)
        output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
        output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
        output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

    elif color_space == "adobe-rgb-1998":
        # degamma / linearization
        data = gamma.degamma_adobe_rgb_1998(data, clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])

        # matrix multiplication
        output = np.empty(np.shape(data), dtype=np.float32)
        output[:, :, 0] = data[:, :, 0] * 0.5767309 + data[:, :, 1] * 0.1855540 + data[:, :, 2] * 0.1881852
        output[:, :, 1] = data[:, :, 0] * 0.2973769 + data[:, :, 1] * 0.6273491 + data[:, :, 2] * 0.0752741
        output[:, :, 2] = data[:, :, 0] * 0.0270343 + data[:, :, 1] * 0.0706872 + data[:, :, 2] * 0.9911085

    elif color_space == "linear":

        # matrix multiplication`
        output = np.empty(np.shape(data), dtype=np.float32)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])
        output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
        output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
        output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

    else:
        print("Warning! color_space must be srgb or adobe-rgb-1998.")
        return

    return output


def xyz2rgb(data, color_space="srgb", clip_range=[0, 65535]):
    # input xyz is in range 0 to 1
    # output rgb in clip_range

    # allocate space for output
    output = np.empty(np.shape(data), dtype=np.float32)

    if color_space == "srgb":
        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 3.2406 + data[:, :, 1] * -1.5372 + data[:, :, 2] * -0.4986
        output[:, :, 1] = data[:, :, 0] * -0.9689 + data[:, :, 1] * 1.8758 + data[:, :, 2] * 0.0415
        output[:, :, 2] = data[:, :, 0] * 0.0557 + data[:, :, 1] * -0.2040 + data[:, :, 2] * 1.0570

        # gamma to retain nonlinearity
        output = gamma.gamma_srgb(data, clip_range)

    elif color_space == "adobe-rgb-1998":
        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 2.0413690 + data[:, :, 1] * -0.5649464 + data[:, :, 2] * -0.3446944
        output[:, :, 1] = data[:, :, 0] * -0.9692660 + data[:, :, 1] * 1.8760108 + data[:, :, 2] * 0.0415560
        output[:, :, 2] = data[:, :, 0] * 0.0134474 + data[:, :, 1] * -0.1183897 + data[:, :, 2] * 1.0154096

        # gamma to retain nonlinearity
        output = gamma.gamma_adobe_rgb_1998(data, clip_range)

    elif color_space == "linear":
        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 3.2406 + data[:, :, 1] * -1.5372 + data[:, :, 2] * -0.4986
        output[:, :, 1] = data[:, :, 0] * -0.9689 + data[:, :, 1] * 1.8758 + data[:, :, 2] * 0.0415
        output[:, :, 2] = data[:, :, 0] * 0.0557 + data[:, :, 1] * -0.2040 + data[:, :, 2] * 1.0570

        # gamma to retain nonlinearity
        output = output * clip_range[1]

    else:
        print("Warning! color_space must be srgb or adobe-rgb-1998.")
        return

    return output


def get_xyz_reference(cie_version="1931", illuminant="d65"):
    if cie_version == "1931":
        xyz_reference_dictionary = {"A": [109.850, 100.0, 35.585], "B": [99.0927, 100.0, 85.313],
                                    "C": [98.074, 100.0, 118.232], "d50": [96.422, 100.0, 82.521],
                                    "d55": [95.682, 100.0, 92.149], "d65": [95.047, 100.0, 108.883],
                                    "d75": [94.972, 100.0, 122.638], "E": [100.0, 100.0, 100.0],
                                    "F1": [92.834, 100.0, 103.665], "F2": [99.187, 100.0, 67.395],
                                    "F3": [103.754, 100.0, 49.861], "F4": [109.147, 100.0, 38.813],
                                    "F5": [90.872, 100.0, 98.723], "F6": [97.309, 100.0, 60.191],
                                    "F7": [95.044, 100.0, 108.755], "F8": [96.413, 100.0, 82.333],
                                    "F9": [100.365, 100.0, 67.868], "F10": [96.174, 100.0, 81.712],
                                    "F11": [100.966, 100.0, 64.370], "F12": [108.046, 100.0, 39.228]}

    elif cie_version == "1964":
        xyz_reference_dictionary = {"A": [111.144, 100.0, 35.200], "B": [99.178, 100.0, 84.3493],
                                    "C": [97.285, 100.0, 116.145], "D50": [96.720, 100.0, 81.427],
                                    "D55": [95.799, 100.0, 90.926], "D65": [94.811, 100.0, 107.304],
                                    "D75": [94.416, 100.0, 120.641], "E": [100.0, 100.0, 100.0],
                                    "F1": [94.791, 100.0, 103.191], "F2": [103.280, 100.0, 69.026],
                                    "F3": [108.968, 100.0, 51.965], "F4": [114.961, 100.0, 40.963],
                                    "F5": [93.369, 100.0, 98.636], "F6": [102.148, 100.0, 62.074],
                                    "F7": [95.792, 100.0, 107.687], "F8": [97.115, 100.0, 81.135],
                                    "F9": [102.116, 100.0, 67.826], "F10": [99.001, 100.0, 83.134],
                                    "F11": [103.866, 100.0, 65.627], "F12": [111.428, 100.0, 40.353]}

    else:
        print("Warning! cie_version must be 1931 or 1964.")
        return

    return np.divide(xyz_reference_dictionary[illuminant], 100.0)


def xyz2lab(self, cie_version="1931", illuminant="d65"):
    xyz_reference = get_xyz_reference(cie_version, illuminant)

    data = self.data
    data[:, :, 0] = data[:, :, 0] / xyz_reference[0]
    data[:, :, 1] = data[:, :, 1] / xyz_reference[1]
    data[:, :, 2] = data[:, :, 2] / xyz_reference[2]

    data = np.asarray(data)

    # if data[x, y, c] > 0.008856, data[x, y, c] = data[x, y, c] ^ (1/3)
    # else, data[x, y, c] = 7.787 * data[x, y, c] + 16/116
    mask = data > 0.008856
    data[mask] **= 1. / 3.
    data[np.invert(mask)] *= 7.787
    data[np.invert(mask)] += 16. / 116.

    data = np.float32(data)
    output = np.empty(np.shape(self.data), dtype=np.float32)
    output[:, :, 0] = 116. * data[:, :, 1] - 16.
    output[:, :, 1] = 500. * (data[:, :, 0] - data[:, :, 1])
    output[:, :, 2] = 200. * (data[:, :, 1] - data[:, :, 2])

    return output


def lab2xyz(self, cie_version="1931", illuminant="d65"):
    output = np.empty(np.shape(self.data), dtype=np.float32)

    output[:, :, 1] = (self.data[:, :, 0] + 16.) / 116.
    output[:, :, 0] = (self.data[:, :, 1] / 500.) + output[:, :, 1]
    output[:, :, 2] = output[:, :, 1] - (self.data[:, :, 2] / 200.)

    # if output[x, y, c] > 0.008856, output[x, y, c] ^ 3
    # else, output[x, y, c] = ( output[x, y, c] - 16/116 ) / 7.787
    output = np.asarray(output)
    mask = output > 0.008856
    output[mask] **= 3.
    output[np.invert(mask)] -= 16 / 116
    output[np.invert(mask)] /= 7.787

    xyz_reference = get_xyz_reference(cie_version, illuminant)

    output = np.float32(output)
    output[:, :, 0] = output[:, :, 0] * xyz_reference[0]
    output[:, :, 1] = output[:, :, 1] * xyz_reference[1]
    output[:, :, 2] = output[:, :, 2] * xyz_reference[2]

    return output


def lab2lch(data):
    output = np.empty(np.shape(data), dtype=np.float32)

    output[:, :, 0] = data[:, :, 0]  # L transfers directly
    output[:, :, 1] = np.power(np.power(data[:, :, 1], 2) + np.power(data[:, :, 2], 2), 0.5)
    output[:, :, 2] = np.arctan2(data[:, :, 2], data[:, :, 1]) * 180 / np.pi

    return output


def lch2lab(data):
    output = np.empty(np.shape(data), dtype=np.float32)

    output[:, :, 0] = data[:, :, 0]  # L transfers directly
    output[:, :, 1] = np.multiply(np.cos(data[:, :, 2] * np.pi / 180), data[:, :, 1])
    output[:, :, 2] = np.multiply(np.sin(data[:, :, 2] * np.pi / 180), data[:, :, 1])

    return output


if __name__ == '__main__':
    print("cj_csc")
