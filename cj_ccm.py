import numpy as np
import matplotlib.pyplot as plt
import cj_gamma as gamma
import cj_rgbimage as rgbimage


def CCM_convert(data, CCMMat, color_space="srgb", clip_range=[0, 255]):
    # CCM工作在线性RGB因此需要先进行degamma
    if color_space == "srgb":
        # degamma / linearization
        data = gamma.degamma_srgb(data, clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])
        # 归一化,可不这么做

    # matrix multiplication`
    output = np.empty(np.shape(data), dtype=np.float32)
    output[:, :, 0] = data[:, :, 0] * CCMMat[0, 0] + data[:, :, 1] * CCMMat[0, 1] + data[:, :, 2] * CCMMat[0, 2]
    output[:, :, 1] = data[:, :, 0] * CCMMat[1, 0] + data[:, :, 1] * CCMMat[1, 1] + data[:, :, 2] * CCMMat[1, 2]
    output[:, :, 2] = data[:, :, 0] * CCMMat[2, 0] + data[:, :, 1] * CCMMat[2, 1] + data[:, :, 2] * CCMMat[2, 2]
    if color_space == "srgb":
        # gamma
        output = output * clip_range[1]
        output = gamma.gamma_srgb(output, clip_range)
    return output


if __name__ == "__main__":
    CCM = np.array([[1.507812, -0.546875, 0.039062],
                    [-0.226562, 1.085938, 0.140625],
                    [-0.062500, -0.648438, 1.718750], ])
    maxvalue = 255
    image = plt.imread("../pic/ccm/kodim19.png")
    if np.max(image) <= 1:
        image = image * maxvalue

    new_image = CCM_convert(image, CCM, color_space="srgb", clip_range=[0, maxvalue])
    rgbimage.rgb_image_show_color(image, maxvalue=255,  color="color", compress_ratio=1)
    rgbimage.rgb_image_show_color(new_image, maxvalue=255,  color="color", compress_ratio=1)
