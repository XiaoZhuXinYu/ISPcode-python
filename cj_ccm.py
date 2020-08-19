import numpy as np
import color_utils as color
import matplotlib.pyplot as plt


def CCM_convert(data, CCM, color_space="srgb", clip_range=[0, 255]):
    # CCM工作在线性RGB因此需要先进行degamma
    if color_space == "srgb":
        # degamma / linearization
        data = color.degamma_srgb(data, clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])
        # 归一化,可不这么做

    # matrix multiplication`
    output = np.empty(np.shape(data), dtype=np.float32)
    output[:, :, 0] = data[:, :, 0] * CCM[0, 0] + data[:, :, 1] * CCM[0, 1] + data[:, :, 2] * CCM[0, 2]
    output[:, :, 1] = data[:, :, 0] * CCM[1, 0] + data[:, :, 1] * CCM[1, 1] + data[:, :, 2] * CCM[1, 2]
    output[:, :, 2] = data[:, :, 0] * CCM[2, 0] + data[:, :, 1] * CCM[2, 1] + data[:, :, 2] * CCM[2, 2]
    if color_space == "srgb":
        # gamma
        output = output * clip_range[1]
        output = color.gamma_srgb(output, clip_range)
    return output


if __name__ == "__main__":
    # read image
    # read_image
    CCM = np.array([[1.507812, -0.546875, 0.039062],
                    [-0.226562, 1.085938, 0.140625],
                    [-0.062500, -0.648438, 1.718750], ])
    maxvalue = 255
    image = plt.imread('kodim19.png')
    if np.max(image) <= 1:
        image = image * maxvalue

    new_image = CCM_convert(image, CCM, color_space="srgb", clip_range=[0, maxvalue])
    color.rgb_show(image / 255)
    color.rgb_show(new_image / 255)
