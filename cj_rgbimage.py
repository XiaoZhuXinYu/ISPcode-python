import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def rgb_image_show_color(image, maxvalue=255,  color="color", compress_ratio=1):
    height = image.shape[0]
    width = image.shape[1]
    x = width / (compress_ratio * 100)
    y = height / (compress_ratio * 100)

    plt.figure(num='test', figsize=(x, y))
    if color == "gray":
        plt.imshow(image / maxvalue, cmap='gray', interpolation='bicubic', vmax=1.0)
    else:
        plt.imshow(image / maxvalue, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()


def read_bmpimage(image1, width, height, dtype):
    image = np.fromfile(image1, dtype)  # 相比于open，该函数可以指定数据类型
    image = image[1078:]
    image.shape = [height, width]
    # cv.imwrite("../pic/sm/000023-1.bmp", image)
    testimage = np.zeros(image.shape)
    # testimage = image  # 直接进行赋值，下面的for循环后，得不到想要的结果，可能和若拷贝有关
    # 采图工具采集到的图电脑上显示是正常的，但是数据确是和正常bmp图像上下对称的。
    # 也就是说两张图像的数据是关于x轴对称，但是电脑上显示的图却是一样的。特别记下。
    # 关于bmp图像的数据格式，我也没有过多研究，这边知道有这个问题即可，不必深究。
    for i in range(height):
        testimage[i] = image[height-1-i]
    # testimage = image
    return testimage


def rgb_separation(image):
    image = image.astype(np.float)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    return R, G, B


def rgb2gray(path):
    length = len(path)
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(path[:length-4] + "-gray.bmp", image)


if __name__ == "__main__":
    print('This is main of module')
    # file_name1 = "../pic/GAIN7EXP2.0_0.bmp"
    file_name1 = "../pic/demosaic/blinnear-demosaic.bmp"
    rgb2gray(file_name1)
    # image = read_bmpimage(file_name1, 640, 480, dtype="uint8")
    # rgb_image_show_color(image, maxvalue=255, color="color", compress_ratio=1)


