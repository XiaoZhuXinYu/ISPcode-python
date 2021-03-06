import numpy as np
from matplotlib import pyplot as plt
import cj_csc as csc
import math


# 显示RGB图像
def rgb_image_show(image, width, height, compress_ratio=1):
    x = width/(compress_ratio*100)
    y = height/(compress_ratio*100)
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(image, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()


def ycbcrshow(image, width, height):
    imagergb = csc.ycbcr2rgb(image, width, height)
    rgb_image_show(imagergb, width, height)


# 支持的yuv类型 ：
# yuv422：YUYV YVYU UYVY VYUY YUV422P-I422( Y U V)
# yuv420：YU12(Y U V) YV12(Y V U) NV12(Y UV) NV21(Y VU)
def read_yuv_file(filename, width, height, yuvformat):
    # 文件长度
    if yuvformat == 'YUYV' or yuvformat == 'YVYU' or yuvformat == 'UYVY' or yuvformat == 'VYUY' or yuvformat == 'YUV422P':
        image_bytes = int(width * height * 2)
        h_h = height
        h_w = width // 2
    elif yuvformat == 'YU12' or yuvformat == 'YV12' or yuvformat == 'NV12' or yuvformat == 'NV21':
        image_bytes = int(width * height * 3 / 2)
        h_h = height // 2
        h_w = width // 2
    else:
        print(yuvformat, "is not a support format")
        return 0

    # 读出文件
    frame = np.fromfile(filename, count=image_bytes, dtype="uint8")

    Yt = np.zeros(shape=(height, width))
    Cb = np.zeros(shape=(h_h, h_w))
    Cr = np.zeros(shape=(h_h, h_w))

    # 读取之后直接做reshape
    if yuvformat == 'YUYV':
        Yt[:, :] = np.reshape(frame[0:image_bytes:2], newshape=(height, width))
        Cb[:, :] = np.reshape(frame[1:image_bytes:4], newshape=(h_h, h_w))
        Cr[:, :] = np.reshape(frame[3:image_bytes:4], newshape=(h_h, h_w))
    elif yuvformat == 'YVYU':
        Yt[:, :] = np.reshape(frame[0:image_bytes:2], newshape=(height, width))
        Cb[:, :] = np.reshape(frame[3:image_bytes:4], newshape=(h_h, h_w))
        Cr[:, :] = np.reshape(frame[1:image_bytes:4], newshape=(h_h, h_w))
    elif yuvformat == 'UYVY':
        Yt[:, :] = np.reshape(frame[1:image_bytes:2], newshape=(height, width))
        Cb[:, :] = np.reshape(frame[0:image_bytes:4], newshape=(h_h, h_w))
        Cr[:, :] = np.reshape(frame[2:image_bytes:4], newshape=(h_h, h_w))
    elif yuvformat == 'VYUY':
        Yt[:, :] = np.reshape(frame[1:image_bytes:2], newshape=(height, width))
        Cb[:, :] = np.reshape(frame[2:image_bytes:4], newshape=(h_h, h_w))
        Cr[:, :] = np.reshape(frame[0:image_bytes:4], newshape=(h_h, h_w))
    elif yuvformat == 'YUV422P':
        Yt[:, :] = np.reshape(frame[0:width * height], newshape=(height, width))
        Cb[:, :] = np.reshape(frame[width * height:width * height * 3 / 2], newshape=(h_h, h_w))
        Cr[:, :] = np.reshape(frame[width * height * 3 / 2:image_bytes], newshape=(h_h, h_w))
    elif yuvformat == 'YU12':
        Yt[:, :] = np.reshape(frame[0:width * height], newshape=(height, width))
        Cb[:, :] = np.reshape(frame[width * height:width * height * 5 / 4], newshape=(h_h, h_w))
        Cr[:, :] = np.reshape(frame[height:width * height * 5 / 4:image_bytes], newshape=(h_h, h_w))
    elif yuvformat == 'YV12':
        Yt[:, :] = np.reshape(frame[0:width * height], newshape=(height, width))
        Cr[:, :] = np.reshape(frame[width * height:width * height * 5 / 4], newshape=(h_h, h_w))
        Cb[:, :] = np.reshape(frame[height:width * height * 5 / 4:image_bytes], newshape=(h_h, h_w))
    elif yuvformat == 'NV12':
        Yt[:, :] = np.reshape(frame[0:width * height], newshape=(height, width))
        Cb[:, :] = np.reshape(frame[width * height:image_bytes:2], newshape=(h_h, h_w))
        Cr[:, :] = np.reshape(frame[width * height + 1:image_bytes:2], newshape=(h_h, h_w))
    elif yuvformat == 'NV21':
        Yt[:, :] = np.reshape(frame[0:width * height], newshape=(height, width))
        Cr[:, :] = np.reshape(frame[width * height:image_bytes:2], newshape=(h_h, h_w))
        Cb[:, :] = np.reshape(frame[width * height + 1:image_bytes:2], newshape=(h_h, h_w))

    # 由422或420扩展到444
    if yuvformat == 'YUYV' or yuvformat == 'YVYU' or yuvformat == 'UYVY' or yuvformat == 'VYUY' or yuvformat == 'YUV422P':
        Cb = Cb.repeat(2, 1)
        Cr = Cr.repeat(2, 1)
    elif yuvformat == 'YU12' or yuvformat == 'YV12' or yuvformat == 'NV12' or yuvformat == 'NV21':
        Cb = Cb.repeat(2, 0)
        Cb = Cb.repeat(2, 1)
        Cr = Cr.repeat(2, 0)
        Cr = Cr.repeat(2, 1)

    # 拼接到Ycbcr444
    img = np.zeros(shape=(height, width, 3))
    img[:, :, 0] = Yt[:, :]
    img[:, :, 1] = Cb[:, :]
    img[:, :, 2] = Cr[:, :]
    return img


if __name__ == '__main__':
    testimg = read_yuv_file(filename='NV12_1920(1920)x1080.yuv', height=1080, width=1920, yuvformat='NV12')
    print(np.max(testimg), np.min(testimg))
    rgb = csc.ycbcr2rgb(testimg, height=1080, width=1920)
    rgb_image_show(rgb/255, height=1080, width=1920, compress_ratio=1)
