import numpy as np
import cj_rawimage as rawimage
import matplotlib.pyplot as plt


# 统计直方图
def mono_cumuhistogram(image, max1):
    hist, bins = np.histogram(image, bins=range(0, max1 + 1))
    sum1 = 0
    for i in range(0, max1):
        sum1 = sum1 + hist[i]
        hist[i] = sum1
    return hist


def mono_average(image):
    a = np.mean(image)
    return a


# bayer直方图统计
def bayer_cumuhistogram(image, pattern, max1):
    R, GR, GB, B = rawimage.bayer_channel_separation(image, pattern)
    R_hist = mono_cumuhistogram(R, max1)
    GR_hist = mono_cumuhistogram(GR, max1)
    GB_hist = mono_cumuhistogram(GB, max1)
    B_hist = mono_cumuhistogram(B, max1)
    return R_hist, GR_hist, GB_hist, B_hist


# bayer的颜色通道的平均值
def bayer_average(image, pattern):
    R, GR, GB, B = rawimage.bayer_channel_separation(image, pattern)
    R_a = mono_average(R)
    GR_a = mono_average(GR)
    GB_a = mono_average(GB)
    B_a = mono_average(B)
    return R_a, GR_a, GB_a, B_a


# 获得块
def get_region(image, y1, x1, h, w):
    region_data = image[y1:y1 + h, x1:x1 + w]
    return region_data


# block和原图需要能够整除,返回值为浮点出
def binning_image(image, height, width, block_size_h, block_size_w, pattern):
    region_h_n = int(height / block_size_h)
    region_w_n = int(width / block_size_w)
    binning_image1 = np.empty((region_h_n * 2, region_w_n * 2), dtype=np.float32)
    x1 = 0
    y1 = 0
    for j in range(region_h_n):
        for i in range(region_w_n):
            region_data = get_region(image, y1, x1, block_size_h, block_size_w)
            R, GR, GB, B = rawimage.bayer_channel_separation(region_data, pattern)
            binning_image1[j * 2, i * 2] = np.mean(R)
            binning_image1[j * 2, (i * 2) + 1] = np.mean(GR)
            binning_image1[(j * 2) + 1, i * 2] = np.mean(GB)
            binning_image1[(j * 2) + 1, (i * 2) + 1] = np.mean(B)
            x1 = x1 + block_size_w
        y1 = y1 + block_size_h
        x1 = 0
    return binning_image1


def get_statistcs_test():
    b = np.fromfile("../pic/RAW_GRBG_plained_4608(9216)x3456_A.raw", dtype="uint16")
    print("b shape", b.shape)
    print('%#x' % b[0])
    b.shape = [3456, 4608]
    out = b.copy()
    out = out / 1023.0
    rawimage.raw_image_show_gray(out, 3456, 4608, 10)
    binning_image_data = binning_image(b, height=3456, width=4608, block_size_h=4, block_size_w=4, pattern='GRBG')
    size = binning_image_data.shape
    rawimage.raw_image_show_gray(binning_image_data / 1023, size[1], size[0])

    return 0


def get_statistcs_point():
    x = np.array([])
    y = np.array([])


if __name__ == "__main__":
    print('This is main of module')
    get_statistcs_test()
