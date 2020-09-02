import numpy as np
import os
import matplotlib.pyplot as plt
import cj_rawimage as rawimage
import cj_rgbimage as rgbmage


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


def test_show_bmp_histogram(image1, dtype, width, height, start_x, start_y, len_x, len_y, step_x, step_y, num, show):
    image = rgbmage.read_bmpimage(image1, width, height, dtype)
    testimage = image[start_y:(len_y + start_y):step_y, start_x:(len_x + start_x):step_x]
    # array_bins = np.arange(0, 256, 255 / num)  # 等差数列数组支持任意个数组元素
    array_bins = np.array([0, 40, 100, 170, 250, 256])  # 特殊数组单独添加测试
    testimage_flatten = testimage.flatten()  # 将二维数组转成一维数组
    n = plt.hist(testimage_flatten, bins=array_bins)  # 第一个参数必须是一个一维数组
    np.set_printoptions(precision=1, suppress=True)  # 设置输出小数点位数 取消科学计数法
    print("平均灰度:", format(testimage_flatten.mean(), '.1f'))  # 上一句设置小数点位数对它无效，进行另外设置
    print("直方图统计个数:", n[0])
    percent = np.zeros(num)
    for i in range(0, num):
        percent[i] = n[0][i] / n[0].sum()
    print("直方图统计百分比:", percent * 100)

    if show:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 添加这两句支持plt中文显示

        plt.xlabel('灰度值')
        plt.ylabel('个数')
        plt.title('histogram')
        plt.show()

    testimage = image[0:start_y:step_y, 0:width:step_x]
    testimage_flatten = testimage.flatten()  # 将二维数组转成一维数组
    print("ave1:", format(testimage_flatten.mean(), '.1f'))

    testimage = image[start_y:height:step_y, 0:start_x:step_x]
    testimage_flatten = testimage.flatten()  # 将二维数组转成一维数组
    print("ave2:", format(testimage_flatten.mean(), '.1f'))

    testimage = image[start_y:height:step_y, (start_x + len_x):width:step_x]
    testimage_flatten = testimage.flatten()  # 将二维数组转成一[[维数组
    print("ave4:", format(testimage_flatten.mean(), '.1f'))
    # cv.imwrite("../pic/123456.bmp", image)


if __name__ == "__main__":
    print('This is main of module')
    # file_name1 = "../pic/qrcode/pic_99.bmp"
    # # file_name1 = "../pic/run/exp4/paper/7-1/pic_5.bmp"
    # # get_statistcs_test()
    # test_show_bmp_histogram(file_name1, dtype="uint8", width=640, height=480, start_x=160, start_y=160, len_x=320,
    #                         len_y=320, step_x=4, step_y=4, num=5, show=1)

    for root, dirs, files in os.walk("../pic/qrcode/"):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            filename = os.path.join(root, f)
            print(filename)
            test_show_bmp_histogram(filename, dtype="uint8", width=640, height=480, start_x=160, start_y=160, len_x=320,
                                    len_y=320, step_x=4, step_y=4, num=5, show=0)
