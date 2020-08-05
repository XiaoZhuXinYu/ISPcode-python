from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage import filters

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from pycallgraph import GlobbingFilter

import cj_yuvimage as yuvimage
import cj_rawimage as rawimage
import cj_bmpimage as bmpimage


# 传统灰度世界法，只进行求平均，效率高
def grey_world(R, G, B):
    mu_r = np.average(R)
    mu_g = np.average(G)
    mu_b = np.average(B)

    R_gain = mu_g / mu_r
    G_gain = mu_g / mu_g
    B_gain = mu_g / mu_b

    return R_gain, G_gain, B_gain


# 动态阈值法，在ycbcr域上进行，算法简单，便于实现，来自一篇论文
def auto_threshold(R, G, B):
    x, y = R.shape
    ycc = yuvimage.rgb2ycbcr(R, G, B)
    Lu = ycc[:, :, 0]  # Lu Cb Cr 三者的shape是一样的。
    Cb = ycc[:, :, 1]
    Cr = ycc[:, :, 2]

    Mb = np.mean(Cb)  # 计算cbcr的均值 Mb Mr
    Mr = np.mean(Cr)
    Db = np.sum(abs(Cb - Mb)) / (x * y)  # 计算Cb Cr的方差,这边没有进行平方操作
    Dr = np.sum(abs(Cr - Mr)) / (x * y)

    # 根据阈值的要求提取出near- white区域的像素点
    b1 = np.abs(Cb - (Mb + Db * np.sign(Mb)))
    b2 = np.abs(Cr - (Mr + Dr * np.sign(Mr)))
    itemindex = np.where((b1 < (1.5 * Db)) & (b2 < (1.5 * Dr)))  # 公式3,4
    L_NW = Lu[itemindex]
    print("itemindex = ", itemindex)
    print("Lu shape = ", Lu.shape, "Cb shape = ", Cb.shape)
    print("L_NW = ", L_NW, "L_NW shape = ", L_NW.shape)

    L_NW_sorted = np.sort(L_NW)  # 像素点排序 从小到大
    print("L_NW_sorted = ", L_NW_sorted)

    # 提取满足公式3,4的，前10%的点，作为参考白点
    count = L_NW.shape  # 注意：经过前面的操作，L_NW和L_NW_sorted已经是一个一维数组了 count就是数组的长度了。
    nn = round(count[0] * 9 / 10)  # round表示
    print(count, count[0])  # count为(a, ) count[0]为a 注意格式
    threshold = L_NW_sorted[nn - 1]
    itemindex2 = np.where(L_NW >= threshold)  # 这边可以直接取数组的后1/10，python不考虑效率，代码先不做修改了。

    # 提取参考白点的RGB三信道的值
    R_NW = R[itemindex]  # itemindex表示图像中满足公式3,4点的坐标
    G_NW = G[itemindex]
    B_NW = B[itemindex]
    R_selected = R_NW[itemindex2]  # itemindex2表示R_NW中前10%点的坐标
    G_selected = G_NW[itemindex2]
    B_selected = B_NW[itemindex2]

    # 计算对应点的RGB均值
    mu_r = np.mean(R_selected)
    mu_g = np.mean(G_selected)
    mu_b = np.mean(B_selected)

    # 计算增益
    R_gain = mu_g / mu_r
    G_gain = mu_g / mu_g
    B_gain = mu_g / mu_b

    return R_gain, G_gain, B_gain


def set_border(im, width, method):
    # sets border to either zero method=0,or method=1 to average
    # 把图像im上下左右四条宽度为width的边灰度值设置为method
    hh, ll = im.shape
    im[0:width, :] = method
    im[hh - width:hh, :] = method
    im[:, 0:width] = method
    im[:, ll - width:ll] = method
    return im


# 功能：将im四条边上的数据分别于其临近的边对比数据，
# 如果边上的某个像素值小于临近边对应点的像素值，就用临近边的像素值替代边上原有点的像素值
def dilation33(im):
    hh, ll = im.shape
    out1 = np.zeros((hh, ll))
    out2 = np.zeros((hh, ll))
    out3 = np.zeros((hh, ll))
    # H 方向扩展上下像素
    out1[0:hh - 1, :] = im[1:hh, :]
    out1[hh - 1, :] = im[hh - 1, :]
    out2 = im
    out3[0, :] = im[0, :]
    out3[1:hh, :] = im[0:hh - 1, :]
    out_max = np.maximum(out1, out2)
    out_max = np.maximum(out_max, out3)
    # W方向扩展上下像素
    out1[:, 0:ll - 1] = out_max[:, 1:ll]
    out1[:, ll - 1] = out_max[:, ll - 1]
    out2 = out_max
    out3[:, 0] = out_max[:, 0]
    out3[:, 1:ll] = out_max[:, 0:ll - 1]
    out_max = np.maximum(out1, out2)
    out_max = np.maximum(out_max, out3)
    return out_max


def gDer(f, sigma, iorder, jorder):
    break_off_sigma = 3.
    H, W = f.shape
    filtersize = np.floor(break_off_sigma * sigma + 0.5)
    filtersize = filtersize.astype(np.int)
    # 扩展边
    f = np.pad(f, ((filtersize, filtersize), (filtersize, filtersize)), 'edge')
    x = np.arange(-filtersize, filtersize + 1)
    # 翻转滤波核
    x = x * -1
    Gauss = 1 / (np.power(2 * np.pi, 0.5) * sigma) * np.exp((x ** 2) / (-2 * sigma * sigma))
    print("Gauss:", Gauss, "x:", x)
    iorder = 2
    if iorder == 0:
        # 高斯滤波
        Gx = Gauss / sum(Gauss)
    elif iorder == 1:
        Gx = -(x / sigma ** 2) * Gauss  # 对x一阶求导
        print("gx1:", Gx)
        Gx = Gx / (np.sum(x * Gx))
        print("gx2:", Gx)
    elif iorder == 2:
        # 二阶求导
        Gx = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * Gauss  # 对x求二阶导数
        print("gx1:", Gx)
        Gx = Gx - sum(Gx) / (2 * filtersize + 1)
        Gx = Gx / sum(0.5 * x * x * Gx)
        print("gx2:", Gx)
    # 扩展到二维
    Gx = Gx.reshape(1, -1)
    # Gx=np.transpose([Gx])
    # 卷积
    h = signal.convolve(f, Gx, mode="same")

    # print(Gauss)
    if jorder == 0:
        Gy = Gauss / sum(Gauss)
    elif jorder == 1:
        Gy = -(x / sigma ** 2) * Gauss
        Gy = Gy / (np.sum(x * Gy))
    elif jorder == 2:
        Gy = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * Gauss
        Gy = Gy - sum(Gy) / (2 * filtersize + 1)
        Gy = Gy / sum(0.5 * x * x * Gy)
    # 扩展到二维,转成二维
    Gy = Gy.reshape(1, -1).T  # 注意这一步和Gx的区别
    res = signal.convolve(h, Gy, mode="same")
    res2 = res[1:2, 1:2]
    end_h = (filtersize + H)
    end_w = (filtersize + W)
    res2 = np.array(res)[filtersize:end_h, filtersize:end_w]
    return res2


def NormDerivative(img, sigma, order):
    # 一阶求导
    if order == 1:
        Ix = gDer(img, sigma, 1, 0)
        Iy = gDer(img, sigma, 0, 1)
        Iw = np.power(Ix ** 2 + Iy ** 2, 0.5)

    # 二阶求导
    if order == 2:  # computes frobius norm
        Ix = gDer(img, sigma, 2, 0)
        Iy = gDer(img, sigma, 0, 2)
        Ixy = gDer(img, sigma, 1, 1)
        Iw = np.power(Ix ** 2 + Iy ** 2 + 4 * Ixy, 0.5)
    return Iw


# njet 是否edge mink_norm shade的参数  sigma 滤波和求导参数
def grey_edge(R, G, B, njet=0, mink_norm=1, sigma=1, saturation_threshold=255):
    """
    Estimates the light source of an input_image as proposed in:
    J. van de Weijer, Th. Gevers, A. Gijsenij
    "Edge-Based Color Constancy"
    IEEE Trans. Image Processing, accepted 2007.
    Depending on the parameters the estimation is equal to Grey-World, Max-RGB, general Grey-World,
    Shades-of-Grey or Grey-Edge algorithm.
    """
    mask_im = np.zeros(R.shape)
    img_max = np.maximum(R, G)
    img_max = np.maximum(img_max, B)

    # 移除所有饱和像素
    itemindex = np.where(img_max >= saturation_threshold)
    saturation_map = np.zeros(R.shape)
    saturation_map[itemindex] = 1
    # 扩散
    mask_im = dilation33(saturation_map)
    mask_im = 1 - mask_im
    # 移除边的像素生成最终的有效像素mask
    mask_im2 = set_border(mask_im, sigma + 1, 0)
    # 不去掉饱和像素尤其是buiding图片差别很大
    # mask_im2 = np.ones(R.shape)
    if njet == 0:
        if sigma != 0:
            # 去噪
            gauss_image_R = filters.gaussian(R, sigma=sigma, multichannel=True)
            gauss_image_G = filters.gaussian(G, sigma=sigma, multichannel=True)
            gauss_image_B = filters.gaussian(B, sigma=sigma, multichannel=True)
        else:  # grey world2 shade of grey max rgb
            gauss_image_R = R
            gauss_image_G = G
            gauss_image_B = B
        deriv_image_R = gauss_image_R[:, :]
        deriv_image_G = gauss_image_G[:, :]
        deriv_image_B = gauss_image_B[:, :]
    else:
        # grey edge
        deriv_image_R = NormDerivative(R, sigma, njet)
        deriv_image_G = NormDerivative(G, sigma, njet)
        deriv_image_B = NormDerivative(B, sigma, njet)

    # estimate illuminations
    if mink_norm == -1:  # mink_norm = inf (max rgb)
        estimating_func = lambda x: np.max(x * mask_im2.astype(np.int))
    else:
        estimating_func = lambda x: np.power(np.sum(np.power(x * mask_im2.astype(np.int), mink_norm)), 1 / mink_norm)
    RS = np.sum(np.power(deriv_image_R, mink_norm))
    GS = np.sum(np.power(deriv_image_G, mink_norm))
    BS = np.sum(np.power(deriv_image_B, mink_norm))
    illum_R = estimating_func(deriv_image_R)
    illum_G = estimating_func(deriv_image_G)
    illum_B = estimating_func(deriv_image_B)

    illum_max = np.maximum(illum_R, illum_G)
    illum_max = np.maximum(illum_max, illum_B)

    R_gain = illum_max / illum_R
    G_gain = illum_max / illum_G
    B_gain = illum_max / illum_B

    return R_gain, G_gain, B_gain


def apply_raw(pattern, R, GR, GB, B, R_gain, G_gain, B_gain, max):
    R = np.minimum(R * R_gain, max)
    GR = np.minimum(GR * G_gain, max)
    GB = np.minimum(GB * G_gain, max)
    B = np.minimum(B * B_gain, max)
    result_image = rawimage.bayer_channel_integration(R, GR, GB, B, pattern)
    return result_image


def apply_to_rgb(R, G, B, R_gain, G_gain, B_gain):
    h, w = R.shape
    img = np.zeros(shape=(h, w, 3))
    img[:, :, 0] = np.minimum(R * R_gain, 255)
    img[:, :, 1] = np.minimum(G * G_gain, 255)
    img[:, :, 2] = np.minimum(B * B_gain, 255)
    return img


def raw_white_balance(type):
    pattern = "GRBG"
    file_name = "../pic/awb/D65_4032_2752_GRBG_1_LSC.raw"
    image = rawimage.read_plained_file(file_name, dtype="uint16", width=4032, height=2752, shift_bits=0)
    # rawimage.show_planedraw(image, width=4032, height=2752, pattern="MONO", sensorbit=10, compress_ratio=1)

    max = 1023
    R, GR, GB, B, G = rawimage.bayer_channel_separation(image, pattern)
    if type == "grey_world":
        R_gain, G_gain, B_gain = grey_world(R, G, B)
    elif type == "auto_threshold":
        R_gain, G_gain, B_gain = auto_threshold(R, G, B)
    elif type == "grey_world2":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=1, sigma=0, saturation_threshold=max)
    elif type == "shade_of_grey":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=5, sigma=0, saturation_threshold=max)
    elif type == "max_RGB":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=-1, sigma=0, saturation_threshold=max)
    elif type == "grey_edge":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=1, mink_norm=5, sigma=2, saturation_threshold=max)

    result_image = apply_raw(pattern, R, GR, GB, B, R_gain, G_gain, B_gain, max)

    # rawimage.show_planedraw(result_image, width=4032, height=2742, pattern="MONO", sensorbit=10, compress_ratio=1)
    # rawimage.show_planedraw(result_image, width=4032, height=2752, pattern="GRBG", sensorbit=10, compress_ratio=1)

    h, w = R.shape
    img = np.zeros(shape=(h, w, 3))
    img2 = np.zeros(shape=(h, w, 3))
    img[:, :, 0] = R
    img[:, :, 1] = G
    img[:, :, 2] = B
    R2, GR2, GB2, B2, G2 = rawimage.bayer_channel_separation(result_image, pattern)
    img2[:, :, 0] = R2
    img2[:, :, 1] = GR2
    img2[:, :, 2] = B2

    bmpimage.show_bmpimage(img, width=w, height=h, sensorbit=10, color="color", compress_ratio=1)
    bmpimage.show_bmpimage(img2, width=w, height=h, sensorbit=10, color="color", compress_ratio=1)


def RGB_white_balance(type):
    # image, type
    # image = plt.imread('building.jpg')
    image = plt.imread('8D5U5568_D_N.png')
    if np.max(image) <= 1:
        image = image * 255
    # type = "auto_threshold"
    result_image = None
    processor = None
    R, G, B = bmpimage.rgb_separation(image)
    if type == "grey_world":
        R_gain, G_gain, B_gain = grey_world(R, G, B)
    elif type == "auto_threshold":
        R_gain, G_gain, B_gain = auto_threshold(R, G, B)
    elif type == "grey_world2":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=1, sigma=0)
    elif type == "shade_of_grey":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=5, sigma=0)
    elif type == "max_RGB":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=-1, sigma=0)
    elif type == "grey_edge":
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=1, mink_norm=5, sigma=2)

    result_image = apply_to_rgb(R, G, B, R_gain, G_gain, B_gain)

    # color.rgb_show(image / 255)
    # color.rgb_show(result_image / 255)
    h, w = R.shape
    bmpimage.show_bmpimage(image, width=w, height=h, sensorbit=8, color="color", compress_ratio=1)
    bmpimage.show_bmpimage(result_image, width=w, height=h, sensorbit=8, color="color", compress_ratio=1)


if __name__ == "__main__":
    config = Config()
    # 关系图中包括(include)哪些函数名。
    # 如果是某一类的函数，例如类gobang，则可以直接写'gobang.*'，表示以gobang.开头的所有函数。（利用正则表达式）。
    config.trace_filter = GlobbingFilter(include=[
        'main',
        'rgb2ycbcr',
        'grey_world',
        'auto_threshold',
        'set_border',
        'dilation33',
        'gDer',
        'NormDerivative',
        'grey_edge',
        'apply_raw',
        'apply_to_rgb',
        'rgb_separation',
        'raw_awb_separation',
        'raw_white_balance',
        'RGB_white_balance'
    ])
    # 该段作用是关系图中不包括(exclude)哪些函数。(正则表达式规则)
    # config.trace_filter = GlobbingFilter(exclude=[
    #     'pycallgraph.*',
    #     '*.secret_function',
    #     'FileFinder.*',
    #     'ModuleLockManager.*',
    #     'SourceFilLoader.*'
    # ])
    graphviz = GraphvizOutput()
    graphviz.output_file = 'graph_grey_edge.png'
    with PyCallGraph(output=graphviz, config=config):
        # 以下是main函数的内容
        type1 = "auto_threshold"  # auto_threshold grey_world grey_world2 shade_of_grey max_RGB grey_edge
        # RGB_white_balance(type1)
        raw_white_balance(type1)
