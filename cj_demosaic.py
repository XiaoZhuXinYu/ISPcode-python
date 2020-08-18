import cj_rawimage as rawimage
import cj_yuvimage as yuvimage
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import cv2 as cv


# 得到RGB三个分量的模板，例如说R分量的模板，raw图中存R分量的位置为1，存G B 分量的位置为0。
# 但是三个分量的模板和raw图的shape是相同的。
def masks_Bayer(im, pattern):
    w, h = im.shape
    R = np.zeros((w, h))
    GR = np.zeros((w, h))
    GB = np.zeros((w, h))
    B = np.zeros((w, h))

    # 将对应位置的元素取出来,因为懒所以没有用效率最高的方法,大家可以自己去实现
    if pattern == "RGGB":
        R[::2, ::2] = 1
        GR[::2, 1::2] = 1
        GB[1::2, ::2] = 1
        B[1::2, 1::2] = 1
    elif pattern == "GRBG":
        GR[::2, ::2] = 1
        R[::2, 1::2] = 1
        B[1::2, ::2] = 1
        GB[1::2, 1::2] = 1
    elif pattern == "GBRG":
        GB[::2, ::2] = 1
        B[::2, 1::2] = 1
        R[1::2, ::2] = 1
        GR[1::2, 1::2] = 1
    elif pattern == "BGGR":
        B[::2, ::2] = 1
        GB[::2, 1::2] = 1
        GR[1::2, ::2] = 1
        R[1::2, 1::2] = 1
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    R_m = R
    G_m = GB + GR
    B_m = B
    return R_m, G_m, B_m


def blinnear(img, pattern):
    img = img.astype(np.float)
    R_m, G_m, B_m = masks_Bayer(img, pattern)  # 得到rgb三个分量的模板

    H_G = np.array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4  # yapf: disable

    H_RB = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4  # yapf: disable

    R = signal.convolve(img * R_m, H_RB, 'same')
    G = signal.convolve(img * G_m, H_G, 'same')
    B = signal.convolve(img * B_m, H_RB, 'same')
    h, w = img.shape
    result_img = np.zeros((h, w, 3))

    result_img[:, :, 0] = R
    result_img[:, :, 1] = G
    result_img[:, :, 2] = B
    del R_m, G_m, B_m, H_RB, H_G
    return result_img


def AH_gradient(img, pattern):
    X = img
    Rm, Gm, Bm = masks_Bayer(img, pattern)  # 得到rgb三个分量的模板
    # green
    Hg1 = np.array([0, 1, 0, -1, 0])
    Hg2 = np.array([-1, 0, 2, 0, -1])

    Hg1 = Hg1.reshape(1, -1)
    Hg2 = Hg2.reshape(1, -1)
    Ga = (Rm + Bm) * (np.abs(signal.convolve(X, Hg1, 'same')) + np.abs(signal.convolve(X, Hg2, 'same')))

    return Ga


def AH_gradientX(img, pattern):
    Ga = AH_gradient(img, pattern)

    return Ga


def AH_gradientY(img, pattern):  # 计算y方向上的梯度相当于把图像进行了翻转，pattern就变了。
    if pattern == "RGGB":
        new_pattern = "RGGB"
    elif pattern == "GRBG":
        new_pattern = "GBRG"
    elif pattern == "GBRG":
        new_pattern = "GRBG"
    elif pattern == "BGGR":
        new_pattern = "BGGR"
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    new_img = img.T
    Ga = AH_gradient(new_img, new_pattern)
    new_Ga = Ga.T
    return new_Ga


# 插值函数
def AH_interpolate(img, pattern, gamma, max_value):
    X = img
    Rm, Gm, Bm = masks_Bayer(img, pattern)  # 得到rgb三个分量的模板

    # green
    Hg1 = np.array([0, 1 / 2, 0, 1 / 2, 0])
    Hg2 = np.array([-1 / 4, 0, 1 / 2, 0, -1 / 4])
    Hg = Hg1 + Hg2 * gamma  # shape 为 (5,) 矩阵Hg参考公众号的论文
    Hg = Hg.reshape(1, -1)  # shape 为 (1,5)
    G = Gm * X + (Rm + Bm) * signal.convolve(X, Hg, mode='same')  # 得到所有的G

    # red / blue
    Hr = [[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]]
    R = G + signal.convolve(Rm * (X - G), Hr, 'same')
    B = G + signal.convolve(Bm * (X - G), Hr, 'same')

    R = np.clip(R, 0, max_value)
    G = np.clip(G, 0, max_value)
    B = np.clip(B, 0, max_value)
    return R, G, B


# X方向进行插值
def AH_interpolateX(img, pattern, gamma, max_value):
    h, w = img.shape
    Y = np.zeros((h, w, 3))
    R, G, B = AH_interpolate(img, pattern, gamma, max_value)
    Y[:, :, 0] = R
    Y[:, :, 1] = G
    Y[:, :, 2] = B
    return Y


# Y方向进行插值
def AH_interpolateY(img, pattern, gamma, max_value):
    h, w = img.shape
    Y = np.zeros((h, w, 3))
    if pattern == "RGGB":
        new_pattern = "RGGB"
    elif pattern == "GRBG":
        new_pattern = "GBRG"
    elif pattern == "GBRG":
        new_pattern = "GRBG"
    elif pattern == "BGGR":
        new_pattern = "BGGR"
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    new_img = img.T
    R, G, B = AH_interpolate(new_img, new_pattern, gamma, max_value)
    Y[:, :, 0] = R.T
    Y[:, :, 1] = G.T
    Y[:, :, 2] = B.T
    return Y


def MNballset(delta):
    #   MNballset returns a set of convolution filters describing
    #   the relative locations in the elements of the ball set.
    #
    #   This algorithm was developed according to Hirakawa's master's
    #   thesis.
    #

    index = delta
    H = np.zeros((index * 2 + 1, index * 2 + 1, (index * 2 + 1) ** 2))  # initialize

    k = 0
    for i in range(-index, index + 1):
        for j in range(-index, index + 1):
            if np.linalg.norm([i, j]) <= delta:
                p = np.linalg.norm([i, j])
                H[index + i, index + j, k] = 1  # included
                k = k + 1
    H = H[:, :, 0:k]
    return H


def MNparamA(YxLAB, YyLAB):
    #
    # epsilon = min(max(left, right), max(top, bottm))
    X = YxLAB
    Y = YyLAB
    kernel_H1 = np.array([1, -1, 0])
    kernel_H1 = kernel_H1.reshape(1, -1)
    kernel_H2 = np.array([0, -1, 1])
    kernel_H2 = kernel_H2.reshape(1, -1)
    kernel_V1 = kernel_H1.reshape(1, -1).T
    kernel_V2 = kernel_H2.reshape(1, -1).T
    xxx = np.abs(signal.convolve(X[:, :, 0], kernel_H1, 'same'))
    eLM1 = np.maximum(np.abs(signal.convolve(X[:, :, 0], kernel_H1, 'same')),
                      np.abs(signal.convolve(X[:, :, 0], kernel_H2, 'same')))
    eLM2 = np.maximum(np.abs(signal.convolve(Y[:, :, 0], kernel_V1, 'same')),
                      abs(signal.convolve(Y[:, :, 0], kernel_V2, 'same')))
    eL = np.minimum(eLM1, eLM2)
    eCx = np.maximum(
        signal.convolve(X[:, :, 1], kernel_H1, 'same') ** 2 + signal.convolve(X[:, :, 2], kernel_H1, 'same') ** 2,
        signal.convolve(X[:, :, 1], kernel_H2, 'same') ** 2 + signal.convolve(X[:, :, 2], kernel_H2, 'same') ** 2)
    eCy = np.maximum(
        signal.convolve(Y[:, :, 1], kernel_V2, 'same') ** 2 + signal.convolve(Y[:, :, 2], kernel_V2, 'same') ** 2,
        signal.convolve(Y[:, :, 1], kernel_V1, 'same') ** 2 + signal.convolve(Y[:, :, 2], kernel_V1, 'same') ** 2)
    eC = np.minimum(eCx, eCy)
    eL = eL
    eC = eC ** 0.5
    return eL, eC


# 计算相似度ƒ
def MNhomogeneity(LAB_image, delta, epsilonL, epsilonC):
    index = delta
    H = MNballset(delta)
    X = LAB_image
    epsilonC_sq = epsilonC ** 2
    h, w, c = LAB_image.shape
    K = np.zeros((h, w))
    kh, kw, kc = H.shape

    # 注意浮点数精度可能会有影响
    for i in range(kc):
        L = np.abs(signal.convolve(X[:, :, 0], H[:, :, i], 'same') - X[:, :, 0]) <= epsilonL  # level set
        C = ((signal.convolve(X[:, :, 1], H[:, :, i], 'same') - X[:, :, 1]) ** 2 + (
                    signal.convolve(X[:, :, 2], H[:, :, i], 'same') - X[:, :, 2]) ** 2) <= epsilonC_sq  # color set
        U = C & L  # metric neighborhood
        K = K + U  # homogeneity
    return K


# 去artifact
def MNartifact(R, G, B, iterations):
    h, w = R.shape
    Rt = np.zeros((h, w, 8))
    Bt = np.zeros((h, w, 8))
    Grt = np.zeros((h, w, 4))
    Gbt = np.zeros((h, w, 4))
    kernel_1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    kernel_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    kernel_3 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    kernel_4 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    kernel_5 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    kernel_7 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    kernel_8 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    kernel_9 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

    for i in range(iterations):
        Rt[:, :, 0] = signal.convolve(R - G, kernel_1, 'same')
        Rt[:, :, 1] = signal.convolve(R - G, kernel_2, 'same')
        Rt[:, :, 2] = signal.convolve(R - G, kernel_3, 'same')
        Rt[:, :, 3] = signal.convolve(R - G, kernel_4, 'same')
        Rt[:, :, 4] = signal.convolve(R - G, kernel_6, 'same')
        Rt[:, :, 5] = signal.convolve(R - G, kernel_7, 'same')
        Rt[:, :, 6] = signal.convolve(R - G, kernel_8, 'same')
        Rt[:, :, 7] = signal.convolve(R - G, kernel_9, 'same')

        Rm = np.median(Rt, axis=2)
        R = G + Rm

        Bt[:, :, 0] = signal.convolve(B - G, kernel_1, 'same')
        Bt[:, :, 1] = signal.convolve(B - G, kernel_2, 'same')
        Bt[:, :, 2] = signal.convolve(B - G, kernel_3, 'same')
        Bt[:, :, 3] = signal.convolve(B - G, kernel_4, 'same')
        Bt[:, :, 4] = signal.convolve(B - G, kernel_6, 'same')
        Bt[:, :, 5] = signal.convolve(B - G, kernel_7, 'same')
        Bt[:, :, 6] = signal.convolve(B - G, kernel_8, 'same')
        Bt[:, :, 7] = signal.convolve(B - G, kernel_9, 'same')

        Bm = np.median(Bt, axis=2)
        B = G + Bm

        Grt[:, :, 0] = signal.convolve(G - R, kernel_2, 'same')
        Grt[:, :, 1] = signal.convolve(G - R, kernel_4, 'same')
        Grt[:, :, 2] = signal.convolve(G - R, kernel_6, 'same')
        Grt[:, :, 3] = signal.convolve(G - R, kernel_8, 'same')
        Grm = np.median(Grt, axis=2)
        Gr = R + Grm

        Gbt[:, :, 0] = signal.convolve(G - B, kernel_2, 'same')
        Gbt[:, :, 1] = signal.convolve(G - B, kernel_4, 'same')
        Gbt[:, :, 2] = signal.convolve(G - B, kernel_6, 'same')
        Gbt[:, :, 3] = signal.convolve(G - B, kernel_8, 'same')
        Gbm = np.median(Gbt, axis=2)
        Gb = B + Gbm
        G = (Gr + Gb) / 2
    return R, G, B


# adams hamilton
def AH_demosaic(img, pattern, gamma=1, max_value=255):
    print("AH demosaic start")
    imgh, imgw = img.shape
    imgs = 10  # 向外延申10个像素不改变 pattern。
    # X,Y方向插值
    # 扩展大小
    # Y = [X(n + 1:-1: 2, n + 1: -1:2,:)       X(n + 1: -1:2,:,:)        X(n + 1: -1:2, end - 1: -1:end - n,:)
    # X(:, n + 1: -1:2,:)               X                                X(:, end - 1: -1:end - n,:)
    # X(end - 1: -1:end - n, n + 1: -1:2,:)  X(end - 1: -1:end - n,:,:)  X(end - 1: -1:end - n, end - 1: -1:end - n,:)];
    f = np.pad(img, ((imgs, imgs), (imgs, imgs)), 'reflect')
    Yx = AH_interpolateX(f, pattern, gamma, max_value)  # 对x方向进行插值,得到的是含有RGB三个分量的数据
    Yy = AH_interpolateY(f, pattern, gamma, max_value)  # 对y方向进行插值,得到的是含有RGB三个分量的数据
    Hx = AH_gradientX(f, pattern)  # 这边计算的是梯度，梯度越小，相似度就越大
    Hy = AH_gradientY(f, pattern)
    # set output to Yy if Hy <= Hx
    index = np.where(Hy <= Hx)
    R = Yx[:, :, 0]
    G = Yx[:, :, 1]
    B = Yx[:, :, 2]
    Ry = Yy[:, :, 0]
    Gy = Yy[:, :, 1]
    By = Yy[:, :, 2]
    Rs = R
    Gs = G
    Bs = B
    Rs[index] = Ry[index]
    Gs[index] = Gy[index]
    Bs[index] = By[index]
    h, w = Rs.shape
    Y = np.zeros((h, w, 3))
    Y[:, :, 0] = Rs
    Y[:, :, 1] = Gs
    Y[:, :, 2] = Bs
    # 调整size和值的范畴
    # Y = np.clip(Y, 0, max_value)
    resultY = Y[imgs:imgs + imgh, imgs:imgs + imgw, :]
    return resultY


# 改进方向：转到lab空间计算相似度，花费了大量的时间。有些改进可以在yuv域进行计算，但是lab域计算的效果还是很好的。
def AHD(img, pattern, delta=2, gamma=1, maxvalue=255):  # delta是用来调试 artifact 的，gamma 不建议调试
    print("AHD demosaic start")
    iterations = 2
    imgh, imgw = img.shape
    imgs = 10  # 向外扩展10不改变原有的pattern

    f = np.pad(img, ((imgs, imgs), (imgs, imgs)), 'reflect')  # 扩展大小
    Yx = AH_interpolateX(f, pattern, gamma, maxvalue)  # 对x方向进行插值,得到的是含有RGB三个分量的数据
    Yy = AH_interpolateY(f, pattern, gamma, maxvalue)  # 对y方向进行插值,得到的是含有RGB三个分量的数据
    # 转LAB
    YxLAB = yuvimage.rgb2lab(Yx)
    YyLAB = yuvimage.rgb2lab(Yy)
    # 色彩差异的运算
    epsilonL, epsilonC = MNparamA(YxLAB, YyLAB)
    Hx = MNhomogeneity(YxLAB, delta, epsilonL, epsilonC)  # homogeneity 这边计算的是相似度，和AH算法计算梯度不同
    Hy = MNhomogeneity(YyLAB, delta, epsilonL, epsilonC)
    f_kernel = np.ones((3, 3))
    Hx = signal.convolve(Hx, f_kernel, 'same')
    Hy = signal.convolve(Hy, f_kernel, 'same')
    # 选择 X,Y
    # set output initially to Yx
    R = Yx[:, :, 0]
    G = Yx[:, :, 1]
    B = Yx[:, :, 2]
    Ry = Yy[:, :, 0]
    Gy = Yy[:, :, 1]
    By = Yy[:, :, 2]
    # set output to Yy if Hy >= Hx
    # 所有的都找到
    bigger_index = np.where(Hy >= Hx)
    Rs = R
    Gs = G
    Bs = B
    Rs[bigger_index] = Ry[bigger_index]
    Gs[bigger_index] = Gy[bigger_index]
    Bs[bigger_index] = By[bigger_index]
    h, w = Rs.shape
    YT = np.zeros((h, w, 3))
    YT[:, :, 0] = Rs
    YT[:, :, 1] = Gs
    YT[:, :, 2] = Bs
    # 去掉artifact
    Rsa, Gsa, Bsa = MNartifact(Rs, Gs, Bs, iterations)  # find
    # R and B
    # values
    #

    Y = np.zeros((h, w, 3))
    Y[:, :, 0] = Rsa
    Y[:, :, 1] = Gsa
    Y[:, :, 2] = Bsa

    # 调整size和值的范畴
    Y = np.clip(Y, 0, maxvalue)
    resultY = Y[imgs:imgs + imgh, imgs:imgs + imgw, :]
    return resultY


def test_blinnear_demosaic():
    pattern = "GRBG"
    file_name = "../pic/demosaic/kodim19.raw"
    maxvalue = 255
    h = 768
    w = 512
    image = rawimage.read_plained_file(file_name, dtype="uint16", width=w, height=h, shift_bits=0)
    rawimage.show_planedraw(image, w, h, pattern, sensorbit=8, compress_ratio=1)
    result = blinnear(image, pattern)
    height, width = image.shape
    x = width / 100
    y = height / 100
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(result / maxvalue, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    print("hello")
    return


def test_AH_demosaic():
    pattern = "GRBG"
    file_name = "../pic/demosaic/kodim19.raw"
    h = 768
    w = 512
    maxvalue = 255
    image = rawimage.read_plained_file(file_name, dtype="uint16", width=w, height=h, shift_bits=0)
    rawimage.show_planedraw(image, w, h, pattern, sensorbit=8, compress_ratio=1)
    result = AH_demosaic(image, pattern)
    height, width = image.shape
    x = width / 100
    y = height / 100
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(result / maxvalue, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    print("hello")
    return


def test_AHD_demosaic():
    pattern = "GRBG"  # 显示正确，保存下来的图片颜色不正确
    # pattern = "GBRG"  # 显示不正确，保存下来的图片颜色正确
    file_name = "../pic/demosaic/kodim19.raw"
    maxvalue = 255
    h = 768
    w = 512
    image = rawimage.read_plained_file(file_name, dtype="uint16", width=w, height=h, shift_bits=0)
    rawimage.show_planedraw(image, w, h, pattern="MONO", sensorbit=8, compress_ratio=1)
    result = AHD(image, pattern)
    height, width = image.shape
    x = width / 100
    y = height / 100
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(result / maxvalue, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    filename1 = file_name + "-ahd.bmp"
    cv.imwrite(filename1, result)
    print("hello")
    return


if __name__ == "__main__":
    # test_blinnear_demosaic()
    # test_AH_demosaic()
    test_AHD_demosaic()

    # data_R = scio.loadmat("data_R.mat")
    # R = data_R["R"]
    #
    # data_G = scio.loadmat("data_G.mat")
    # G = data_G["G"]
    #
    # data_B = scio.loadmat("data_B.mat")
    # B = data_B["B"]
    # RR,GG,BB=MNartifact(R, G, B, 2)

    # aa = np.array([1, 2, 3, 4])
    # aa.shape = (2, 2)
    # print(aa)
    # bb = aa.T
    # print(bb)

