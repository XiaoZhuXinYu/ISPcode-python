import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import cj_csc as csc


def load_images(path, mode='color'):
    """
   FUNCTION: 将几张需要做hdr合成的图像加入队列中
    """
    image_stack = []
    i = 0
    for filename in os.listdir(path):
        print("Loading... /" + filename + "...as Image_stack[" + str(i) + "]")  # 这样可以显示出 str(i) 的内容
        # print("Loading... /" + filename + "...as Image_stack[str(i)]")  # 这样就只能单单显示字符串 str(i)

        if mode == 'color':
            image = cv.imread(os.path.join(path, filename))
            temp = image[:, :, 0].copy()
            image[:, :, 0] = image[:, :, 2]
            image[:, :, 2] = temp  # 将BGR 转成 RGB
        else:  # mode == 'gray':
            image = cv.imread(os.path.join(path, filename))
        image_stack.append(image)
        i += 1
    return image_stack


def alignment(image_stack):
    """
   FUNCTION: 将所有图像的大小调整成一致
     INPUTS:
        image_stack = 参与融合的图像栈
    OUTPUTS:
        调整完大小的图像栈
    """
    # '-----------------------------------------------------------------------------#
    sizes = []
    D = len(image_stack)  # 几张图片长度就是几，测试例子中为4
    for i in range(D):
        sizes.append(np.shape(image_stack[i]))  # 将每张图的shape复制给 sizes, 即(高度，宽度，3) 3表示rgb三个通道
    sizes = np.array(sizes)

    for i in range(D):
        if np.shape(image_stack[i])[:2] != (min(sizes[:, 0]), min(sizes[:, 1])):
            # 判断 当前图像 的宽高跟 所有参与融合图像 的最小宽，最小高是否相同
            print("Detected Non-Uniform Sized Image" + str(i) + " ... Resolving ...")
            image_stack[i] = cv.resize(image_stack[i], (min(sizes[:, 1]), min(sizes[:, 0])))  # 不相同就调整大小
            print(" *Done")
    print("\n")
    return image_stack


def contrast(image, ksize=1):
    """
   FUNCTION: contrast
        Call to compute the first quality measure: contrast using laplacian kernel
     INPUTS:
        image = input image (colored)
        ksize = 1 means: [[0,1,0],[1,-4,1],[0,1,0]] kernel
    OUTPUTS:
        contrast measure
    """
    # '-----------------------------------------------------------------------------#
    image = cv.cvtColor(image.astype('uint8'), cv.COLOR_BGR2GRAY)  # rgb转y，注意参数不能选择 COLOR_RGB2GRAY
    laplacian = cv.Laplacian(image.astype('float64'), cv.CV_64F, ksize)
    C = cv.convertScaleAbs(laplacian)  # 取绝对值
    C = cv.medianBlur(C.astype('float32'), 5)  # 中值滤波
    return C.astype('float64')


def saturation(image):
    """
   FUNCTION: saturation
        得到融合图像饱和度所占有的比例
     INPUTS:
        image = input image (colored)
    OUTPUTS:
        saturation measure
    """
    # '-----------------------------------------------------------------------------#
    S = np.std(image, 2)  # 计算标准差，要注意axis参数的用法。
    return S.astype('float64')


def exposedness(image, sigma=0.2):
    """
   FUNCTION: exposedness
        得到融合图像曝光部分所占的比例
     INPUTS:
        image = input image (colored)
        sigma = gaussian curve parameter
    OUTPUTS:
        exposedness measure
    """
    # '-----------------------------------------------------------------------------#
    image = cv.normalize(image, None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)  # 将图像进行 0-1 归一化
    gauss_curve = lambda i: np.exp(-((i - 0.5) ** 2) / (2 * sigma * sigma))
    R_gauss_curve = gauss_curve(image[:, :, 0])
    G_gauss_curve = gauss_curve(image[:, :, 1])
    B_gauss_curve = gauss_curve(image[:, :, 2])
    E = R_gauss_curve * G_gauss_curve * B_gauss_curve
    return E.astype('float64')


def scalar_weight_map(image_stack, weights=[1, 1, 1]):
    """
   FUNCTION: 得到image_stack 里每个像素的变换权重
   INPUTS:
        image_measures 以及 对比度，饱和度，亮度的权重
    OUTPUTS:
        得到一个和 image_stack 维数，大小都相同的一个权重表，后续hdr算法根据 image_stack 和得到的权重表进行融合。
    """
    # '-----------------------------------------------------------------------------#
    # print("image_stack[0].shape:", image_stack[0].shape)  # (428, 642, 3)
    H = np.shape(image_stack[0])[0]
    W = np.shape(image_stack[0])[1]
    D = len(image_stack)
    Wijk = np.zeros((H, W, D), dtype='float64')
    wc = weights[0]  # 确认对比度，饱和度，曝光的比例
    ws = weights[1]
    we = weights[2]
    print("Computing Weight Maps from Measures using: C=%1.1d, S=%1.1d, E=%1.1d" % (wc, ws, we))

    epsilon = 0.000005
    for i in range(D):
        C = contrast(image_stack[i])  # C.shape=(428, 642) S E 的shape和C相同
        S = saturation(image_stack[i])
        E = exposedness(image_stack[i])
        Wijk[:, :, i] = (np.power(C, wc) * np.power(S, ws) * np.power(E, we)) + epsilon  # Wijk.shape=(428, 642, 4)
    normalizer = np.sum(Wijk, 2)  # normalizer.shape=(428, 642) 这里要注意sum第二个参数的用法

    for i in range(D):
        # Wijk[:, :, i] 和 normalizer 的 shape 是相同的。
        Wijk[:, :, i] = np.divide(Wijk[:, :, i], normalizer)  # 和直接用 / 一样，都是矩阵的对应元素相除。
    print("*Done\n")
    return Wijk.astype('float64')


def measures_fusion_naive(image_stack, weight_maps, blurType=None, blurSize=(0, 0), blurSigma=15):
    """
   FUNCTION: measures_fusion_naive
         调用图像和权重进行融合归一化
     INPUTS:
        image_stack = 图像栈
        weight_maps = 权重
        blurType    = gaussian or bilateral filter applied to weight-map
        blurSize/Sigma = blurring parameters
    OUTPUTS:
        img_fused = single image with fusion of measures
        Rij = fusion of individual images with their weight maps
    """
    # '-----------------------------------------------------------------------------#
    H = np.shape(image_stack[0])[0]
    W = np.shape(image_stack[0])[1]
    D = len(image_stack)  # 图片的数量
    img_fused = np.zeros((H, W, 3), dtype='float64')

    if blurType == 'gaussian':
        print("Performing Gaussian-Blur Blending")
        Rij = []
        for i in range(D):
            weight_map = cv.GaussianBlur(weight_maps[:, :, i], blurSize, blurSigma)
            Rijk = image_stack[i] * np.dstack([weight_map, weight_map, weight_map])
            Rij.append(Rijk)
            img_fused += Rijk

    elif blurType == 'bilateral':
        print("Performing Bilateral-Blur Blending")
        Rij = []
        for i in range(D):
            weight_map = cv.bilateralFilter(weight_maps[:, :, i].astype('float32'), blurSigma, blurSize[0],
                                            blurSize[1])
            Rijk = image_stack[i] * np.dstack([weight_map, weight_map, weight_map])
            Rij.append(Rijk)
            img_fused += Rijk
    else:
        print("Performing Naive Blending")
        Rij = []
        for i in range(D):
            tmp = np.dstack([weight_maps[:, :, i], weight_maps[:, :, i], weight_maps[:, :, i]])  # shape: (428, 642, 3)
            Rijk = image_stack[i] * tmp  # shape: (428, 642, 3)
            print("shape:", tmp.shape, image_stack[i].shape, Rijk.shape)
            Rij.append(Rijk)
            img_fused += Rijk

    print(" *Done\n")

    return img_fused, Rij


def multires_pyramid(image, weight_map, levels):
    """
   FUNCTION: multires_pyramid 多分辨率金字塔
        Call to compute image and weights pyramids 计算图像和权重金字塔
     INPUTS:
        image_stack = list contains the stack of "exposure-bracketed" images
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weight_maps = scalar_weight_map for N images N张图像的标量权重图
        levels = height of pyramid to use including base pyramid base 要使用的金字塔高度，包括基础金字塔底
    OUTPUTS:
        imgLpyr = list containing image laplacian pyramid 包含图像拉普拉斯金字塔的列表
        wGpyr   = list containing weight gaussian pyramid 包含权重高斯金字塔的列表
    """
    # '-----------------------------------------------------------------------------#
    levels = levels - 1
    imgGpyr = [image]
    wGpyr = [weight_map]

    for i in range(levels):
        imgGpyr.append(cv.pyrDown(imgGpyr[i].astype('float64')))
        wGpyr.append(cv.pyrDown(wGpyr[i].astype('float64')))

    imgLpyr = [imgGpyr[levels]]
    wLpyr = [wGpyr[levels]]

    for i in range(levels, 0, -1):
        imgW = np.shape(imgGpyr[i - 1])[1]
        imgH = np.shape(imgGpyr[i - 1])[0]
        imgLpyr.append(imgGpyr[i - 1] - cv.resize(cv.pyrUp(imgGpyr[i]), (imgW, imgH)))

        imgW = np.shape(wGpyr[i - 1])[1]
        imgH = np.shape(wGpyr[i - 1])[0]
        wLpyr.append(wGpyr[i - 1] - cv.resize(cv.pyrUp(wGpyr[i]), (imgW, imgH)))

    return imgLpyr[::-1], wGpyr


def measures_fusion_multires(image_stack, weight_maps, levels=6):
    """
   FUNCTION: measures_fusion_multires 多分辨率曝光融合
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        levels = desired height of the pyramids 金字塔的期望高度
        weight_maps = scalar_weight_map for N images N张图像的标量权重图
    OUTPUTS:
        finalImage = single exposure fused image
    """
    # '-----------------------------------------------------------------------------#
    print("Performing Multiresolution Blending using: " + str(levels) + " Pyramid levels")
    D = np.shape(image_stack)[0]  # 表示图像栈中一共有几张图
    imgPyramids = []
    wPyramids = []
    # 图像和权重展开
    for i in range(D):
        imgLpyr, wGpyr = multires_pyramid(image_stack[i].astype('float64'), weight_maps[:, :, i], levels)
        imgPyramids.append(imgLpyr)
        wPyramids.append(wGpyr)

    # 图像和权重相乘
    blendedPyramids = []
    for i in range(D):
        blended_multires = []
        for j in range(levels):
            blended_multires.append(imgPyramids[i][j] * np.dstack([wPyramids[i][j], wPyramids[i][j], wPyramids[i][j]]))
        blendedPyramids.append(blended_multires)
    # 多图融合
    finalPyramid = []
    for i in range(levels):
        intermediate = []
        tmp = np.zeros_like(blendedPyramids[0][i])
        for j in range(D):
            tmp += np.array(blendedPyramids[j][i])
        intermediate.append(tmp)
        finalPyramid.append(intermediate)
    # 反向金字塔
    finalImage = []
    blended_final = np.array(finalPyramid[0][0])
    for i in range(levels - 1):
        imgH = np.shape(image_stack[0])[0]
        imgW = np.shape(image_stack[0])[1]
        layerx = cv.pyrUp(finalPyramid[i + 1][0])
        blended_final += cv.resize(layerx, (imgW, imgH))

    blended_final[blended_final < 0] = 0
    blended_final[blended_final > 255] = 255
    finalImage.append(blended_final)
    print(" *Done\n")

    return finalImage[0].astype('uint8')


def measures_fusion_simple(image_stack, max_value=255):
    print("Simple multi exp")
    D, H, W, C = np.shape(image_stack)
    sigma = 0.5
    gauss_curve = lambda i: np.exp(-((i - 0.5) ** 2) / (2 * sigma * sigma))

    nromal = round((D - 1) / 2)  # 四舍五入
    ycc_out = np.zeros([H, W, C])
    y_out = np.zeros([H, W])
    ycc_weight_sum = np.zeros([H, W])
    for i in range(D):
        RGB = image_stack[i]
        ycc = csc.rgb2ycbcr(RGB, W, H)
        y = ycc[:, :, 0]
        weight = gauss_curve(y / max_value)
        y_out = y_out + y * weight
        ycc_weight_sum = ycc_weight_sum + weight
        if i == nromal:  # 这里没看懂
            ycc_out[:, :, 1] = ycc[:, :, 1]
            ycc_out[:, :, 2] = ycc[:, :, 2]

    ycc_out[:, :, 0] = y_out / ycc_weight_sum
    rgb_out = csc.ycbcr2rgb(ycc_out, W, H)
    print(" *Done\n")

    return rgb_out


if __name__ == "__main__":
    path = "../pic/hdr"
    cwd = os.getcwd()  # 获取当前系统路径
    image_stack = load_images(path)  # 把要参数hdr融合的图片加入到栈中。
    image_stack = alignment(image_stack)  # 把要参数hdr融合的图片大小调整一致。image_stack[0].shape为(428, 642, 3)

    resultsPath = cwd + "/results"  # 不要把融合后的图像放到 path ，否则下次运行程序又会加到 image_stack 。
    print("resultsPath = ", resultsPath)
    if os.path.isdir(resultsPath):
        os.chdir(resultsPath)  # 改变当前的系统路径
    else:
        os.mkdir(resultsPath)
        os.chdir(resultsPath)

    "Compute Quality Measures"
    # ------------------------------------------------------------------------------#
    weight_map = scalar_weight_map(image_stack, weights=[1, 1, 1])  # weight_map.shape (428, 642, 4)

    "Naive Exposure Fusion"
    # ------------------------------------------------------------------------------#
    final_imageA, RijA = measures_fusion_naive(image_stack, weight_map)

    plt.figure(num='Naive Exposure Fusion')
    plt.imshow(final_imageA / 255)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()

    "Blurred Exposure Fusion"  # 模糊曝光融合
    # ------------------------------------------------------------------------------#
    final_imageB, RijB = measures_fusion_naive(image_stack, weight_map, blurType='gaussian', blurSize=(0, 0),
                                               blurSigma=15)

    plt.figure(num='Blurred Exposure Fusion')
    plt.imshow(final_imageB / 255)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()

    "Bilateral Exposure Fusion"
    # ------------------------------------------------------------------------------#
    final_imageC, RijC = measures_fusion_naive(image_stack, weight_map, blurType='bilateral', blurSize=(115, 115),
                                               blurSigma=51)

    plt.figure(num='Bilateral Exposure Fusion')
    plt.imshow(final_imageC / 255)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()

    "Multiresolution Exposure Fusion"  # 多分辨率曝光融合
    # ------------------------------------------------------------------------------#
    final_imageD = measures_fusion_multires(image_stack, weight_map, levels=6)

    plt.figure(num='Multiresolution Exposure Fusion')
    plt.imshow(final_imageD / 255)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()

    "simple Exposure Fusion"  # 简单曝光融合
    # ------------------------------------------------------------------------------#
    final_imageE = measures_fusion_simple(image_stack)

    plt.figure(num='simple Exposure Fusion')
    plt.imshow(final_imageE / 255)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()

    os.chdir(cwd)  # 重新切换到原来的系统路径
