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
   FUNCTION: alignmentent
        Call to Create Uniform Images by adjusting image sizes
     INPUTS:
        image_stack = stack of images from load_images
    OUTPUTS:
        images files of the same size
    """
    # '-----------------------------------------------------------------------------#
    sizes = []
    D = len(image_stack)
    for i in range(D):
        sizes.append(np.shape(image_stack[i]))
    sizes = np.array(sizes)
    for i in range(D):
        if np.shape(image_stack[i])[:2] != (min(sizes[:, 0]), min(sizes[:, 1])):
            print("Detected Non-Uniform Sized Image" + str(i) + " ... Resolving ...")
            image_stack[i] = cv.resize(image_stack[i], (min(sizes[:, 1]), min(sizes[:, 0])))
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
    image = cv.cvtColor(image.astype('uint8'), cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(image.astype('float64'), cv.CV_64F, ksize)
    C = cv.convertScaleAbs(laplacian)
    C = cv.medianBlur(C.astype('float32'), 5)
    return C.astype('float64')


def saturation(image):
    """
   FUNCTION: saturation
        Call to compute second quality measure - st.dev across RGB channels
     INPUTS:
        image = input image (colored)
    OUTPUTS:
        saturation measure
    """
    # '-----------------------------------------------------------------------------#
    S = np.std(image, 2)
    return S.astype('float64')


def exposedness(image, sigma=0.2):
    """
   FUNCTION: exposedness
        Call to compute third quality measure - exposure using a gaussian curve
     INPUTS:
        image = input image (colored)
        sigma = gaussian curve parameter
    OUTPUTS:
        exposedness measure
    """
    # '-----------------------------------------------------------------------------#
    image = cv.normalize(image, None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    gauss_curve = lambda i: np.exp(-((i - 0.5) ** 2) / (2 * sigma * sigma))
    R_gauss_curve = gauss_curve(image[:, :, 0])
    G_gauss_curve = gauss_curve(image[:, :, 1])
    B_gauss_curve = gauss_curve(image[:, :, 2])
    E = R_gauss_curve * G_gauss_curve * B_gauss_curve
    return E.astype('float64')


def scalar_weight_map(image_stack, weights=[1, 1, 1]):
    """
   FUNCTION: scalar_weight_map
        Call to forcefully "AND"-combine all quality measures defined 
     INPUTS:
        image_measures = stack of quality measures computed for image i 
        image_measures[contrast, saturation, exposedness]
        weights = weight for each quality measure : weights[wc, ws, we]
    OUTPUTS:
        scalar_weight_map for particular image
    """
    # '-----------------------------------------------------------------------------#
    H = np.shape(image_stack[0])[0]
    W = np.shape(image_stack[0])[1]
    D = len(image_stack)
    Wijk = np.zeros((H, W, D), dtype='float64')
    wc = weights[0]
    ws = weights[1]
    we = weights[2]
    print("Computing Weight Maps from Measures using: C=%1.1d, S=%1.1d, E=%1.1d" % (wc, ws, we))

    epsilon = 0.000005
    for i in range(D):
        C = contrast(image_stack[i])
        S = saturation(image_stack[i])
        E = exposedness(image_stack[i])
        Wijk[:, :, i] = (np.power(C, wc) * np.power(S, ws) * np.power(E, we)) + epsilon
    normalizer = np.sum(Wijk, 2)

    for i in range(D):
        Wijk[:, :, i] = np.divide(Wijk[:, :, i], normalizer)
    print(" *Done")
    print("\n")

    return Wijk.astype('float64')


def measures_fusion_naive(image_stack, weight_maps, blurType=None, blurSize=(0, 0), blurSigma=15):
    """
   FUNCTION: measures_fusion_naive
        Call to fuse normalized weightmaps and their images
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weight_maps = scalar_weight_map for N images
        blurType    = gaussian or bilateral filter applied to weight-map
        blurSize/Sigma = blurring parameters
    OUTPUTS:
        img_fused = single image with fusion of measures
        Rij = fusion of individual images with their weight maps
    """
    # '-----------------------------------------------------------------------------#
    H = np.shape(image_stack[0])[0]
    W = np.shape(image_stack[0])[1]
    D = len(image_stack)
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
            Rijk = image_stack[i] * np.dstack([weight_maps[:, :, i], weight_maps[:, :, i], weight_maps[:, :, i]])
            Rij.append(Rijk)
            img_fused += Rijk

    print(" *Done")
    print("\n")

    return img_fused, Rij


def multires_pyramid(image, weight_map, levels):
    """
   FUNCTION: multires_pyramid
        Call to compute image and weights pyramids
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weight_maps = scalar_weight_map for N images
        levels = height of pyramid to use including base pyramid base
    OUTPUTS:
        imgLpyr = list containing image laplacian pyramid
        wGpyr   = list containing weight gaussian pyramid
    """
    # '-----------------------------------------------------------------------------#
    levels = levels - 1
    imgGpyr = [image]
    wGpyr = [weight_map]

    for i in range(levels):
        imgW = np.shape(imgGpyr[i])[1]
        imgH = np.shape(imgGpyr[i])[0]
        imgGpyr.append(cv.pyrDown(imgGpyr[i].astype('float64')))

    for i in range(levels):
        imgW = np.shape(wGpyr[i])[1]
        imgH = np.shape(wGpyr[i])[0]
        wGpyr.append(cv.pyrDown(wGpyr[i].astype('float64')))

    imgLpyr = [imgGpyr[levels]]
    wLpyr = [wGpyr[levels]]

    for i in range(levels, 0, -1):
        imgW = np.shape(imgGpyr[i - 1])[1]
        imgH = np.shape(imgGpyr[i - 1])[0]
        imgLpyr.append(imgGpyr[i - 1] - cv.resize(cv.pyrUp(imgGpyr[i]), (imgW, imgH)))

    for i in range(levels, 0, -1):
        imgW = np.shape(wGpyr[i - 1])[1]
        imgH = np.shape(wGpyr[i - 1])[0]
        wLpyr.append(wGpyr[i - 1] - cv.resize(cv.pyrUp(wGpyr[i]), (imgW, imgH)))

    return imgLpyr[::-1], wGpyr


def measures_fusion_multires(image_stack, weight_maps, levels=6):
    """
   FUNCTION: measures_fusion_multires
        Call to perform multiresolution blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        levels = desired height of the pyramids
        weight_maps = scalar_weight_map for N images
    OUTPUTS:
        finalImage = single exposure fused image
    """
    # '-----------------------------------------------------------------------------#
    print("Performing Multiresolution Blending using: " + str(levels) + " Pyramid levels")
    D = np.shape(image_stack)[0]

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
    print(" *Done")
    print("\n")

    return finalImage[0].astype('uint8')


def measures_fusion_simple(image_stack, max_value=255):
    print("Simple multi exp")
    D, H, W, C = np.shape(image_stack)
    sigma = 0.5
    gauss_curve = lambda i: np.exp(-((i - 0.5) ** 2) / (2 * sigma * sigma))

    nromal = round((D - 1) / 2)
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
        if i == nromal:
            ycc_out[:, :, 1] = ycc[:, :, 1]
            ycc_out[:, :, 2] = ycc[:, :, 2]

    ycc_out[:, :, 0] = y_out / ycc_weight_sum
    rgb_out = csc.ycbcr2rgb(ycc_out, W, H)
    print(" *Done\n")

    return rgb_out


def meanImage(image_stack, save=False):
    """
   FUNCTION: meanImage
        Call to perform mean image blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        save = save figures to directory
    OUTPUTS:
        mean of all the images in the stack
    """
    # '-----------------------------------------------------------------------------#
    N = len(image_stack)
    H = np.shape(image_stack[0])[0]
    W = np.shape(image_stack[0])[1]
    rr = np.zeros((H, W), dtype='float64')
    gg = np.zeros((H, W), dtype='float64')
    bb = np.zeros((H, W), dtype='float64')
    for i in range(N):
        r, g, b = cv.split(image_stack[i].astype('float64'))
        rr += r.astype('float64')
        gg += g.astype('float64')
        bb += b.astype('float64')
    MeanImage = np.dstack([rr / N, gg / N, bb / N]).astype('uint8')
    if save:
        cv.imwrite('img_MeanImage.png', MeanImage)
    return MeanImage


def visualize_maps(image_stack, weights=[1, 1, 1], save=False):
    """
   FUNCTION: measures_fusion_multires
        Call to perform multiresolution blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weights = importance factor for each measure C,S,E
        save = save figures to directory
    OUTPUTS:
        images of contrast, saturation, exposure, and combined weight for image N
    """
    # '-----------------------------------------------------------------------------#
    for N in range(len(image_stack)):
        img_contrast = contrast(image_stack[N])
        img_saturation = saturation(image_stack[N])
        img_exposedness = exposedness(image_stack[N])
        # weight_map      = scalar_weight_map([image_stack[N]], weights)
        print("Displaying Measures and Weight Map for Image_stack[" + str(N) + "]")

        if save:
            plt.imsave('img_contrast' + str(N) + '.png', img_contrast, cmap='gray', dpi=1800)
            plt.imsave('img_saturation' + str(N) + '.png', img_saturation, cmap='gray', dpi=1800)
            plt.imsave('img_exposedness' + str(N) + '.png', img_exposedness, cmap='gray', dpi=1800)
        else:
            plt.figure(1)
            plt.imshow(img_contrast.astype('float'), cmap='gray')
            plt.figure(2)
            plt.imshow(img_saturation, cmap='gray')
            plt.figure(3)
            plt.imshow(img_exposedness, cmap='gray')
            # plt.figure(4)
            # plt.imshow(weight_map[:,:,0],cmap='gray')  # .astype('uint8')
    print(" *Done\n")


if __name__ == "__main__":
    path = "../pic/hdr"

    cwd = os.getcwd()  # 获取当前系统路径
    print("cwd = ", cwd)
    print("path = ", path)
    image_stack = load_images(path)
    image_stack = alignment(image_stack)
    # resultsPath = path+"\\results"
    resultsPath = cwd + "results"
    if os.path.isdir(resultsPath):
        os.chdir(resultsPath)
    else:
        os.mkdir(resultsPath)
        os.chdir(resultsPath)

    "Compute Quality Measures"
    # ------------------------------------------------------------------------------#
    # Compute Quality measures multiplied and weighted with weights[x,y,z]
    weight_map = scalar_weight_map(image_stack, weights=[1, 1, 1])
    # weight_map      = scalar_weight_map(image_stack, weights = [0,0,0]) #Performs Pyramid Fusion

    "Original Image"
    # ------------------------------------------------------------------------------#
    # load original image i.e center image probably has the median Exposure value(EV)
    # filename = os.listdir(path)[len(os.listdir(path))/2]
    # original_image = cv.imread(os.path.join(path, filename), cv.IMREAD_COLOR)
    # cv.imshow('Original Image', original_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imwrite('img_CenterOriginal.png', original_image.astype('uint8'))

    "Naive Exposure Fusion"
    # ------------------------------------------------------------------------------#
    final_imageA, RijA = measures_fusion_naive(image_stack, weight_map)

    plt.figure(2)
    plt.imshow(final_imageA / 255)
    plt.show()

    "Blurred Exposure Fusion"
    # ------------------------------------------------------------------------------#
    final_imageB, RijB = measures_fusion_naive(image_stack, weight_map, blurType='gaussian', blurSize=(0, 0),
                                               blurSigma=15)

    plt.figure(2)
    plt.imshow(final_imageB / 255)
    plt.show()

    "Bilateral Exposure Fusion"
    # ------------------------------------------------------------------------------#
    final_imageC, RijC = measures_fusion_naive(image_stack, weight_map, blurType='bilateral', blurSize=(115, 115),
                                               blurSigma=51)

    plt.figure(2)
    plt.imshow(final_imageC / 255)
    plt.show()

    "Multiresolution Exposure Fusion"
    # ------------------------------------------------------------------------------#
    final_imageD = measures_fusion_multires(image_stack, weight_map, levels=6)

    plt.figure(1)
    plt.imshow(final_imageD)
    plt.show()

    "simple Exposure Fusion"
    # ------------------------------------------------------------------------------#
    final_imageE = measures_fusion_simple(image_stack)

    plt.figure(2)
    plt.imshow(final_imageE / 255)
    plt.show()

    "Display Intermediate Steps and Save"
    # ------------------------------------------------------------------------------#
    # visualize_maps(image_stack, save=False)

    "Compute Mean of Image Stack"
    # ------------------------------------------------------------------------------#
    # final_imageE = meanImage(image_stack, save=False)
    os.chdir(cwd)
