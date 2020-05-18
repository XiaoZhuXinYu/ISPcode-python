import numpy as np


# 每个channel减去一个值
def blc_process(img1, width, height, dtype, pattern, sensorbit):
    image = np.fromfile(img1, dtype)  # 相比于open，该函数可以指定数据类型
    image = image[0:width * height]  # 这句是为了防止图像中有无效数据
    image.shape = [height, width]  # 将数组转化为二维矩阵

    if pattern == "MONO":
        blc = int(image.mean())
        print("MONO", blc)
    elif pattern == "RGGB":
        R = image[::2, ::2]
        GR = image[::2, 1::2]
        GB = image[1::2, ::2]
        B = image[1::2, 1::2]
        R = int(R.mean())
        GR = int(GR.mean())
        GB = int(GB.mean())
        B = int(B.mean())
        print("RGGB", R, GR, GB, B)
    elif pattern == "GRBG":
        GR = image[::2, ::2]
        R = image[::2, 1::2]
        B = image[1::2, ::2]
        GB = image[1::2, 1::2]
        GR = int(GR.mean())
        R = int(R.mean())
        B = int(B.mean())
        GB = int(GB.mean())
        print("GRBG", GR, R, B, GB)
    elif pattern == "GBRG":
        GB = image[::2, ::2]
        B = image[::2, 1::2]
        R = image[1::2, ::2]
        GR = image[1::2, 1::2]
        GB = int(GB.mean())
        B = int(B.mean())
        R = int(R.mean())
        GR = int(GR.mean())
        print("GBRG", GB, B, R, GR)
    elif pattern == "BGGR":
        B = image[::2, ::2]
        GB = image[::2, 1::2]
        GR = image[1::2, ::2]
        R = image[1::2, 1::2]
        B = int(B.mean())
        GR = int(GR.mean())
        GB = int(GB.mean())
        R = int(R.mean())
        print("BGGR", B, GB, GR, R)
    else:
        print("please input a correct pattern")


if __name__ == "__main__":
    print('This is main of module')
    file_name1 = "BLC.raw"
    blc_process(file_name1, 1920, 1080, dtype="uint16", pattern='RGGB', sensorbit=12)
