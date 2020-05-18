import numpy as np
import math
from matplotlib import pyplot as plt


def read_plained_file(file_path_name, dtype, width, height,  shift_bits):
    frame = np.fromfile(file_path_name, dtype=dtype)  # 相比于open，该函数可以指定数据类型
    # print("b shape",b.shape)
    # print('%#x'%b[0])
    frame = frame[0:height * width]  # 这句是为了防止图像中有无效数据
    frame.shape = [height, width]  # 将数组转化为二维矩阵
    frame = np.right_shift(frame, shift_bits)  # 考虑数据大小端问题，进行移位
    return frame


def write_plained_file(file_path_name, image, dtype):
    if dtype == "uint8":
        image = image.astype(np.uint8)
    elif dtype == "uint16":
        image = image.astype(np.uint16)
    frame = image.tofile(file_path_name)
    return


# 非弱拷贝分离,没带颜色通道
def simple_separation(image):
    C1 = image[::2, ::2]
    C2 = image[::2, 1::2]
    C3 = image[1::2, 0::2]
    C4 = image[1::2, 1::2]
    return C1, C2, C3, C4


# 带颜色通道的分离
def bayer_channel_separation(data, pattern):
    image_data = data
    if pattern == "RGGB":
        R = image_data[::2, ::2]
        GR = image_data[::2, 1::2]
        GB = image_data[1::2, ::2]
        B = image_data[1::2, 1::2]
    elif pattern == "GRBG":
        GR = image_data[::2, ::2]
        R = image_data[::2, 1::2]
        B = image_data[1::2, ::2]
        GB = image_data[1::2, 1::2]
    elif pattern == "GBRG":
        GB = image_data[::2, ::2]
        B = image_data[::2, 1::2]
        R = image_data[1::2, ::2]
        GR = image_data[1::2, 1::2]
    elif pattern == "BGGR":
        B = image_data[::2, ::2]
        GB = image_data[::2, 1::2]
        GR = image_data[1::2, ::2]
        R = image_data[1::2, 1::2]
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return

    return R, GR, GB, B


# 合成
def simple_integration(C1, C2, C3, C4):
    size = np.shape(C1)
    data = np.empty((size[0] * 2, size[1] * 2), dtype=np.float32)
    data[::2, ::2] = C1
    data[::2, 1::2] = C2
    data[1::2, ::2] = C3
    data[1::2, 1::2] = C4
    return data


# 按照颜色根据不同的bayer合成
def bayer_channel_integration(R, GR, GB, B, pattern):
    size = np.shape(R)
    data = np.empty((size[0] * 2, size[1] * 2), dtype=np.float32)
    # casually use float32,maybe change later
    if pattern == "RGGB":
        data[::2, ::2] = R
        data[::2, 1::2] = GR
        data[1::2, ::2] = GB
        data[1::2, 1::2] = B
    elif pattern == "GRBG":
        data[::2, ::2] = GR
        data[::2, 1::2] = R
        data[1::2, ::2] = B
        data[1::2, 1::2] = GB
    elif pattern == "GBRG":
        data[::2, ::2] = GB
        data[::2, 1::2] = B
        data[1::2, ::2] = R
        data[1::2, 1::2] = GR
    elif pattern == "bggr":
        data[::2, ::2] = B
        data[::2, 1::2] = GB
        data[1::2, ::2] = GR
        data[1::2, 1::2] = R
    else:
        print("pattern must be one of these: rggb, grbg, gbrg, bggr")
        return

    return data


def raw_image_show_gray(image,  width, height, compress_ratio=1):
    x = width/(compress_ratio * 100)
    y = height/(compress_ratio * 100)
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(image, cmap='gray', interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    print('show gray image')


def raw_image_show_fakecolor(image,  width, height, pattern, compress_ratio=1):  # 显示伪彩色
    x = width / (compress_ratio * 100)
    y = height / (compress_ratio * 100)
    rgb_img = np.zeros(shape=(height, width, 3))

    R = rgb_img[:, :, 0]
    GR = rgb_img[:, :, 1]
    GB = rgb_img[:, :, 1]  # 此处为弱拷贝，后续对GR的操作也是对GB的操作
    B = rgb_img[:, :, 2]

    # R, GR, GB, B = bayer_channel_separation(image, pattern)  # 未测试过是否可以替代下面的判断语句

    if pattern == "RGGB":
        R[::2, ::2] = image[::2, ::2]
        GR[::2, 1::2] = image[::2, 1::2]
        GB[1::2, ::2] = image[1::2, ::2]
        B[1::2, 1::2] = image[1::2, 1::2]
    elif pattern == "GRBG":
        GR[::2, ::2] = image[::2, ::2]
        R[::2, 1::2] = image[::2, 1::2]
        B[1::2, ::2] = image[1::2, ::2]
        GB[1::2, 1::2] = image[1::2, 1::2]
    elif pattern == "GBRG":
        GB[::2, ::2] = image[::2, ::2]
        B[::2, 1::2] = image[::2, 1::2]
        R[1::2, ::2] = image[1::2, ::2]
        GR[1::2, 1::2] = image[1::2, 1::2]
    elif pattern == "BGGR":
        B[::2, ::2] = image[::2, ::2]
        GB[::2, 1::2] = image[::2, 1::2]
        GR[1::2, ::2] = image[1::2, ::2]
        R[1::2, 1::2] = image[1::2, 1::2]
    else:
        print("show failed")
        return
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(rgb_img, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()
    print('show fake color image')


def show_planedraw(image1,  width, height, pattern, sensorbit, compress_ratio=1):
    # image = read_plained_file(file_path_name, dtype, width, height,  shift_bits)
    image = image1
    if sensorbit == 8:
        image = image / 255  # 8bit sensor 所以是除255，为了和下面函数中 vmax=1进行配合
    elif sensorbit == 10:
        image = image / 1023  # 10bit sensor 所以是除1023，为了和下面函数中 vmax=1进行配合
    elif sensorbit == 12:
        image = image / 4095  # 12bit sensor 所以是除4095，为了和下面函数中 vmax=1进行配合
    else:
        image = image / 4095  # 12bit sensor 所以是除4095，为了和下面函数中 vmax=1进行配合

    if pattern == 'MONO':
        raw_image_show_gray(image, width, height, compress_ratio)
    else:
        raw_image_show_fakecolor(image, width, height, pattern, compress_ratio)


def show_mipiraw_mipi10(image1,  width1, height1, pattern, dtype, sensorbit, compress_ratio=1):
    file_name = image1
    width = width1
    height = height1
    # 单行长度的补齐
    new_width = int(math.floor((width + 3) / 4) * 4)  # 像素对四字节补齐
    packet_num_L = int(new_width / 4)
    width_byte_num = packet_num_L * 5  # 单行byte长度
    width_byte_num = int(math.floor((width_byte_num + 7) / 8) * 8)  # 单行做8个字节补齐
    image_bytes = width_byte_num*height  # 读取特定长度是为了防止无效数据
    frame = np.fromfile(file_name, count=image_bytes, dtype="uint8")
    print("b shape", frame.shape)
    print('%#x'%frame[0])
    frame.shape = [height, width_byte_num]  # 按字节整理的图像矩阵
    one_byte = frame[:, 0:image_bytes:5]  # 每个包的第一个像素
    two_byte = frame[:, 1:image_bytes:5]  # 每个包的第二个像素
    three_byte = frame[:, 2:image_bytes:5]
    four_byte = frame[:, 3:image_bytes:5]
    five_byte = frame[:, 4:image_bytes:5]
    # 数据转换防止溢出
    one_byte = one_byte.astype('uint16')
    two_byte = two_byte.astype('uint16')
    three_byte = three_byte.astype('uint16')
    four_byte = four_byte.astype('uint16')
    five_byte = five_byte.astype('uint16')
    # 用矩阵的方法进行像素的拼接
    one_byte = np.left_shift(one_byte, 2) + np.bitwise_and(five_byte, 3)
    two_byte = np.left_shift(two_byte, 2) + np.right_shift(np.bitwise_and(five_byte, 12), 2)
    three_byte = np.left_shift(three_byte, 2) + np.right_shift(np.bitwise_and(five_byte, 48), 4)
    four_byte = np.left_shift(four_byte, 2) + np.right_shift(np.bitwise_and(five_byte, 192), 6)
    # 重组帧
    frame_pixels = np.zeros(shape=(height, new_width))
    frame_pixels[:, 0: new_width:4] = one_byte[:, 0: packet_num_L]
    frame_pixels[:, 1: new_width:4] = two_byte[:, 0: packet_num_L]
    frame_pixels[:, 2: new_width:4] = three_byte[:, 0: packet_num_L]
    frame_pixels[:, 3: new_width:4] = four_byte[:, 0: packet_num_L]
    # 裁剪无用的数据
    frame_out = frame_pixels[:, 0:width]

    frame_out = frame_out / 1023
    raw_image_show_gray(frame_out, 4032, 3016)
    return 0


if __name__ == "__main__":
    print('This is main of module')
    file_name1 = "1.raw"
    show_planedraw(file_name1, 640, 480, 'BGGR', dtype="uint8", sensorbit=8, compress_ratio=1)
    # show_mipiraw_mipi10()
