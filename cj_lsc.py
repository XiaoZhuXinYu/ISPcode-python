import numpy as np
from matplotlib import pyplot as plt
import raw_image
import cv2
import cj_rawimage


def apply_shading_to_image(img, block_size, shading_R, shading_GR, shading_GB, shading_B, pattern, ratio):
    # 用G做luma
    luma_shading = (shading_GR + shading_GB) / 2
    # 计算color shading
    R_color_shading = shading_R / luma_shading
    GR_color_shading = shading_GR / luma_shading
    GB_color_shading = shading_GB / luma_shading
    B_color_shading = shading_B / luma_shading
    # 计算调整之后luma shading
    new_luma_shading = (luma_shading - 1) * ratio + 1
    # 合并两种shading
    new_shading_R = R_color_shading * new_luma_shading
    new_shading_GR = GR_color_shading * new_luma_shading
    new_shading_GB = GB_color_shading * new_luma_shading
    new_shading_B = B_color_shading * new_luma_shading

    R, GR, GB, B = raw_image.bayer_channel_separation(img, pattern)
    HH, HW = R.shape
    size_new = (HW + block_size, HH + block_size)
    # 插值的方法的选择
    ex_R_gain_map = cv2.resize(new_shading_R, size_new, interpolation=cv2.INTER_CUBIC)
    ex_GR_gain_map = cv2.resize(new_shading_GR, size_new, interpolation=cv2.INTER_CUBIC)
    ex_GB_gain_map = cv2.resize(new_shading_GB, size_new, interpolation=cv2.INTER_CUBIC)
    ex_B_gain_map = cv2.resize(new_shading_B, size_new, interpolation=cv2.INTER_CUBIC)
    # 裁剪到原图大小
    half_b_size = int(block_size / 2)
    R_gain_map = ex_R_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]
    GR_gain_map = ex_GR_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]
    GB_gain_map = ex_GB_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]
    B_gain_map = ex_B_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]

    R_new = R * R_gain_map
    GR_new = GR * GR_gain_map
    GB_new = GB * GB_gain_map
    B_new = B * B_gain_map

    new_image = raw_image.bayer_channel_integration(R_new, GR_new, GB_new, B_new, pattern)
    # 值缩减到0~1023
    new_image = np.clip(new_image, a_min=0, a_max=1023)
    return new_image


def create_lsc_data(img, block_size, pattern):
    # 分开四个颜色通道
    R, GR, GB, B = raw_image.bayer_channel_separation(img, pattern)
    # print(img.shape, R.shape)

    # 每张的高宽
    HH, HW = R.shape

    # 生成分多少块
    Hblocks = int(HH / block_size)
    Wblocks = int(HW / block_size)

    # 整个图像被分成 Hblocks * Wblocks 块，生成一个 Hblocks * Wblocks 的矩阵，用于数据储存。
    R_LSC_data = np.zeros((Hblocks, Wblocks))  # 每个块 R 的平均值
    B_LSC_data = np.zeros((Hblocks, Wblocks))
    GR_LSC_data = np.zeros((Hblocks, Wblocks))
    GB_LSC_data = np.zeros((Hblocks, Wblocks))

    # 块距离光心的距离
    RA = np.zeros((Hblocks, Wblocks))

    # 计算每个块的平均值
    for y in range(0, HH, block_size):
        for x in range(0, HW, block_size):
            block_y_num = int(y / block_size)
            block_x_num = int(x / block_size)
            R_LSC_data[block_y_num, block_x_num] = R[y:y + block_size, x:x + block_size].mean()  # 对一块block数据求平均
            GR_LSC_data[block_y_num, block_x_num] = GR[y:y + block_size, x:x + block_size].mean()
            GB_LSC_data[block_y_num, block_x_num] = GB[y:y + block_size, x:x + block_size].mean()
            B_LSC_data[block_y_num, block_x_num] = B[y:y + block_size, x:x + block_size].mean()

    # 根据GR的最大值，寻找真正的光心块
    center_point = np.where(GR_LSC_data == np.max(GR_LSC_data))
    center_y = center_point[0] * block_size + block_size / 2
    center_x = center_point[1] * block_size + block_size / 2

    # 计算块距离光心的距离
    for y in range(0, HH, block_size):
        for x in range(0, HW, block_size):
            xx = x + block_size / 2
            yy = y + block_size / 2
            block_y_num = int(y / block_size)
            block_x_num = int(x / block_size)
            RA[block_y_num, block_x_num] = (yy - center_y) * (yy - center_y) + (xx - center_x) * (xx - center_x)

    # 4个颜色数据通道展平，便于数据进行拟合，RA_flatten相当于x，R_LSC_data_flatten和其他三个相当于y
    RA_flatten = RA.flatten()
    R_LSC_data_flatten = R_LSC_data.flatten()
    GR_LSC_data_flatten = GR_LSC_data.flatten()
    GB_LSC_data_flatten = GB_LSC_data.flatten()
    B_LSC_data_flatten = B_LSC_data.flatten()

    # 最亮块的值
    Max_R = np.max(R_LSC_data_flatten)
    Max_GR = np.max(GR_LSC_data_flatten)
    Max_GB = np.max(GB_LSC_data_flatten)
    Max_B = np.max(B_LSC_data_flatten)

    # 得到gain,还没有外插
    G_R_LSC_data = Max_R / R_LSC_data_flatten
    G_GR_LSC_data = Max_GR / GR_LSC_data_flatten
    G_GB_LSC_data = Max_GB / GB_LSC_data_flatten
    G_B_LSC_data = Max_B / B_LSC_data_flatten

    # gain
    plt.scatter(RA_flatten, G_R_LSC_data, color='read')
    plt.scatter(RA_flatten, G_GR_LSC_data, color='green')
    plt.scatter(RA_flatten, G_GB_LSC_data, color='green')
    plt.scatter(RA_flatten, G_B_LSC_data, color='blue')
    plt.show()

    # 进行gain曲线拟合拟合
    par_R = np.polyfit(RA_flatten, G_R_LSC_data, 3)
    par_GR = np.polyfit(RA_flatten, G_GR_LSC_data, 3)
    par_GB = np.polyfit(RA_flatten, G_GB_LSC_data, 3)
    par_B = np.polyfit(RA_flatten, G_B_LSC_data, 3)

    # 拟合之后生成所有点的值
    ES_R = par_R[0] * (RA_flatten ** 3) + par_R[1] * (RA_flatten ** 2) + par_R[2] * RA_flatten + par_R[3]
    ES_GR = par_GR[0] * (RA_flatten ** 3) + par_GR[1] * (RA_flatten ** 2) + par_GR[2] * RA_flatten + par_GR[3]
    ES_GB = par_GB[0] * (RA_flatten ** 3) + par_GB[1] * (RA_flatten ** 2) + par_GB[2] * RA_flatten + par_GB[3]
    ES_B = par_B[0] * (RA_flatten ** 3) + par_B[1] * (RA_flatten ** 2) + par_B[2] * RA_flatten + par_B[3]
    # 拟合数据和原有数据有什么不同
    plt.scatter(RA_flatten, ES_R, color='red')
    plt.scatter(RA_flatten, ES_GR, color='green')
    plt.scatter(RA_flatten, ES_GB, color='green')
    plt.scatter(RA_flatten, ES_B, color='blue')
    plt.show()

    # 外插补偿的gain通过曲线拟合得到，这边进行外插主要是考虑有些图像的分辨率不能被block整除
    EX_RA = np.zeros((Hblocks + 2, Wblocks + 2))
    EX_R = np.zeros((Hblocks + 2, Wblocks + 2))
    EX_GR = np.zeros((Hblocks + 2, Wblocks + 2))
    EX_GB = np.zeros((Hblocks + 2, Wblocks + 2))
    EX_B = np.zeros((Hblocks + 2, Wblocks + 2))
    new_center_y = center_point[0] + 1
    new_center_x = center_point[1] + 1
    for y in range(0, Hblocks + 2):
        for x in range(0, Wblocks + 2):
            EX_RA[y, x] = (y - new_center_y) * block_size * (y - new_center_y) * block_size + (
                    x - new_center_x) * block_size * (x - new_center_x) * block_size
            EX_R[y, x] = par_R[0] * (EX_RA[y, x] ** 3) + par_R[1] * (EX_RA[y, x] ** 2) + par_R[2] * (EX_RA[y, x]) + \
                         par_R[3]

            EX_GR[y, x] = par_GR[0] * (EX_RA[y, x] ** 3) + par_GR[1] * (EX_RA[y, x] ** 2) + par_GR[2] * (EX_RA[y, x]) + \
                          par_GR[3]

            EX_GB[y, x] = par_GB[0] * (EX_RA[y, x] ** 3) + par_GB[1] * (EX_RA[y, x] ** 2) + par_GB[2] * (EX_RA[y, x]) + \
                          par_GB[3]

            EX_B[y, x] = par_B[0] * (EX_RA[y, x] ** 3) + par_B[1] * (EX_RA[y, x] ** 2) + par_B[2] * (EX_RA[y, x]) + \
                         par_B[3]

    # 中心用实际采样的数据
    EX_R[1:1 + Hblocks, 1:1 + Wblocks] = G_R_LSC_data
    EX_GR[1:1 + Hblocks, 1:1 + Wblocks] = G_GR_LSC_data
    EX_GB[1:1 + Hblocks, 1:1 + Wblocks] = G_GB_LSC_data
    EX_B[1:1 + Hblocks, 1:1 + Wblocks] = G_B_LSC_data

    return EX_R, EX_GR, EX_GB, EX_B


if __name__ == "__main__":
    img = cj_rawimage.read_plained_file("../pic/D65_4032_2752_GRBG_2_BLC.raw", dtype="uint16", width=4032, height=2752,
                                        shift_bits=0)
    block_size = 16
    pattern = "GRBG"
    shading_R, shading_GR, shading_GB, shading_B = create_lsc_data(img, block_size, pattern)
    img2 = cj_rawimage.read_plained_file("../pic/D65_4032_2752_GRBG_1_BLC.raw", dtype="uint16", width=4032, height=2752,
                                         shift_bits=0)
    cj_rawimage.show_planedraw(img2, width=4032, height=2752, pattern="MONO", sensorbit=10, compress_ratio=1)

    # luma 和color shading
    new_image = apply_shading_to_image(img=img2, block_size=block_size, shading_R=shading_R, shading_GR=shading_GR,
                                       shading_GB=shading_GB, shading_B=shading_B, pattern="GRBG", ratio=1)

    print(np.min(new_image), np.max(new_image))
    # cj_rawimage.show_planedraw(new_image, width=4032, height=2752, pattern=pattern, sensorbit=10, compress_ratio=1)
    cj_rawimage.show_planedraw(new_image, width=4032, height=2752, pattern="MONO", sensorbit=10, compress_ratio=1)
