from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cj_rawimage


# 根据边进行处理, 推荐使用这种方法
def mono_DPC_extreme(img, thred):
    def DPC_process(P, thred):  # 函数中可以做很多卷积做不了的操作
        P2 = P[[0, 1, 2, 3, 5, 6, 7, 8]]
        bad_pixel = False
        max_value = np.max(P2)
        min_value = np.min(P2)

        # 当比最大值大的时候和比最小值小的时候
        if (P[4] - max_value) > (thred * max_value):  # 这样就已经包含了判断 P[4] > max_value
            bad_pixel = True
        elif (min_value - P[4]) > (thred * min_value):  # 这样就已经包含了判断 min_value - P[4]
            bad_pixel = True
        else:
            return P[4]

        if bad_pixel:
            # 计算不同方向边上的梯度,横竖两个对角
            dv = abs(2 * P[4] - P[1] - P[7])
            dh = abs(2 * P[4] - P[3] - P[5])
            ddl = abs(2 * P[4] - P[0] - P[8])
            ddr = abs(2 * P[4] - P[2] - P[6])

            # 梯度最小的是对应的边,取对应边的平均值
            if min(dv, dh, ddl, ddr) == dv:
                value = (P[1] + P[7]) / 2
            elif min(dv, dh, ddl, ddr) == dh:
                value = (P[3] + P[5]) / 2
            elif min(dv, dh, ddl, ddr) == ddl:
                value = (P[0] + P[8]) / 2
            else:
                value = (P[2] + P[6]) / 2
            return value

    footprint = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 3X3区域做滤波
    result = ndimage.generic_filter(img, DPC_process, footprint=footprint, extra_arguments=(thred,))
    return result


# 根据简单的梯度进行处理
def mono_DPC_gradient(img, thred):
    def DPC_process(P, thred):  # 函数中可以做很多卷积做不了的操作
        P2 = P[[0, 1, 2, 3, 5, 6, 7, 8]]
        different_value = np.abs(P2 - P[4])
        compare = (different_value / P2) > thred
        number = np.count_nonzero(compare)
        global a
        if number == 8:
            dv = abs(2 * P[4] - P[1] - P[7])
            dh = abs(2 * P[4] - P[3] - P[5])
            ddl = abs(2 * P[4] - P[0] - P[8])
            ddr = abs(2 * P[4] - P[2] - P[6])
            a = a + 1
            # print(a)
            if min(dv, dh, ddl, ddr) == dv:
                value = (P[1] + P[7]) / 2
            elif min(dv, dh, ddl, ddr) == dh:
                value = (P[3] + P[5]) / 2
            elif min(dv, dh, ddl, ddr) == ddl:
                value = (P[0] + P[8]) / 2
            else:
                value = (P[2] + P[6]) / 2
            return value
        return P[4]

    footprint = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 3X3区域做滤波
    result = ndimage.generic_filter(img, DPC_process, footprint=footprint, extra_arguments=(thred,))
    return result


# 根据均值进行处理
def mono_DPC_mean(img, thred):
    # 子函数开始
    def DPC_process(P, thred):  # 函数中可以做很多卷积做不了的操作
        P2 = P[[0, 1, 2, 3, 5, 6, 7, 8]]
        different_value = np.abs(P2 - P[4])
        compare = different_value > thred
        number = np.count_nonzero(compare)
        i = 0
        if number == 8:
            print("bad")
            return np.mean(P2)
        return P[4]

    # 子函数结束
    footprint = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 3X3区域做滤波
    result = ndimage.generic_filter(img, DPC_process, footprint=footprint, extra_arguments=(thred,))
    return result


# DPC主函数入口
def DPC(img, thre, mode, pattern):
    if mode == "mean":
        mono_DPC = mono_DPC_mean
    elif mode == "gradient":
        mono_DPC = mono_DPC_gradient
    elif mode == "extreme":
        mono_DPC = mono_DPC_extreme
    else:
        print("Error mode")

    if pattern == 'MONO':
        result = mono_DPC(img, thre)
    else:
        C1, C2, C3, C4 = cj_rawimage.simple_separation(img)
        # print(C1)
        C1 = mono_DPC(C1, thre)
        C2 = mono_DPC(C2, thre)
        C3 = mono_DPC(C3, thre)
        C4 = mono_DPC(C4, thre)

        result = cj_rawimage.simple_integration(C1, C2, C3, C4)
    return result


def test_DPC():
    img = cj_rawimage.read_plained_file("DSC16_1339_768x512_rggb_wait_dpc.raw", dtype="uint16", width=768, height=512,
                                        shift_bits=0)
    global a
    a = 0
    thre_ratio = 1
    # thre_ratio = 300   # MEAN
    result = DPC(img, thre_ratio, mode="extreme", pattern="RGGB")
    cj_rawimage.show_planedraw(result, 768, 512, pattern='MONO', sensorbit=10, compress_ratio=1)


if __name__ == "__main__":
    print('This is main of module')
    test_DPC()

