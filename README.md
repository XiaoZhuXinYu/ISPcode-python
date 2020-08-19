# ISPcode-python
isp参考代码，通过python实现

# 20200518
项目说明：这是一个本人自己编写的ispcode，通过python实现，本人是一个python和ispcode的初学者。
很多内容参考了张兴老师图像算法基础课程的代码。本项目主要是自己学习的一个积累，之后会慢慢完善。
如果你觉得对你有帮助，欢迎下载。如果代码中有错误的地方，也欢迎指正。之后我还会出C语言实现的ispcode。

# 20200518
完善了cj_rawimage相关功能，添加了cj_dpc cj_filter 两个文件。

# 20200519
1. cj_curve_fit 增加多项式拟合方式np.polyfit
2. 添加cj_lsc 文件

# 20200519 23::56
1. 优化了cj_lsc 文件

# 20200520 15::55
1. 解决了两个编译错误问题
2. 由于没想到更好的方法，暂时将能整除和不能整除的分开来写了。

# 20200522 18::44
1. cj_bmpimage 优化了直方图统计的算法，支持区域统计。

# 20200524 23::16
1. cj_bmpimage 优化了直方图统计的算法，支持任意指定直方图统计区间的长度。

# 20200528 18::25
1. cj_bmpimage 优化了直方图统计的算法，添加打印数据统计信息等，x，y轴坐标支持中文显示。
read_bmpimage show_bmpimage 函数分开写。
2. cj_curve_fit 添加描点函数。
3. cj_rawimage read_plained_file show_planedraw 函数分开写。
4. cj_histogram 删除一些冗余函数。
5. cj_2dnr 添加该文件

# 20200529 18::51
1. cj_2dnr 添加 bilateral_filter1 函数。只支持3*3矩阵。优化了处理速度。

# 20200608 13::06
1. 添加了gitignore配置文件


# 20200805 20:05
1. 添加了awb和gamma两个文件
2. 2dnr部分添加了一个功能，可以自动生成函数调用关系图，并以png格式保存在本地。
3. bmpimage，show_bmpimage函数支持显示彩色或黑白图片。并将直方图分析函数test_show_bmp_histogram和
gamma曲线拟合函数test_show_bf3a03_gamma 移动到对应的文件中。
4. filter文件增加一个 gaussBlur 函数。
5. histogram文件增加一个 test_show_bmp_histogram 函数。
6. rawimage文件增加一个 raw_image_show_3D 函数，但是该函数没有测试过。
7. yuvimage文件增加一个 rgb2ycbcr 函数。


# 20200818 17:58
1. yuvimage文件增加一个 rgb2lab 函数。
2. 增加 cj_demosaic 文件。


# 20200819 19:53
1. 将  cj_bmpimage.py 文件修改为  cj_rgbimage.py
2. 整理了  cj_demosaic.py 文件，将三种算法整合在一起。该部分算法已经全部测试过可行。
3. cj_gamma.py 文件增加了 degamma_srgb gamma_srgb两个函数，但是该函数没有测试过。
4. 新增了一个 cj_csc.py 文件，将所有的颜色空间转换函数都放在这个文件中，并新增了 rgb xyz lab 三者转换的函数，但是这些函数没有测试过。
5. 新增了一个 cj_ccm.py 文件，但是这些函数没有测试过。
 


 














 