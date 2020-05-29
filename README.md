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