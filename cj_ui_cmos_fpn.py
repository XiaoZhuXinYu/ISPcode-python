from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QFileDialog
import cj_rawimage as rawimage
import cj_histogram as histogram
import math
import sys
import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


class FPN:
    def __init__(self):
        # 从文件中加载UI定义
        qfile = QFile("ui/cmos_fpn.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile)

        # cmos_fpn button
        self.ui.pushButton_fpn_calculate.clicked.connect(self.handle_pushButton_fpn_calculate_clicked)

        # cmos_fpn lineEdit
        self.ui.lineEdit_fpn_width.setText('1920')
        self.ui.lineEdit_fpn_width.textChanged.connect(self.handle_lineEdit_fpn_width_change)
        self.ui.lineEdit_fpn_height.setText('1080')
        self.ui.lineEdit_fpn_height.textChanged.connect(self.handle_lineEdit_fpn_height_change)
        self.ui.lineEdit_fpn_start_x.setText('60')
        self.ui.lineEdit_fpn_start_x.textChanged.connect(self.handle_lineEdit_fpn_start_x_change)
        self.ui.lineEdit_fpn_start_y.setText('30')
        self.ui.lineEdit_fpn_start_y.textChanged.connect(self.handle_lineEdit_fpn_start_y_change)
        self.ui.lineEdit_fpn_roi_width.setText('50')
        self.ui.lineEdit_fpn_roi_width.textChanged.connect(self.handle_lineEdit_fpn_roi_width_change)
        self.ui.lineEdit_fpn_roi_height.setText('50')
        self.ui.lineEdit_fpn_roi_height.textChanged.connect(self.handle_lineEdit_fpn_roi_height_change)

        # cmos_fpn comboBox
        self.ui.comboBox_fpn_bitshift.currentIndexChanged.connect(self.handle_comboBox_fpn_bitshift_change)
        self.ui.comboBox_fpn_bitshift.addItems(['-4', '-8', '-2', '0', '2', '4', '8'])

        self.ui.comboBox_fpn_pixeloffset.currentIndexChanged.connect(self.handle_comboBox_fpn_pixeloffset_change)
        self.ui.comboBox_fpn_pixeloffset.addItems(['0', '2', '4', '8', '16', '32'])

        self.ui.comboBox_fpn_inputformat.currentIndexChanged.connect(self.handle_comboBox_fpn_inputformat_change)
        self.ui.comboBox_fpn_inputformat.addItems(['uint16', 'uint8', 'uint32'])

        self.ui.comboBox_fpn_outputformat.currentIndexChanged.connect(self.handle_comboBox_fpn_outputformat_change)
        self.ui.comboBox_fpn_outputformat.addItems(['uint16', 'uint8', 'uint32'])

        self.ui.comboBox_fpn_dataformat.currentIndexChanged.connect(self.handle_comboBox_fpn_dataformat_change)
        self.ui.comboBox_fpn_dataformat.addItems(['ieee-le', 'ieee-be'])

        self.ui.comboBox_fpn_cfa.currentIndexChanged.connect(self.handle_comboBox_fpn_cfa_change)
        self.ui.comboBox_fpn_cfa.addItems(['rggb', 'bggr', 'gbrg', 'grbg', 'gray', 'color'])

    # cmos_fpn button
    def handle_pushButton_fpn_calculate_clicked(self, checked):
        print("cmos_fpn_calculate pushButton clicked", self)
        i = 0
        filePath = QFileDialog.getExistingDirectory(self.ui, "选择存储路径")
        for root, dirs, files in os.walk(filePath):
            for f in files:
                i = i + 1  # 统计文件夹内总共有几个文件
        # print("i=", i)
        if i == 0:  # 没有加载文件夹
            return

        img_num = i
        filename = [0] * img_num  # 初始化成员为i个的一个列表
        width = int(self.ui.lineEdit_fpn_width.text())
        height = int(self.ui.lineEdit_fpn_height.text())
        roi_x = int(self.ui.lineEdit_fpn_start_x.text())
        roi_y = int(self.ui.lineEdit_fpn_start_y.text())
        roi_width = int(self.ui.lineEdit_fpn_roi_width.text())
        roi_height = int(self.ui.lineEdit_fpn_roi_height.text())

        bit_shift = int(self.ui.comboBox_fpn_bitshift.currentText())
        pattern = self.ui.comboBox_fpn_cfa.currentText()
        dataformat = self.ui.comboBox_fpn_dataformat.currentText()
        inputformat = self.ui.comboBox_fpn_inputformat.currentText()
        Rvalue = np.zeros((roi_height, roi_width), np.int)
        average_Rvalue = np.zeros((roi_height, roi_width), np.int)
        i = 0
        for root, dirs, files in os.walk(filePath):
            for f in files:
                filename[i] = os.path.join(root, f)  # 将文件名填写到列表中
                image = rawimage.read_plained_file(filename[i], inputformat, width, height, dataformat)
                R, GR, GB, B, G = rawimage.bayer_channel_separation(image, pattern)
                Rvalue = R[roi_y:(roi_y + roi_height), roi_x:(roi_x + roi_width)]
                average_Rvalue = average_Rvalue + Rvalue
                # print("Rvalue = \n", Rvalue[0][0])
                # print("average_Rvalue = \n",  average_Rvalue[0][0])
                i = i + 1
        average_Rvalue = average_Rvalue / i
        # print("shape = ", average_Rvalue)
        Noise_tol = np.std(Rvalue)  # 所有元素参加计算标准差
        FPN_total = np.std(average_Rvalue)
        # print("Noise_tol = ", Noise_tol, str(Noise_tol))
        # print("FPN_total = ", FPN_total, str(FPN_total))
        self.ui.lineEdit_fpn_total.setText(str(format(FPN_total, '.7f')))
        self.ui.lineEdit_fpn_temporal_noise.setText(str(format(Noise_tol-FPN_total, '.7f')))

        column_array = np.mean(average_Rvalue, axis=0)  # axis=0，计算每一列的均值
        row_array = np.mean(average_Rvalue, axis=1)  # axis=1，计算每一行的均值
        sum_c = 0
        sum_r = 0
        # print("shape:", column_array.shape[0], row_array.shape)
        for i in range(0, column_array.shape[0]-1):
            sum_c = sum_c + pow((column_array[i] - column_array[i + 1]), 2)
        for i in range(0, row_array.shape[0]-1):
            sum_r = sum_r + pow((row_array[i] - row_array[i + 1]), 2)

        fsd = pow(2, (16 + bit_shift)) - 1
        fpn_v_level = np.sqrt(sum_c / (column_array.shape[0] - 1)) / fsd
        fpn_h_level = np.sqrt(sum_r / (row_array.shape[0] - 1)) / fsd
        KERNEL = np.array([-1, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1]) / 10
        fpn_v_max = max(signal.convolve(column_array, KERNEL)) / fsd
        fpn_h_max = max(signal.convolve(row_array, KERNEL)) / fsd
        self.ui.lineEdit_fpn_v_level.setText(str(format(fpn_v_level, '.7f')))
        self.ui.lineEdit_fpn_h_level.setText(str(format(fpn_h_level, '.7f')))
        self.ui.lineEdit_fpn_v_max.setText(str(format(fpn_v_max, '.7f')))
        self.ui.lineEdit_fpn_h_max.setText(str(format(fpn_h_max, '.7f')))

    # cmos_fpn lineEdit
    def handle_lineEdit_fpn_width_change(self):
        print("currentText is ", self.ui.lineEdit_fpn_width.text())

    def handle_lineEdit_fpn_height_change(self):
        print("currentText is ", self.ui.lineEdit_fpn_height.text())

    def handle_lineEdit_fpn_start_x_change(self):
        print("currentText is ", self.ui.lineEdit_fpn_start_x.text())

    def handle_lineEdit_fpn_start_y_change(self):
        print("currentText is ", self.ui.lineEdit_fpn_start_y.text())

    def handle_lineEdit_fpn_roi_width_change(self):
        print("currentText is ", self.ui.lineEdit_fpn_roi_width.text())

    def handle_lineEdit_fpn_roi_height_change(self):
        print("currentText is ", self.ui.lineEdit_fpn_roi_height.text())

    # cmos_fpn comboBox
    def handle_comboBox_fpn_bitshift_change(self):
        print("currentText is ", self.ui.comboBox_fpn_bitshift.currentText())

    def handle_comboBox_fpn_pixeloffset_change(self):
        print("currentText is ", self.ui.comboBox_fpn_pixeloffset.currentText())

    def handle_comboBox_fpn_inputformat_change(self):
        print("currentText is ", self.ui.comboBox_fpn_inputformat.currentText())

    def handle_comboBox_fpn_outputformat_change(self):
        print("currentText is ", self.ui.comboBox_fpn_outputformat.currentText())

    def handle_comboBox_fpn_dataformat_change(self):
        print("currentText is ", self.ui.comboBox_fpn_dataformat.currentText())

    def handle_comboBox_fpn_cfa_change(self):
        print("currentText is ", self.ui.comboBox_fpn_cfa.currentText())


if __name__ == "__main__":
    app = QApplication([])
    stats_cmos_fpn = FPN()
    stats_cmos_fpn.ui.show()
    app.exec_()
