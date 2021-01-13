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
from matplotlib import pyplot as plt


class TemporalNoise:
    def __init__(self):
        # 从文件中加载UI定义
        qfile = QFile("ui/cmos_temporal_noise.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile)

        # cmos_temporal_noise button
        self.ui.pushButton_temporal_noise_calculate.clicked.connect(
            self.handle_pushButton_temporal_noise_calculate_clicked)

        # cmos_temporal_noise lineEdit
        self.ui.lineEdit_temporal_noise_width.setText('1920')
        self.ui.lineEdit_temporal_noise_width.textChanged.connect(self.handle_lineEdit_temporal_noise_width_change)
        self.ui.lineEdit_temporal_noise_height.setText('1080')
        self.ui.lineEdit_temporal_noise_height.textChanged.connect(self.handle_lineEdit_temporal_noise_height_change)
        self.ui.lineEdit_temporal_noise_start_x.setText('60')
        self.ui.lineEdit_temporal_noise_start_x.textChanged.connect(self.handle_lineEdit_temporal_noise_start_x_change)
        self.ui.lineEdit_temporal_noise_start_y.setText('30')
        self.ui.lineEdit_temporal_noise_start_y.textChanged.connect(self.handle_lineEdit_temporal_noise_start_y_change)
        self.ui.lineEdit_temporal_noise_roi_width.setText('50')
        self.ui.lineEdit_temporal_noise_roi_width.textChanged.connect(
            self.handle_lineEdit_temporal_noise_roi_width_change)
        self.ui.lineEdit_temporal_noise_roi_height.setText('50')
        self.ui.lineEdit_temporal_noise_roi_height.textChanged.connect(
            self.handle_lineEdit_temporal_noise_roi_height_change)

        # cmos_temporal_noise comboBox
        self.ui.comboBox_temporal_noise_bitshift.currentIndexChanged.connect(
            self.handle_comboBox_temporal_noise_bitshift_change)
        self.ui.comboBox_temporal_noise_bitshift.addItems(['-4', '-8', '-2', '0', '2', '4', '8'])

        self.ui.comboBox_temporal_noise_pixeloffset.currentIndexChanged.connect(
            self.handle_comboBox_temporal_noise_pixeloffset_change)
        self.ui.comboBox_temporal_noise_pixeloffset.addItems(['0', '2', '4', '8', '16', '32'])

        self.ui.comboBox_temporal_noise_inputformat.currentIndexChanged.connect(
            self.handle_comboBox_temporal_noise_inputformat_change)
        self.ui.comboBox_temporal_noise_inputformat.addItems(['uint16', 'uint8', 'uint32'])

        self.ui.comboBox_temporal_noise_outputformat.currentIndexChanged.connect(
            self.handle_comboBox_temporal_noise_outputformat_change)
        self.ui.comboBox_temporal_noise_outputformat.addItems(['uint16', 'uint8', 'uint32'])

        self.ui.comboBox_temporal_noise_dataformat.currentIndexChanged.connect(
            self.handle_comboBox_temporal_noise_dataformat_change)
        self.ui.comboBox_temporal_noise_dataformat.addItems(['ieee-le', 'ieee-be'])

        self.ui.comboBox_temporal_noise_cfa.currentIndexChanged.connect(self.handle_comboBox_temporal_noise_cfa_change)
        self.ui.comboBox_temporal_noise_cfa.addItems(['rggb', 'bggr', 'gbrg', 'grbg', 'gray', 'color'])

    # cmos_temporal_noise button
    def handle_pushButton_temporal_noise_calculate_clicked(self, checked):
        print("cmos_temporal_noise_calculate pushButton clicked", self)
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
        width = int(self.ui.lineEdit_temporal_noise_width.text())
        height = int(self.ui.lineEdit_temporal_noise_height.text())
        roi_x = int(self.ui.lineEdit_temporal_noise_start_x.text())
        roi_y = int(self.ui.lineEdit_temporal_noise_start_y.text())
        roi_width = int(self.ui.lineEdit_temporal_noise_roi_width.text())
        roi_height = int(self.ui.lineEdit_temporal_noise_roi_height.text())

        bit_shift = int(self.ui.comboBox_temporal_noise_bitshift.currentText())
        pattern = self.ui.comboBox_temporal_noise_cfa.currentText()
        dataformat = self.ui.comboBox_temporal_noise_dataformat.currentText()
        inputformat = self.ui.comboBox_temporal_noise_inputformat.currentText()
        Rvalue = np.empty((img_num, roi_height, roi_width))
        i = 0
        for root, dirs, files in os.walk(filePath):
            for f in files:
                filename[i] = os.path.join(root, f)  # 将文件名填写到列表中
                image = rawimage.read_plained_file(filename[i], inputformat, width, height, dataformat)
                R, GR, GB, B, G = rawimage.bayer_channel_separation(image, pattern)
                Rvalue[i] = R[roi_y:(roi_y + roi_height), roi_x:(roi_x + roi_width)]
                i = i + 1
        var_Rvalue = Rvalue.var(axis=0)  # 不同图像同一位置的像素求方差
        mean_std = math.sqrt(np.sum(np.sum(var_Rvalue, axis=1), axis=0) / (roi_width * roi_height))
        fsd = pow(2, (16 + bit_shift)) - 1
        db = 20*np.log10(mean_std/fsd)
        db = format(db, '.2f')
        mean_std = format(mean_std, '.2f')
        self.ui.lineEdit_temporal_noise_emva1288.setText(str(mean_std))
        self.ui.lineEdit_temporal_noise_smia.setText(str(db))

    # cmos_temporal_noise lineEdit
    def handle_lineEdit_temporal_noise_width_change(self):
        print("currentText is ", self.ui.lineEdit_temporal_noise_width.text())

    def handle_lineEdit_temporal_noise_height_change(self):
        print("currentText is ", self.ui.lineEdit_temporal_noise_height.text())

    def handle_lineEdit_temporal_noise_start_x_change(self):
        print("currentText is ", self.ui.lineEdit_temporal_noise_start_x.text())

    def handle_lineEdit_temporal_noise_start_y_change(self):
        print("currentText is ", self.ui.lineEdit_temporal_noise_start_y.text())

    def handle_lineEdit_temporal_noise_roi_width_change(self):
        print("currentText is ", self.ui.lineEdit_temporal_noise_roi_width.text())

    def handle_lineEdit_temporal_noise_roi_height_change(self):
        print("currentText is ", self.ui.lineEdit_temporal_noise_roi_height.text())

    # cmos_temporal_noise comboBox
    def handle_comboBox_temporal_noise_bitshift_change(self):
        print("currentText is ", self.ui.comboBox_temporal_noise_bitshift.currentText())

    def handle_comboBox_temporal_noise_pixeloffset_change(self):
        print("currentText is ", self.ui.comboBox_temporal_noise_pixeloffset.currentText())

    def handle_comboBox_temporal_noise_inputformat_change(self):
        print("currentText is ", self.ui.comboBox_temporal_noise_inputformat.currentText())

    def handle_comboBox_temporal_noise_outputformat_change(self):
        print("currentText is ", self.ui.comboBox_temporal_noise_outputformat.currentText())

    def handle_comboBox_temporal_noise_dataformat_change(self):
        print("currentText is ", self.ui.comboBox_temporal_noise_dataformat.currentText())

    def handle_comboBox_temporal_noise_cfa_change(self):
        print("currentText is ", self.ui.comboBox_temporal_noise_cfa.currentText())


if __name__ == "__main__":
    app = QApplication([])
    stats_cmos_temporal_noise = TemporalNoise()
    stats_cmos_temporal_noise.ui.show()
    app.exec_()
