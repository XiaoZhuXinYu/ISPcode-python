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


class SENSITIVITY:
    def __init__(self):
        # 从文件中加载UI定义
        qfile = QFile("ui/cmos_sensitivity.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile)

        # cmos_sensitivity button
        self.ui.pushButton_sensitivity_calculate.clicked.connect(self.handle_pushButton_sensitivity_calculate_clicked)
        self.ui.pushButton_sensitivity_image_illuminated.clicked.connect(
            self.handle_pushButton_sensitivity_image_illuminated_clicked)
        self.ui.pushButton_sensitivity_image_dark.clicked.connect(self.handle_pushButton_sensitivity_image_dark_clicked)

        # cmos_sensitivity lineEdit
        self.ui.lineEdit_sensitivity_width.setText('1920')
        self.ui.lineEdit_sensitivity_width.textChanged.connect(self.handle_lineEdit_sensitivity_width_change)
        self.ui.lineEdit_sensitivity_height.setText('1080')
        self.ui.lineEdit_sensitivity_height.textChanged.connect(self.handle_lineEdit_sensitivity_height_change)
        self.ui.lineEdit_sensitivity_widthpercent.setText('50')
        self.ui.lineEdit_sensitivity_widthpercent.textChanged.connect(
            self.handle_lineEdit_sensitivity_widthpercent_change)
        self.ui.lineEdit_sensitivity_heightpercent.setText('50')
        self.ui.lineEdit_sensitivity_heightpercent.textChanged.connect(
            self.handle_lineEdit_sensitivity_heightpercent_change)
        self.ui.lineEdit_sensitivity_bright.setText('200')
        self.ui.lineEdit_sensitivity_bright.textChanged.connect(self.handle_lineEdit_sensitivity_bright_change)
        self.ui.lineEdit_sensitivity_time_int.setText('30')
        self.ui.lineEdit_sensitivity_time_int.textChanged.connect(self.handle_lineEdit_sensitivity_time_int_change)

        # cmos_sensitivity comboBox
        self.ui.comboBox_sensitivity_sensorbit.currentIndexChanged.connect(
            self.handle_comboBox_sensitivity_sensorbit_change)
        self.ui.comboBox_sensitivity_sensorbit.addItems(['12', '8', '16', '20'])

        self.ui.comboBox_sensitivity_cfa.currentIndexChanged.connect(self.handle_comboBox_sensitivity_cfa_change)
        self.ui.comboBox_sensitivity_cfa.addItems(['rggb', 'bggr', 'gbrg', 'grbg', 'gray', 'color'])

        # cmos_sensitivity global variable
        self.filePath1 = ""
        self.filePath2 = ""

    # cmos_sensitivity button
    def handle_pushButton_sensitivity_image_illuminated_clicked(self, checked):
        print("cmos_sensitivity_image_illuminated pushButton clicked", self)
        self.filePath1, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            "./",  # 起始目录
            "图片类型 (*.raw *.RAW *.pgm)"  # 选择类型过滤项，过滤内容在括号中
        )

    def handle_pushButton_sensitivity_image_dark_clicked(self, checked):
        print("cmos_sensitivity_image_dark pushButton clicked", self)
        self.filePath2, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            "./",  # 起始目录
            "图片类型 (*.raw *.RAW *.pgm)"  # 选择类型过滤项，过滤内容在括号中
        )

    def handle_pushButton_sensitivity_calculate_clicked(self, checked):
        print("cmos_sensitivity_calculate pushButton clicked", self)
        print("path1:\n", self.filePath1)
        print("path2:\n", self.filePath2)

        width = int(self.ui.lineEdit_sensitivity_width.text())
        height = int(self.ui.lineEdit_sensitivity_height.text())
        widthpercent = int(self.ui.lineEdit_sensitivity_widthpercent.text()) / 100
        heightpercent = int(self.ui.lineEdit_sensitivity_heightpercent.text()) / 100
        bright = int(self.ui.lineEdit_sensitivity_bright.text())
        time_int = int(self.ui.lineEdit_sensitivity_time_int.text())

        sensorbit = int(self.ui.comboBox_sensitivity_sensorbit.currentText())
        pattern = self.ui.comboBox_sensitivity_cfa.currentText()

        image_white = rawimage.read_plained_file(self.filePath1, 'uint16', width, height, 'ieee-le')
        image_black = rawimage.read_plained_file(self.filePath2, 'uint16', width, height, 'ieee-le')
        image = image_white - image_black
        R, GR, GB, B, G = rawimage.bayer_channel_separation(image, pattern)
        WidthBorder = round((1 - widthpercent) * width / 4)
        HeightBorder = round((1 - heightpercent) * height / 4)
        G = G[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
        G_mean = np.mean(G)
        fsd = pow(2, sensorbit) - 1
        sensitivity_value = G_mean / (bright * fsd / time_int)
        self.ui.lineEdit_sensitivity_value.setText(str(format(sensitivity_value, '.7f')))

    # cmos_sensitivity lineEdit
    def handle_lineEdit_sensitivity_width_change(self):
        print("currentText is ", self.ui.lineEdit_sensitivity_width.text())

    def handle_lineEdit_sensitivity_height_change(self):
        print("currentText is ", self.ui.lineEdit_sensitivity_height.text())

    def handle_lineEdit_sensitivity_widthpercent_change(self):
        print("currentText is ", self.ui.lineEdit_sensitivity_widthpercent.text())

    def handle_lineEdit_sensitivity_heightpercent_change(self):
        print("currentText is ", self.ui.lineEdit_sensitivity_heightpercent.text())

    def handle_lineEdit_sensitivity_bright_change(self):
        print("currentText is ", self.ui.lineEdit_sensitivity_bright.text())

    def handle_lineEdit_sensitivity_time_int_change(self):
        print("currentText is ", self.ui.lineEdit_sensitivity_time_int.text())

    # cmos_sensitivity comboBox
    def handle_comboBox_sensitivity_sensorbit_change(self):
        print("currentText is ", self.ui.comboBox_sensitivity_sensorbit.currentText())

    def handle_comboBox_sensitivity_cfa_change(self):
        print("currentText is ", self.ui.comboBox_sensitivity_cfa.currentText())


if __name__ == "__main__":
    app = QApplication([])
    stats_cmos_sensitivity = SENSITIVITY()
    stats_cmos_sensitivity.ui.show()
    app.exec_()
