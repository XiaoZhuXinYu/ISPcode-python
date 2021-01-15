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


class DSNU:
    def __init__(self):
        # 从文件中加载UI定义
        qfile = QFile("ui/cmos_dsnu.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile)

        # cmos_dsnu button
        self.ui.pushButton_dsnu_calculate.clicked.connect(self.handle_pushButton_dsnu_calculate_clicked)
        self.ui.pushButton_dsnu_image1.clicked.connect(self.handle_pushButton_dsnu_image1_clicked)
        self.ui.pushButton_dsnu_image2.clicked.connect(self.handle_pushButton_dsnu_image2_clicked)

        # cmos_dsnu lineEdit
        self.ui.lineEdit_dsnu_width.setText('5344')
        self.ui.lineEdit_dsnu_width.textChanged.connect(self.handle_lineEdit_dsnu_width_change)
        self.ui.lineEdit_dsnu_height.setText('3744')
        self.ui.lineEdit_dsnu_height.textChanged.connect(self.handle_lineEdit_dsnu_height_change)
        self.ui.lineEdit_dsnu_widthpercent.setText('50')
        self.ui.lineEdit_dsnu_widthpercent.textChanged.connect(self.handle_lineEdit_dsnu_widthpercent_change)
        self.ui.lineEdit_dsnu_heightpercent.setText('50')
        self.ui.lineEdit_dsnu_heightpercent.textChanged.connect(self.handle_lineEdit_dsnu_heightpercent_change)
        self.ui.lineEdit_dsnu_time_image1.setText('200')
        self.ui.lineEdit_dsnu_time_image1.textChanged.connect(self.handle_lineEdit_dsnu_time_image1_change)
        self.ui.lineEdit_dsnu_time_image2.setText('66')
        self.ui.lineEdit_dsnu_time_image2.textChanged.connect(self.handle_lineEdit_dsnu_time_image2_change)

        # cmos_dsnu comboBox
        self.ui.comboBox_dsnu_sensorbit.currentIndexChanged.connect(self.handle_comboBox_dsnu_sensorbit_change)
        self.ui.comboBox_dsnu_sensorbit.addItems(['16', '8', '12', '20'])

        self.ui.comboBox_dsnu_cfa.currentIndexChanged.connect(self.handle_comboBox_dsnu_cfa_change)
        self.ui.comboBox_dsnu_cfa.addItems(['rggb', 'bggr', 'gbrg', 'grbg', 'gray', 'color'])

        # cmos_dsnu global variable
        self.filePath1 = ""
        self.filePath2 = ""

    # cmos_dsnu button
    def handle_pushButton_dsnu_image1_clicked(self, checked):
        print("cmos_dsnu_image1 pushButton clicked", self)
        self.filePath1, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            "./",  # 起始目录
            "图片类型 (*.raw *.RAW *.pgm)"  # 选择类型过滤项，过滤内容在括号中
        )

    def handle_pushButton_dsnu_image2_clicked(self, checked):
        print("cmos_dsnu_image2 pushButton clicked", self)
        self.filePath2, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            "./",  # 起始目录
            "图片类型 (*.raw *.RAW *.pgm)"  # 选择类型过滤项，过滤内容在括号中
        )

    def handle_pushButton_dsnu_calculate_clicked(self, checked):
        print("cmos_dsnu_calculate pushButton clicked", self)
        print("path1:\n", self.filePath1)
        print("path2:\n", self.filePath2)

        width = int(self.ui.lineEdit_dsnu_width.text())
        height = int(self.ui.lineEdit_dsnu_height.text())
        widthpercent = int(self.ui.lineEdit_dsnu_widthpercent.text()) / 100
        heightpercent = int(self.ui.lineEdit_dsnu_heightpercent.text()) / 100
        time_image1 = int(self.ui.lineEdit_dsnu_time_image1.text())
        time_image2 = int(self.ui.lineEdit_dsnu_time_image2.text())

        sensorbit = int(self.ui.comboBox_dsnu_sensorbit.currentText())
        pattern = self.ui.comboBox_dsnu_cfa.currentText()

        image1 = rawimage.read_plained_file(self.filePath1, 'uint16', width, height, 'ieee-le')
        image2 = rawimage.read_plained_file(self.filePath2, 'uint16', width, height, 'ieee-le')
        image = image1 - image2
        R, GR, GB, B, G = rawimage.bayer_channel_separation(image, pattern)
        WidthBorder = round((1 - widthpercent) * width / 4)
        HeightBorder = round((1 - heightpercent) * height / 4)
        G = G[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
        std_G = np.std(G)
        fsd = pow(2, sensorbit) - 1
        dsnu_value = 1000 * std_G / ((time_image1 - time_image2) * fsd)
        self.ui.lineEdit_dsnu_value.setText(str(format(dsnu_value, '.7f')))

    # cmos_dsnu lineEdit
    def handle_lineEdit_dsnu_width_change(self):
        print("currentText is ", self.ui.lineEdit_dsnu_width.text())

    def handle_lineEdit_dsnu_height_change(self):
        print("currentText is ", self.ui.lineEdit_dsnu_height.text())

    def handle_lineEdit_dsnu_widthpercent_change(self):
        print("currentText is ", self.ui.lineEdit_dsnu_widthpercent.text())

    def handle_lineEdit_dsnu_heightpercent_change(self):
        print("currentText is ", self.ui.lineEdit_dsnu_heightpercent.text())

    def handle_lineEdit_dsnu_time_image1_change(self):
        print("currentText is ", self.ui.lineEdit_dsnu_time_image1.text())

    def handle_lineEdit_dsnu_time_image2_change(self):
        print("currentText is ", self.ui.lineEdit_dsnu_time_image2.text())

    # cmos_dsnu comboBox
    def handle_comboBox_dsnu_sensorbit_change(self):
        print("currentText is ", self.ui.comboBox_dsnu_sensorbit.currentText())

    def handle_comboBox_dsnu_cfa_change(self):
        print("currentText is ", self.ui.comboBox_dsnu_cfa.currentText())


if __name__ == "__main__":
    app = QApplication([])
    stats_cmos_dsnu = DSNU()
    stats_cmos_dsnu.ui.show()
    app.exec_()
