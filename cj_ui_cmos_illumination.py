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


class ILLUMINATION:
    def __init__(self):
        # 从文件中加载UI定义
        qfile = QFile("ui/cmos_illumination.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile)

        # cmos_illumination button
        self.ui.pushButton_illumination_calculate.clicked.connect(self.handle_pushButton_illumination_calculate_clicked)

        # cmos_illumination lineEdit
        self.ui.lineEdit_illumination_width.setText('1920')
        self.ui.lineEdit_illumination_width.textChanged.connect(self.handle_lineEdit_illumination_width_change)
        self.ui.lineEdit_illumination_height.setText('1080')
        self.ui.lineEdit_illumination_height.textChanged.connect(self.handle_lineEdit_illumination_height_change)
        self.ui.lineEdit_illumination_sensitivity.setText('1000')
        self.ui.lineEdit_illumination_sensitivity.textChanged.connect(
            self.handle_lineEdit_illumination_sensitivity_change)

        self.ui.lineEdit_illumination_MinIntTime.setText('0.001')
        self.ui.lineEdit_illumination_MinIntTime.textChanged.connect(
            self.handle_lineEdit_illumination_MinIntTime_change)

        self.ui.lineEdit_illumination_MaxIntTime.setText('1')
        self.ui.lineEdit_illumination_MaxIntTime.textChanged.connect(
            self.handle_lineEdit_illumination_MaxIntTime_change)

        # cmos_illumination comboBox
        self.ui.comboBox_illumination_sensorbit.currentIndexChanged.connect(
            self.handle_comboBox_illumination_sensorbit_change)
        self.ui.comboBox_illumination_sensorbit.addItems(['12', '8', '16', '20'])

        # cmos_illumination global variable

    # cmos_illumination button
    def handle_pushButton_illumination_calculate_clicked(self, checked):
        print("cmos_illumination_calculate pushButton clicked", self)

        filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            "./",  # 起始目录
            "图片类型 (*.raw *.RAW *.pgm)"  # 选择类型过滤项，过滤内容在括号中
        )

        width = int(self.ui.lineEdit_illumination_width.text())
        height = int(self.ui.lineEdit_illumination_height.text())
        sensitivity = int(self.ui.lineEdit_illumination_sensitivity.text())
        MinIntTime = float(self.ui.lineEdit_illumination_MinIntTime.text())
        MaxIntTime = float(self.ui.lineEdit_illumination_MaxIntTime.text())
        sensorbit = int(self.ui.comboBox_illumination_sensorbit.currentText())
        fsd = pow(2, sensorbit) - 1
        image = rawimage.read_plained_file(filePath, 'uint16', width, height, 'ieee-le')
        noise_floor = np.mean(image) + np.std(image)

        Min_Bright = noise_floor / (fsd * sensitivity * MaxIntTime)
        Max_Bright = 1 / (sensitivity * MinIntTime)

        self.ui.lineEdit_illumination_value_max.setText(str(format(Max_Bright, '.7f')))
        self.ui.lineEdit_illumination_value_min.setText(str(format(Min_Bright, '.7f')))

    # cmos_illumination lineEdit
    def handle_lineEdit_illumination_width_change(self):
        print("currentText is ", self.ui.lineEdit_illumination_width.text())

    def handle_lineEdit_illumination_height_change(self):
        print("currentText is ", self.ui.lineEdit_illumination_height.text())

    def handle_lineEdit_illumination_sensitivity_change(self):
        print("currentText is ", self.ui.lineEdit_illumination_sensitivity.text())

    def handle_lineEdit_illumination_MinIntTime_change(self):
        print("currentText is ", self.ui.lineEdit_illumination_MinIntTime.text())

    def handle_lineEdit_illumination_MaxIntTime_change(self):
        print("currentText is ", self.ui.lineEdit_illumination_MaxIntTime.text())

    # cmos_illumination comboBox
    def handle_comboBox_illumination_sensorbit_change(self):
        print("currentText is ", self.ui.comboBox_illumination_sensorbit.currentText())


if __name__ == "__main__":
    app = QApplication([])
    stats_cmos_illumination = ILLUMINATION()
    stats_cmos_illumination.ui.show()
    app.exec_()
