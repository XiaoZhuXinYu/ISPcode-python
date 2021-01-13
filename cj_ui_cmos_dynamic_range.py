from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QFileDialog
import cj_rawimage as rawimage
import cj_histogram as histogram
import os
import numpy as np
from matplotlib import pyplot as plt


class DynamicRange:
    def __init__(self):
        # 从文件中加载UI定义
        qfile1 = QFile("ui/cmos_dynamic_range.ui")
        qfile1.open(QFile.ReadOnly)
        qfile1.close()
        print("test dynamic range")
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile1)

        # cmos_dynamic_range button
        self.ui.pushButton_dynamic_range_calculate.clicked.connect(
            self.handle_pushButton_dynamic_range_calculate_clicked)

        # cmos_dynamic_range lineEdit
        self.ui.lineEdit_dynamic_range_width.setText('1920')
        self.ui.lineEdit_dynamic_range_width.textChanged.connect(self.handle_lineEdit_dynamic_range_width_change)
        self.ui.lineEdit_dynamic_range_height.setText('1080')
        self.ui.lineEdit_dynamic_range_height.textChanged.connect(self.handle_lineEdit_dynamic_range_height_change)
        self.ui.lineEdit_dynamic_range_bitdepth.setText('12')
        self.ui.lineEdit_dynamic_range_bitdepth.textChanged.connect(self.handle_lineEdit_dynamic_range_bitdepth_change)
        self.ui.lineEdit_dynamic_range_fsd.setText('4095')
        self.ui.lineEdit_dynamic_range_fsd.textChanged.connect(self.handle_lineEdit_dynamic_range_fsd_change)

    # cmos_dynamic_range button
    def handle_pushButton_dynamic_range_calculate_clicked(self, checked):
        print("dynamic_range_calculate pushButton clicked", self)
        filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            "./",  # 起始目录
            "图片类型 (*.raw *.RAW)"  # 选择类型过滤项，过滤内容在括号中
        )
        width = int(self.ui.lineEdit_dynamic_range_width.text())
        height = int(self.ui.lineEdit_dynamic_range_height.text())
        bitdepth = int(self.ui.lineEdit_dynamic_range_bitdepth.text())
        fsd = int(self.ui.lineEdit_dynamic_range_fsd.text())

        iamge = rawimage.read_plained_file(filePath, 'uint16', width, height, 'ieee-le')
        noise = np.std(iamge)
        db = 20 * np.log10(fsd / noise)
        db = format(db, '.2f')
        self.ui.lineEdit_dynamic_range_db.setText(str(db))

        # cmos_dynamic_range lineEdit
    def handle_lineEdit_dynamic_range_width_change(self):
        print("currentText is ", self.ui.lineEdit_dynamic_range_width.text())

    def handle_lineEdit_dynamic_range_height_change(self):
        print("currentText is ", self.ui.lineEdit_dynamic_range_height.text())

    def handle_lineEdit_dynamic_range_bitdepth_change(self):
        print("currentText is ", self.ui.lineEdit_dynamic_range_bitdepth.text())

    def handle_lineEdit_dynamic_range_fsd_change(self):
        print("currentText is ", self.ui.lineEdit_dynamic_range_fsd.text())


if __name__ == "__main__":
    app = QApplication([])
    stats_cmos_dynamic_range = DynamicRange()
    stats_cmos_dynamic_range.ui.show()
    app.exec_()
