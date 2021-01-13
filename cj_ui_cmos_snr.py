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


class SNR1:
    def __init__(self):
        print("test snr123")


class SNR:
    def __init__(self):
        # 从文件中加载UI定义
        qfile1 = QFile("ui/cmos_snr.ui")
        qfile1.open(QFile.ReadOnly)
        qfile1.close()
        print("test snr")
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile1)

        # cmos_snr button
        self.ui.pushButton_cmos_snr_snr.clicked.connect(self.handle_pushButton_cmos_snr_snr_clicked)
        self.ui.pushButton_cmos_snr_linearity.clicked.connect(self.handle_pushButton_cmos_snr_linearity_clicked)

        # cmos_snr lineEdit
        self.ui.lineEdit_cmos_snr_width.setText('1280')
        self.ui.lineEdit_cmos_snr_width.textChanged.connect(self.handle_lineEdit_cmos_snr_width_change)
        self.ui.lineEdit_cmos_snr_height.setText('1080')
        self.ui.lineEdit_cmos_snr_height.textChanged.connect(self.handle_lineEdit_cmos_snr_height_change)
        self.ui.lineEdit_cmos_snr_w_percent.setText('10')
        self.ui.lineEdit_cmos_snr_w_percent.textChanged.connect(self.handle_lineEdit_cmos_snr_w_percent_change)
        self.ui.lineEdit_cmos_snr_h_percent.setText('10')
        self.ui.lineEdit_cmos_snr_h_percent.textChanged.connect(self.handle_lineEdit_cmos_snr_h_percent_change)

        # cmos_snr comboBox
        self.ui.comboBox_cmos_snr_bitshift.currentIndexChanged.connect(self.handle_comboBox_cmos_snr_bitshift_change)
        self.ui.comboBox_cmos_snr_bitshift.addItems(['-4', '-8', '-2', '0', '2', '4', '8'])

        self.ui.comboBox_cmos_snr_dataformat.currentIndexChanged.connect(self.handle_comboBox_cmos_snr_dataformat_change)
        self.ui.comboBox_cmos_snr_dataformat.addItems(['ieee-le', 'ieee-be'])

        self.ui.comboBox_cmos_snr_cfa.currentIndexChanged.connect(self.handle_comboBox_cmos_snr_cfa_change)
        self.ui.comboBox_cmos_snr_cfa.addItems(['rggb', 'bggr', 'gbrg', 'grbg', 'gray', 'color'])

    # cmos_snr button
    def handle_pushButton_cmos_snr_snr_clicked(self, checked):
        print("cmos_snr_snr pushButton clicked", self)
        i = 0
        filePath = QFileDialog.getExistingDirectory(self.ui, "选择存储路径")
        for root, dirs, files in os.walk(filePath):
            for f in files:
                i = i + 1  # 统计文件夹内总共有几个文件
        print("i=", i)
        if i == 0:  # 没有加载文件夹
            return

        filename = [0] * i  # 初始化成员为i个的一个列表
        Rvalue = [0] * i
        RNoise = [0] * i
        R_SNR = [0] * i
        GRvalue = [0] * i
        GRNoise = [0] * i
        GR_SNR = [0] * i
        GBvalue = [0] * i
        GBNoise = [0] * i
        GB_SNR = [0] * i
        Bvalue = [0] * i
        BNoise = [0] * i
        B_SNR = [0] * i
        width = int(self.ui.lineEdit_cmos_snr_width.text())
        height = int(self.ui.lineEdit_cmos_snr_height.text())
        w_percent = int(self.ui.lineEdit_cmos_snr_w_percent.text()) / 100
        h_percent = int(self.ui.lineEdit_cmos_snr_h_percent.text()) / 100
        pattern = self.ui.comboBox_cmos_snr_cfa.currentText()
        dataformat = self.ui.comboBox_cmos_snr_dataformat.currentText()
        shift_bits = int(self.ui.comboBox_cmos_snr_bitshift.currentText())
        WidthBorder = round((1 - w_percent) * width / 4)
        HeightBorder = round((1 - h_percent) * height / 4)
        # print("WidthBorder:", WidthBorder, HeightBorder, (width / 2 - WidthBorder), (height / 2 - HeightBorder))
        i = 0
        for root, dirs, files in os.walk(filePath):
            for f in files:
                filename[i] = os.path.join(root, f)  # 将文件名填写到列表中
                iamge = rawimage.read_plained_file(filename[i], "uint16", width, height, dataformat)
                R, GR, GB, B, G = rawimage.bayer_channel_separation(iamge, pattern)
                R = R[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                GR = GR[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                GB = GB[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                B = B[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                # print("shape:", np.shape(R))
                Rvalue[i] = np.mean(R)
                RNoise[i] = np.std(R)
                R_SNR[i] = Rvalue[i] / RNoise[i]
                GRvalue[i] = np.mean(GR)
                GRNoise[i] = np.std(GR)
                GR_SNR[i] = GRvalue[i] / GRNoise[i]
                GBvalue[i] = np.mean(GB)
                GBNoise[i] = np.std(GB)
                GB_SNR[i] = GBvalue[i] / GBNoise[i]
                Bvalue[i] = np.mean(B)
                BNoise[i] = np.std(B)
                B_SNR[i] = Bvalue[i] / BNoise[i]
                print(filename[i])
                i = i + 1
        print("len = ", len(filename))
        if i > 1:
            x = np.arange(0, i)
            plt.plot(x, R_SNR, "r", label="R")
            plt.plot(x, GR_SNR, "g", label="GR")
            plt.plot(x, GB_SNR, "c", label="GB")
            plt.plot(x, B_SNR, "b", label="B")
        else:
            plt.scatter(1, R_SNR[0], color="r", label="R", linewidth=3)
            plt.scatter(2, GR_SNR[0], color="g", label="GR", linewidth=3)
            plt.scatter(3, GB_SNR[0], color="c", label="GB", linewidth=3)
            plt.scatter(4, B_SNR[0], color="b", label="B", linewidth=3)

        plt.title("SNR")
        plt.legend(loc="lower right")
        plt.show()
        self.ui.close()

    def handle_pushButton_cmos_snr_linearity_clicked(self, checked):
        print("cmos_snr_linearity pushButton clicked", self)
        i = 0
        filePath = QFileDialog.getExistingDirectory(self.ui, "选择存储路径")
        for root, dirs, files in os.walk(filePath):
            for f in files:
                i = i + 1  # 统计文件夹内总共有几个文件
        print("i=", i)
        if i < 2:  # 少于2个文件，无法测试线性化程度
            QMessageBox.warning(
                self.ui,
                '文件太少',
                '文件夹内文件数目必须大于1个')
            return

        filename = [0] * i  # 初始化成员为i个的一个列表
        Rvalue = [0] * i
        GRvalue = [0] * i
        GBvalue = [0] * i
        Bvalue = [0] * i

        width = int(self.ui.lineEdit_cmos_snr_width.text())
        height = int(self.ui.lineEdit_cmos_snr_height.text())
        w_percent = int(self.ui.lineEdit_cmos_snr_w_percent.text()) / 100
        h_percent = int(self.ui.lineEdit_cmos_snr_h_percent.text()) / 100
        pattern = self.ui.comboBox_cmos_snr_cfa.currentText()
        dataformat = self.ui.comboBox_cmos_snr_dataformat.currentText()
        shift_bits = int(self.ui.comboBox_cmos_snr_bitshift.currentText())
        WidthBorder = round((1 - w_percent) * width / 4)
        HeightBorder = round((1 - h_percent) * height / 4)
        # print("WidthBorder:", WidthBorder, HeightBorder, (width / 2 - WidthBorder), (height / 2 - HeightBorder))
        i = 0
        for root, dirs, files in os.walk(filePath):
            for f in files:
                filename[i] = os.path.join(root, f)  # 将文件名填写到列表中
                iamge = rawimage.read_plained_file(filename[i], "uint16", width, height, dataformat)
                R, GR, GB, B, G = rawimage.bayer_channel_separation(iamge, pattern)
                R = R[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                GR = GR[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                GB = GB[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                B = B[HeightBorder:int(height / 2 - HeightBorder), WidthBorder:int(width / 2 - WidthBorder)]
                # print("shape:", np.shape(R))
                Rvalue[i] = np.mean(R)
                GRvalue[i] = np.mean(GR)
                GBvalue[i] = np.mean(GB)
                Bvalue[i] = np.mean(B)

                print(filename[i])
                i = i + 1
        print("len = ", len(filename))
        x = np.arange(0, i)
        plt.plot(x, Rvalue, "r", label="R")
        plt.plot(x, GRvalue, "g", label="GR")
        plt.plot(x, GBvalue, "c", label="GB")
        plt.plot(x, Bvalue, "b", label="B")

        plt.title("Linearity")
        plt.legend(loc="lower right")
        plt.show()
        self.ui.close()

    # cmos_snr lineEdit
    def handle_lineEdit_cmos_snr_width_change(self):
        print("currentText is ", self.ui.lineEdit_cmos_snr_width.text())

    def handle_lineEdit_cmos_snr_height_change(self):
        print("currentText is ", self.ui.lineEdit_cmos_snr_height.text())

    def handle_lineEdit_cmos_snr_w_percent_change(self):
        print("currentText is ", self.ui.lineEdit_cmos_snr_w_percent.text())

    def handle_lineEdit_cmos_snr_h_percent_change(self):
        print("currentText is ", self.ui.lineEdit_cmos_snr_h_percent.text())

    # cmos_snr comboBox
    def handle_comboBox_cmos_snr_bitshift_change(self):
        print("currentText is ", self.ui.comboBox_cmos_snr_bitshift.currentText())

    def handle_comboBox_cmos_snr_dataformat_change(self):
        print("currentText is ", self.ui.comboBox_cmos_snr_dataformat.currentText())

    def handle_comboBox_cmos_snr_cfa_change(self):
        print("currentText is ", self.ui.comboBox_cmos_snr_cfa.currentText())


if __name__ == "__main__":
    app = QApplication([])
    stats_cmossnr = SNR()
    stats_cmossnr.ui.show()
    app.exec_()