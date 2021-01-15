from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QFileDialog
import cj_rawimage as rawimage
import cj_histogram as histogram
import cj_ui_cmos_snr as snr
import cj_ui_cmos_dynamic_range as dyrange
import cj_ui_cmos_temporal_noise as tempnoise
import cj_ui_cmos_fpn as fpn
import cj_ui_cmos_dsnu as dsnu
import cj_ui_cmos_prnu as prnu
import cj_ui_cmos_sensitivity as sensitivity
import cj_ui_cmos_illumination as illumination


cmos_snr_win = None
cmos_dyrange_win = None
cmos_tempnoise_win = None
cmos_fpn_win = None
cmos_dsnu_win = None
cmos_prnu_win = None
cmos_sensitivity_win = None
cmos_illumination_win = None


class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        qfile = QFile("ui/isp.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        print("test")
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile)

        # RawImage button
        self.ui.pushButton_addfile.clicked.connect(self.handle_pushButton_addfile_clicked)
        self.ui.pushButton_viewrawfile.clicked.connect(self.handle_pushButton_viewrawfile_clicked)
        self.ui.pushButton_raw2plain16.clicked.connect(self.handle_pushButton_raw2plain16_clicked)
        self.ui.pushButton_raw2dng.clicked.connect(self.handle_pushButton_raw2dng_clicked)

        # RawImage lineEdit
        self.ui.lineEdit_dir.setPlaceholderText('请点击按钮添加文件')
        self.ui.lineEdit_width.setText('1920')
        self.ui.lineEdit_width.textChanged.connect(self.handle_lineEdit_width_change)
        self.ui.lineEdit_height.setText('1080')
        self.ui.lineEdit_height.textChanged.connect(self.handle_lineEdit_height_change)
        self.ui.lineEdit_dgain.setText('1.0')
        self.ui.lineEdit_dgain.textChanged.connect(self.handle_lineEdit_dgain_change)

        # RawImage comboBox
        self.ui.comboBox_bitshift.currentIndexChanged.connect(self.handle_comboBox_bitshift_change)
        # 这里做一个 addItem addItems 的用法说明，后面添加多个统一用 addItems
        self.ui.comboBox_bitshift.addItem('-4')
        self.ui.comboBox_bitshift.addItems(['-8', '-2', '0', '2', '4', '8'])

        self.ui.comboBox_pixeloffset.currentIndexChanged.connect(self.handle_comboBox_pixeloffset_change)
        self.ui.comboBox_pixeloffset.addItems(['0', '2', '4', '8', '16', '32'])

        self.ui.comboBox_inputformat.currentIndexChanged.connect(self.handle_comboBox_inputformat_change)
        self.ui.comboBox_inputformat.addItems(['uint16', 'uint8', 'uint32'])

        self.ui.comboBox_outputformat.currentIndexChanged.connect(self.handle_comboBox_outputformat_change)
        self.ui.comboBox_outputformat.addItems(['uint16', 'uint8', 'uint32'])

        self.ui.comboBox_dataformat.currentIndexChanged.connect(self.handle_comboBox_dataformat_change)
        self.ui.comboBox_dataformat.addItems(['ieee-le', 'ieee-be'])

        self.ui.comboBox_cfa.currentIndexChanged.connect(self.handle_comboBox_cfa_change)
        self.ui.comboBox_cfa.addItems(['rggb', 'bggr', 'gbrg', 'grbg', 'gray', 'color'])

        self.ui.comboBox_rawtype.currentIndexChanged.connect(self.handle_comboBox_rawtype_change)
        self.ui.comboBox_rawtype.addItems(['raw_plain16', 'customer_raw', 'mipi_raw8', 'mipi_raw10'])

        # CmosSensor
        self.ui.pushButton_snr_linear.clicked.connect(self.handle_pushButton_snr_linear_clicked)
        self.ui.pushButton_dynamicrange.clicked.connect(self.handle_pushButton_dynamicrange_clicked)
        self.ui.pushButton_temporalnoise.clicked.connect(self.handle_pushButton_temporalnoise_clicked)
        self.ui.pushButton_fpn.clicked.connect(self.handle_pushButton_fpn_clicked)
        self.ui.pushButton_dsnu.clicked.connect(self.handle_pushButton_dsnu_clicked)
        self.ui.pushButton_prnu.clicked.connect(self.handle_pushButton_prnu_clicked)
        self.ui.pushButton_sensitivity.clicked.connect(self.handle_pushButton_sensitivity_clicked)
        self.ui.pushButton_illumination.clicked.connect(self.handle_pushButton_illumination_clicked)

        # other page
        self.ui.pushButton_isppipeline.clicked.connect(self.isppipeline_pushButton_clicked)
        self.ui.pushButton_imagequality.clicked.connect(self.imagequality_pushButton_clicked)

    # RawImage button
    def handle_pushButton_addfile_clicked(self, checked):
        filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            "./",  # 起始目录
            "图片类型 (*.raw *.RAW)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.ui.lineEdit_dir.setText(filePath)
        print("addfile pushButton clicked", filePath)

    def handle_pushButton_viewrawfile_clicked(self, checked):
        dir1 = self.ui.lineEdit_dir.text()
        width = int(self.ui.lineEdit_width.text())
        height = int(self.ui.lineEdit_height.text())
        pattern = self.ui.comboBox_cfa.currentText()
        sensorbit = int(self.ui.comboBox_pixeloffset.currentText()) + 8
        dtype = self.ui.comboBox_inputformat.currentText()
        shift_bits = int(self.ui.comboBox_bitshift.currentText())
        dataformat = int(self.ui.comboBox_dataformat.currentText())
        print(sensorbit)
        iamge = rawimage.read_plained_file(dir1, dtype, width, height, dataformat)
        rawimage.show_planedraw(iamge, width, height, pattern, sensorbit)

        print("viewrawfile pushButton clicked", checked, self)

    def handle_pushButton_raw2plain16_clicked(self, checked):
        print("raw2plain16 pushButton clicked", checked, self)

    def handle_pushButton_raw2dng_clicked(self, checked):
        print("raw2dng pushButton clicked", checked, self)

    # RawImage lineEdit
    def handle_lineEdit_width_change(self):
        print("currentText is ", self.ui.lineEdit_width.text())

    def handle_lineEdit_height_change(self):
        print("currentText is ", self.ui.lineEdit_height.text())

    def handle_lineEdit_dgain_change(self):
        print("currentText is ", self.ui.lineEdit_dgain.text())

    # RawImage comboBox
    def handle_comboBox_bitshift_change(self):
        print("currentText is ", self.ui.comboBox_bitshift.currentText())

    def handle_comboBox_pixeloffset_change(self):
        print("currentText is ", self.ui.comboBox_pixeloffset.currentText())

    def handle_comboBox_inputformat_change(self):
        print("currentText is ", self.ui.comboBox_inputformat.currentText())

    def handle_comboBox_outputformat_change(self):
        print("currentText is ", self.ui.comboBox_outputformat.currentText())

    def handle_comboBox_dataformat_change(self):
        print("currentText is ", self.ui.comboBox_dataformat.currentText())

    def handle_comboBox_cfa_change(self):
        print("currentText is ", self.ui.comboBox_cfa.currentText())

    def handle_comboBox_rawtype_change(self):
        print("currentText is ", self.ui.comboBox_rawtype.currentText())

    # CmosSensor
    def handle_pushButton_snr_linear_clicked(self):
        global cmos_snr_win
        print("snr pushButton clicked", self)
        cmos_snr_win = snr.SNR()
        cmos_snr_win.ui.show()

    def handle_pushButton_dynamicrange_clicked(self):
        global cmos_dyrange_win
        print("dynamic range pushButton clicked", self)
        cmos_dyrange_win = dyrange.DynamicRange()
        cmos_dyrange_win.ui.show()

    def handle_pushButton_temporalnoise_clicked(self):
        global cmos_tempnoise_win
        print("temporal noise pushButton clicked", self)
        cmos_tempnoise_win = tempnoise.TemporalNoise()
        cmos_tempnoise_win.ui.show()

    def handle_pushButton_fpn_clicked(self):
        global cmos_fpn_win
        print("fpn pushButton clicked", self)
        cmos_fpn_win = fpn.FPN()
        cmos_fpn_win.ui.show()

    def handle_pushButton_dsnu_clicked(self):
        global cmos_dsnu_win
        print("dsnu pushButton clicked", self)
        cmos_dsnu_win = dsnu.DSNU()
        cmos_dsnu_win.ui.show()

    def handle_pushButton_prnu_clicked(self):
        global cmos_prnu_win
        print("prnu pushButton clicked", self)
        cmos_prnu_win = prnu.PRNU()
        cmos_prnu_win.ui.show()

    def handle_pushButton_sensitivity_clicked(self):
        global cmos_sensitivity_win
        print("sensitivity pushButton clicked", self)
        cmos_sensitivity_win = sensitivity.SENSITIVITY()
        cmos_sensitivity_win.ui.show()

    def handle_pushButton_illumination_clicked(self):
        global cmos_illumination_win
        print("illumination pushButton clicked", self)
        cmos_illumination_win = illumination.ILLUMINATION()
        cmos_illumination_win.ui.show()

    # other page
    def isppipeline_pushButton_clicked(self, checked):
        print("isppipeline pushButton clicked", checked, self)

    def imagequality_pushButton_clicked(self, checked):
        print("imagequality pushButton clicked", checked, self)


def handle_pushButton_snr1_clicked(checked):
    print("imagequality pushButton clicked", checked)
    # checked.ui.show()


if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(QIcon("pic/icon.png"))
    stats = Stats()

    # stats.ui.pushButton_snr.clicked.connect(handle_pushButton_snr1_clicked(stats_cmossnr))
    stats.ui.show()
    app.exec_()
