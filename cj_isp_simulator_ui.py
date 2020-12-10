from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox
from PySide2.QtUiTools import QUiLoader


class Stats:
    def __init__(self):
        self.ui = QUiLoader().load('ui/isp_simulator.ui')
        self.ui.button.clicked.connect(self.handleCalc)

    def handleCalc(self):
        print("clicked button", self)


if __name__ == "__main__":
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
