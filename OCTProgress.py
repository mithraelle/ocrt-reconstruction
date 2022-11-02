from PyQt5.QtWidgets import QWidget, QDialog
from forms.ocrtProcessingDlg import Ui_processingDialog

class OCTProgressUI:
    form: Ui_processingDialog
    window: QWidget

    def __init__(self):
        self.form = Ui_processingDialog()
        self.window = QDialog()
        self.form.setupUi(self.window)

    def show(self, label="Processing"):
        self.form.statusLabel.setText(label)
        self.window.exec()

    def hide(self):
        self.window.hide()
