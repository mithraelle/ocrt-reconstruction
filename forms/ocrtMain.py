# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'U:\ocrt-nn-reconstruction\forms/ocrtMain.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ocrtMain(object):
    def setupUi(self, ocrtMain):
        ocrtMain.setObjectName("ocrtMain")
        ocrtMain.setWindowModality(QtCore.Qt.NonModal)
        ocrtMain.resize(786, 654)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ocrtMain.sizePolicy().hasHeightForWidth())
        ocrtMain.setSizePolicy(sizePolicy)
        ocrtMain.setMinimumSize(QtCore.QSize(786, 654))
        ocrtMain.setMaximumSize(QtCore.QSize(786, 654))
        self.centralwidget = QtWidgets.QWidget(ocrtMain)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(782, 609))
        self.centralwidget.setMaximumSize(QtCore.QSize(782, 609))
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.tabMain = QtWidgets.QTabWidget(self.centralwidget)
        self.tabMain.setEnabled(False)
        self.tabMain.setGeometry(QtCore.QRect(4, 68, 780, 545))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabMain.sizePolicy().hasHeightForWidth())
        self.tabMain.setSizePolicy(sizePolicy)
        self.tabMain.setMinimumSize(QtCore.QSize(780, 545))
        self.tabMain.setMaximumSize(QtCore.QSize(780, 545))
        self.tabMain.setObjectName("tabMain")
        self.tabDataSourceView = QtWidgets.QWidget()
        self.tabDataSourceView.setObjectName("tabDataSourceView")
        self.label_11 = QtWidgets.QLabel(self.tabDataSourceView)
        self.label_11.setGeometry(QtCore.QRect(520, 16, 81, 16))
        self.label_11.setObjectName("label_11")
        self.bscansList = QtWidgets.QListWidget(self.tabDataSourceView)
        self.bscansList.setGeometry(QtCore.QRect(520, 35, 250, 481))
        self.bscansList.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.bscansList.setObjectName("bscansList")
        self.datasetSourceView = QtWidgets.QLabel(self.tabDataSourceView)
        self.datasetSourceView.setGeometry(QtCore.QRect(4, 4, 512, 512))
        self.datasetSourceView.setAlignment(QtCore.Qt.AlignCenter)
        self.datasetSourceView.setObjectName("datasetSourceView")
        self.checkBox_2 = QtWidgets.QCheckBox(self.tabDataSourceView)
        self.checkBox_2.setGeometry(QtCore.QRect(520, 0, 121, 17))
        self.checkBox_2.setObjectName("checkBox_2")
        self.tabMain.addTab(self.tabDataSourceView, "")
        self.tabReconstructionView = QtWidgets.QWidget()
        self.tabReconstructionView.setObjectName("tabReconstructionView")
        self.graphicsViewResult = QtWidgets.QGraphicsView(self.tabReconstructionView)
        self.graphicsViewResult.setGeometry(QtCore.QRect(4, 4, 512, 512))
        self.graphicsViewResult.setObjectName("graphicsViewResult")
        self.pushButton = QtWidgets.QPushButton(self.tabReconstructionView)
        self.pushButton.setEnabled(False)
        self.pushButton.setGeometry(QtCore.QRect(520, 112, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButtonSaveResult = QtWidgets.QPushButton(self.tabReconstructionView)
        self.pushButtonSaveResult.setEnabled(False)
        self.pushButtonSaveResult.setGeometry(QtCore.QRect(600, 112, 75, 23))
        self.pushButtonSaveResult.setObjectName("pushButtonSaveResult")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tabReconstructionView)
        self.groupBox_3.setGeometry(QtCore.QRect(520, 0, 249, 73))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(8, 19, 47, 13))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(8, 43, 47, 13))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(104, 43, 15, 13))
        self.label_9.setObjectName("label_9")
        self.sourceWidth_2 = QtWidgets.QLineEdit(self.groupBox_3)
        self.sourceWidth_2.setGeometry(QtCore.QRect(48, 16, 50, 20))
        self.sourceWidth_2.setObjectName("sourceWidth_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_3.setGeometry(QtCore.QRect(48, 40, 50, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(104, 19, 15, 13))
        self.label_10.setObjectName("label_10")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_2.setGeometry(QtCore.QRect(124, 15, 121, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3.setGeometry(QtCore.QRect(124, 39, 121, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.checkBox = QtWidgets.QCheckBox(self.tabReconstructionView)
        self.checkBox.setEnabled(False)
        self.checkBox.setGeometry(QtCore.QRect(520, 79, 105, 17))
        self.checkBox.setObjectName("checkBox")
        self.pushButton_4 = QtWidgets.QPushButton(self.tabReconstructionView)
        self.pushButton_4.setGeometry(QtCore.QRect(656, 76, 105, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.tabMain.addTab(self.tabReconstructionView, "")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(4, 0, 777, 68))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(8, 19, 47, 13))
        self.label.setObjectName("label")
        self.btnOpenDlg = QtWidgets.QToolButton(self.groupBox)
        self.btnOpenDlg.setGeometry(QtCore.QRect(488, 16, 25, 19))
        self.btnOpenDlg.setObjectName("btnOpenDlg")
        self.fieldDataFileName = QtWidgets.QLineEdit(self.groupBox)
        self.fieldDataFileName.setGeometry(QtCore.QRect(48, 16, 440, 20))
        self.fieldDataFileName.setObjectName("fieldDataFileName")
        self.fieldSourceDepth = QtWidgets.QLineEdit(self.groupBox)
        self.fieldSourceDepth.setGeometry(QtCore.QRect(696, 16, 50, 20))
        self.fieldSourceDepth.setObjectName("fieldSourceDepth")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(528, 19, 47, 13))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(624, 19, 47, 13))
        self.label_3.setObjectName("label_3")
        self.fieldSourceWidth = QtWidgets.QLineEdit(self.groupBox)
        self.fieldSourceWidth.setGeometry(QtCore.QRect(568, 16, 50, 20))
        self.fieldSourceWidth.setObjectName("fieldSourceWidth")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(752, 19, 47, 13))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(656, 19, 47, 13))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(528, 40, 81, 22))
        self.label_6.setObjectName("label_6")
        self.btnLoadData = QtWidgets.QPushButton(self.groupBox)
        self.btnLoadData.setEnabled(True)
        self.btnLoadData.setGeometry(QtCore.QRect(8, 38, 90, 23))
        self.btnLoadData.setObjectName("btnLoadData")
        self.btnDetectFeatures = QtWidgets.QPushButton(self.groupBox)
        self.btnDetectFeatures.setEnabled(False)
        self.btnDetectFeatures.setGeometry(QtCore.QRect(100, 38, 90, 23))
        self.btnDetectFeatures.setObjectName("btnDetectFeatures")
        self.listRotation = QtWidgets.QComboBox(self.groupBox)
        self.listRotation.setGeometry(QtCore.QRect(578, 40, 44, 22))
        self.listRotation.setObjectName("listRotation")
        self.listRotation.addItem("")
        self.listRotation.addItem("")
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(634, 40, 81, 22))
        self.label_12.setObjectName("label_12")
        self.listDirection = QtWidgets.QComboBox(self.groupBox)
        self.listDirection.setGeometry(QtCore.QRect(712, 40, 57, 22))
        self.listDirection.setObjectName("listDirection")
        self.listDirection.addItem("")
        self.listDirection.addItem("")
        self.listDirection.addItem("")
        self.listDirection.addItem("")
        ocrtMain.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ocrtMain)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 786, 21))
        self.menubar.setObjectName("menubar")
        ocrtMain.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ocrtMain)
        self.statusbar.setObjectName("statusbar")
        ocrtMain.setStatusBar(self.statusbar)

        self.retranslateUi(ocrtMain)
        self.tabMain.setCurrentIndex(0)
        self.listRotation.setCurrentIndex(0)
        self.listDirection.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(ocrtMain)

    def retranslateUi(self, ocrtMain):
        _translate = QtCore.QCoreApplication.translate
        ocrtMain.setWindowTitle(_translate("ocrtMain", "OCRT Reconstruction tool"))
        self.label_11.setText(_translate("ocrtMain", "B-Scans list"))
        self.datasetSourceView.setText(_translate("ocrtMain", "BScan preview"))
        self.checkBox_2.setText(_translate("ocrtMain", "Show in physical size"))
        self.tabMain.setTabText(self.tabMain.indexOf(self.tabDataSourceView), _translate("ocrtMain", "Overview"))
        self.pushButton.setText(_translate("ocrtMain", "Run"))
        self.pushButtonSaveResult.setText(_translate("ocrtMain", "Save to file"))
        self.groupBox_3.setTitle(_translate("ocrtMain", "Sample center"))
        self.label_7.setText(_translate("ocrtMain", "Width:"))
        self.label_8.setText(_translate("ocrtMain", "Depth:"))
        self.label_9.setText(_translate("ocrtMain", "px"))
        self.sourceWidth_2.setText(_translate("ocrtMain", "0"))
        self.lineEdit_3.setText(_translate("ocrtMain", "0"))
        self.label_10.setText(_translate("ocrtMain", "px"))
        self.pushButton_2.setText(_translate("ocrtMain", "Use image center"))
        self.pushButton_3.setText(_translate("ocrtMain", "Estimate from features"))
        self.checkBox.setText(_translate("ocrtMain", "Use RI correction"))
        self.pushButton_4.setText(_translate("ocrtMain", "Build RI map"))
        self.tabMain.setTabText(self.tabMain.indexOf(self.tabReconstructionView), _translate("ocrtMain", "Reconstruction"))
        self.groupBox.setTitle(_translate("ocrtMain", "Data source"))
        self.label.setText(_translate("ocrtMain", "Source:"))
        self.btnOpenDlg.setText(_translate("ocrtMain", "..."))
        self.fieldSourceDepth.setText(_translate("ocrtMain", "2"))
        self.label_2.setText(_translate("ocrtMain", "Width:"))
        self.label_3.setText(_translate("ocrtMain", "mm"))
        self.fieldSourceWidth.setText(_translate("ocrtMain", "2"))
        self.label_4.setText(_translate("ocrtMain", "mm"))
        self.label_5.setText(_translate("ocrtMain", "Depth:"))
        self.label_6.setText(_translate("ocrtMain", "Rotation:"))
        self.btnLoadData.setText(_translate("ocrtMain", "Load"))
        self.btnDetectFeatures.setText(_translate("ocrtMain", "Detect features"))
        self.listRotation.setCurrentText(_translate("ocrtMain", "CV"))
        self.listRotation.setItemText(0, _translate("ocrtMain", "CV"))
        self.listRotation.setItemText(1, _translate("ocrtMain", "CCV"))
        self.label_12.setText(_translate("ocrtMain", "Scan direction:"))
        self.listDirection.setItemText(0, _translate("ocrtMain", "N"))
        self.listDirection.setItemText(1, _translate("ocrtMain", "E"))
        self.listDirection.setItemText(2, _translate("ocrtMain", "S"))
        self.listDirection.setItemText(3, _translate("ocrtMain", "W"))
