import math

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QErrorMessage, QListWidgetItem, QErrorMessage, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPathItem
from PyQt5 import QtGui

from OCTAsset import OCTAssetList
from OCTFeature import OCTFeatureMatchSetList
from forms.ocrtFeatureDetectForm import Ui_octFDForm
from OCTFeatureMatchesView import OCTFeatureMatchesView

import OCTFeatureDection
from OCTAsset import OCTAssetList

import sys
import os

class OCTFeatureDetectForm:
    form: Ui_octFDForm
    window: QWidget

    assets_list: OCTAssetList
    raw_matches: OCTFeatureMatchSetList
    filtered_matches: OCTFeatureMatchSetList

    old_set_pair_id = -1

    feature_viewer: OCTFeatureMatchesView

    error_dialog: QErrorMessage

    base_ri: float = 1
    sample_w: float = 1.5
    sample_h: float = 2.0
    angle_dir: float = 1.0
    center_x: float = 0
    center_z: float = 0

    def init(self, window: QMainWindow):
        self.window = window
        self.form = Ui_octFDForm()
        self.form.setupUi(self.window)

        for i in OCTFeatureDection.OCTFeatureDetectionMethodsList:
            self.form.detectionMethods.addItem(i["name"])

        self.form.detect.clicked.connect(self.evaluate_feature_list)
        self.feature_viewer = OCTFeatureMatchesView()
        self.form.fdResultPreview.setScene(self.feature_viewer.scene)

        self.form.scanPairs.currentIndexChanged.connect(self.draw_matches)

        self.raw_matches = OCTFeatureMatchSetList()
        self.filtered_matches = OCTFeatureMatchSetList()

        self.form.selectBscan.clicked.connect(self.load_assets)
        self.form.loadBtn.clicked.connect(self.load_dataset)

        self.error_dialog = QErrorMessage(self.window)
        self.error_dialog.setModal(True)
        self.error_dialog.setWindowTitle("OCT Feature Detection Error")

        self.form.enableAllBtn.clicked.connect(self.select_matches)
        self.form.disableAllBtn.clicked.connect(self.deselect_matches)

        self.form.calculateBtn.clicked.connect(self.calculate_center)

    def load_assets(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.window, "Load dataset", ".",
                                                  "All Files (*)", options=options)
        if fileName:
            self.form.bScanPath.setText(fileName)

    def load_dataset(self):
        self.assets_list = OCTAssetList()
        dataset_fname = self.form.bScanPath.text()
        if len(dataset_fname) == 0 or self.assets_list.load(dataset_fname) == False:
            self.error_dialog.showMessage('Invalid datasource name')
        else:
            self.sample_w = float(self.form.XSizemm.text())
            self.sample_h = float(self.form.ZSizemm.text())
            self.base_ri = float(self.form.baseRI.text())
            self.center_x = float(self.form.centerX.text())
            self.center_z = float(self.form.centerZ.text())
            self.angle_dir = 1.0 if self.form.angleDirection.currentIndex() == 0 else -1.0
            self.raw_matches = OCTFeatureMatchSetList()
            self.set_feature_matches_list()
            self.resize_form()

    def show(self):
        self.window.show()
        self.form.scanPairs.clear()

    def assign_assets_list(self, assets_list: OCTAssetList):
        self.assets_list = assets_list
        self.raw_matches = OCTFeatureMatchSetList()
        self.set_feature_matches_list()
        self.resize_form()

    def evaluate_feature_list(self):
        detection_method = OCTFeatureDection.get_method_by_name(self.form.detectionMethods.currentText())
        self.raw_matches = self.assets_list.evaluate_feature_pairs(detection_method)
        self.set_feature_matches_list()

    def get_z_scale(self):
        [w, h] = self.assets_list.get_image_size()
        if self.sample_h != 0 and self.sample_w != 0:
            real_h = w * self.sample_h / self.sample_w / self.base_ri
        else:
            real_h = h / self.base_ri
        return real_h / h

    def set_feature_matches_list(self):
        dist = float(self.form.maxDist.text())
        area = float(self.form.activeArea.text()) / 100

        [w, h] = self.assets_list.get_image_size()
        z_scale = self.get_z_scale()

        center_x = self.center_x
        center_z = self.center_z
        if abs(center_x) < 0.0001:
            center_x = w / 2

        if abs(center_z) < 0.0001:
            center_z = h * z_scale / 2

        angle = self.angle_dir * 360.0 / self.assets_list.count()
        t = cv2.getRotationMatrix2D((center_x, center_z), angle, 1.0)

        self.filtered_matches = self.raw_matches.filter_matches(self.assets_list.get_image_size(),
                                                                dist, area, z_scale, t)
        total = 0
        self.form.scanPairs.clear()
        for i in self.raw_matches.feature_match_set:
            caption = str(self.assets_list.get_asset(i.idx0).angle) + " - "
            caption = caption + str(self.assets_list.get_asset(i.idx1).angle)
            self.form.scanPairs.addItem(caption)
            total = total + len(i.features)

        filtered = 0
        for i in self.filtered_matches.feature_match_set:
            filtered = filtered + len(i.features)

        print("Total: ", total)
        print("Filtered: ", filtered)


    def resize_form(self):
        offset = self.form.featureViewPanel.y() + 48
        min_form_height = offset + 20
        min_form_width = self.form.detect.x() + self.form.detect.width() + 4

        preview_w, preview_h = self.feature_viewer.set_view_size(self.assets_list)
        preview_w = preview_w + 2
        preview_h = preview_h + 2

        self.form.fdResultPreview.resize(preview_w, preview_h)

        form_w = max(min_form_width, preview_w + 8)
        form_h = max(min_form_height, preview_h + offset)

        self.form.featureViewPanel.resize(form_w, form_h)

        self.window.setFixedWidth(form_w+8)
        self.window.setFixedHeight(form_h+8)

    def draw_matches(self):
        set_id = self.form.scanPairs.currentIndex()

        if set_id > -1:
            if self.old_set_pair_id != set_id:
                self.old_set_pair_id = set_id
                feature_set = self.filtered_matches.get_feature_match_set(set_id)
                self.feature_viewer.set_b_scans(self.assets_list.get_asset(feature_set.idx0),
                                            self.assets_list.get_asset(feature_set.idx1))
                self.feature_viewer.draw_feature_matches(feature_set)
        else:
            self.old_set_pair_id = -1
            self.feature_viewer.clear()

    def deselect_matches(self):
        set_id = self.form.scanPairs.currentIndex()

        if set_id > -1:
            feature_set = self.filtered_matches.get_feature_match_set(set_id)
            for i in feature_set.features:
                i.set_state(False)
            self.feature_viewer.update()

    def select_matches(self):
        set_id = self.form.scanPairs.currentIndex()

        if set_id > -1:
            feature_set = self.filtered_matches.get_feature_match_set(set_id)
            for i in feature_set.features:
                i.set_state(True)
            self.feature_viewer.update()

    def calculate_center(self):
        center_x = []
        center_z = []

        angle = -self.angle_dir * 2 * math.pi / self.assets_list.count()
        sina = math.sin(angle)
        cosa = math.cos(angle)

        det = sina**2+(1-cosa)**2

        z_scale = self.get_z_scale()
        for feature_set in self.filtered_matches.feature_match_set:
            for feature in feature_set.features:
                if feature.get_state():
                    tx = feature.x0[0] - cosa * feature.x1[0] + sina * feature.x1[1] * z_scale
                    tz = feature.x0[1] * z_scale - sina * feature.x1[0] - cosa * feature.x1[1] * z_scale
                    x = (1 - cosa) * tx - sina * tz
                    z = sina * tx + (1 - cosa) * tz
                    center_x.append(x / det)
                    center_z.append(z / det)

        self.center_x = np.array(center_x).mean()
        self.center_z = np.array(center_z).mean()
        self.form.centerX.setText("{:.4f}".format(self.center_x))
        self.form.centerZ.setText("{:.4f}".format(self.center_z))


if __name__ == '__main__':
    print(os.getcwd())

    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = OCTFeatureDetectForm()
    ui.init(window)
    window.show()

    sys.exit(app.exec_())