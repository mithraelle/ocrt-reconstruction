from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QErrorMessage, QListWidgetItem, QErrorMessage, QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPathItem
from forms.ocrtRIForm import Ui_ocrtRIForm
from PyQt5.QtGui import QPixmap, QBrush, QColor, QPen, QPainterPath

from OCTAsset import OCTAsset, OCTAssetList

import OCTHelper

import RIMap

from RILegend import RILegendItem

import cv2

from OCTProgress import OCTProgressUI

import numpy as np
import math

import sys
import os

class OCTRIMapUI:
    form: Ui_ocrtRIForm
    window: QMainWindow
    ri_map: RIMap
    ri_pixmap: QGraphicsPixmapItem
    ri_gen_pixmap: QGraphicsPixmapItem
    ri_preview_scene: QGraphicsScene
    ri_gen_scene: QGraphicsScene

    ri_rays: []
    ri_original_rays: []

    x_step: int
    z_step: float = 1.0
    sigma: float
    nodes: int

    sample_center_x: float = 0
    sample_center_z: float = 0
    base_ri: float = 1
    sample_w: float = 1.5
    sample_h: float = 2.0

    angle_count: int = 0
    angle_step: float = 0
    angle_dir: float = 1.0

    offset_x: int = 0
    offset_z: int = 0

    old_angle_idx: int = 0

    oct_bscan_list: []
    oct_asset_list: OCTAssetList

    error_dialog: QErrorMessage

    drawn_asset_id: int = -1

    legend: RILegendItem
    legend_gen: RILegendItem

    ri_scale: float = 1.0

    def init(self, window: QMainWindow):
        self.form = Ui_ocrtRIForm()
        self.form.setupUi(window)
        self.window = window

        self.ri_original_rays = []
        self.ri_rays = []
        self._rays = []

        self.ri_preview_scene = QGraphicsScene()
        self.form.RIPreview.setScene(self.ri_preview_scene)
        self.ri_pixmap = self.form.RIPreview.scene().addPixmap(QPixmap())

        self.form.makeRIbtn.clicked.connect(self.process_map)
        self.form.previewMode.currentIndexChanged.connect(self.draw_ri_map)
        self.form.angleList.currentIndexChanged.connect(self.draw_angle_sample)
        self.form.resetRIbtn.clicked.connect(self.reset_params)
        self.form.BScanBase.currentIndexChanged.connect(self.draw_bscan)

        self.form.saveRIBtn.clicked.connect(self.save_ri)
        self.form.saveBScanSampleBtn.clicked.connect(self.save_bscan_sample)
        self.form.saveSampleBtn.clicked.connect(self.save_sample_reconstruction)

        self.form.raysShow.stateChanged.connect(self.draw_rays)
        self.form.originalTraceHide.stateChanged.connect(self.draw_rays)
        self.form.rays.textChanged.connect(self.draw_rays)

        self.form.antialiasingBScan.stateChanged.connect(self.draw_bscan)

        self.form.openBScans.clicked.connect(self.load_assets)
        self.form.BScansAngle.currentIndexChanged.connect(self.draw_source_bscan)
        self.form.showHeatmap.stateChanged.connect(self.draw_reconstructed_sample)
        self.form.mergeResults.stateChanged.connect(self.draw_reconstructed_sample)
        self.form.antialiasingReconstruction.stateChanged.connect(self.draw_reconstructed_sample)

        self.set_params()

        #self.draw_source_bscan()

        self.error_dialog = QErrorMessage(self.window)
        self.error_dialog.setModal(True)
        self.error_dialog.setWindowTitle("OCT Reconstruction Error")

        self.oct_asset_list = OCTAssetList()

        self.legend = RILegendItem()
        self.legend.setPos(0, 520)
        self.ri_preview_scene.addItem(self.legend)

        self.form.openRIFile.clicked.connect(self.set_ri_path)

        self.form.genStep.clicked.connect(self.run_step)
        self.form.genRun.clicked.connect(self.run_steps)

        self.ri_gen_scene = QGraphicsScene()
        self.form.RIGenView.setScene(self.ri_gen_scene)
        self.ri_gen_pixmap = self.form.RIGenView.scene().addPixmap(QPixmap())
        self.legend_gen = RILegendItem()
        self.legend_gen.setPos(0, 520)
        self.ri_gen_scene.addItem(self.legend_gen)

    def set_ri_map(self):
        if self.form.sourceType.currentIndex() == 0:
            self.ri_map = RIMap.OCRTIterativeMap()
            self.ri_map.set_asset(self.form.riFile.text(), self.sample_h, self.sample_w, self.angle_dir)
        elif self.form.sourceType.currentIndex() == 1:
            self.ri_map = RIMap.OCRTImageMap()
            self.ri_map.set_image(self.form.riFile.text())
        elif self.form.sourceType.currentIndex() == 3:
            self.ri_map = RIMap.OCRTCapillaryV2()
        elif self.form.sourceType.currentIndex() == 4:
            self.ri_map = RIMap.OCRTMedium()
        else:
            self.ri_map = RIMap.OCRTLens()

        self.ri_map.set_params(self.x_step, self.z_step, self.nodes, self.sigma,
                               self.base_ri, self.sample_center_x, self.sample_center_z)

        self._N = []
        self._rays = []
        self.ri_scale = 512 / max(self.ri_map.X, self.ri_map.Z)

        if self.form.sourceType.currentIndex() == 0:
            return False

        self.form.angleList.clear()

        if self.angle_count > 1:
            angle = 0
            while angle < 360:
                self._N.append(self.ri_map.print(self.ri_scale, self.angle_dir * angle))
                self._rays.append(self.ri_map.propagate(self.angle_dir * angle))
                self.form.angleList.addItem(format(angle, '.2f'))
                angle = angle + self.angle_step
        else:
            self._N.append(self.ri_map.print(self.ri_scale))
            self._rays.append(self.ri_map.propagate())

        return True

    def draw_ri_map(self):
        channel_id = self.form.previewMode.currentIndex()
        angle_id = 0
        if self.angle_count > 1:
            angle_id = self.form.angleList.currentIndex()

        if angle_id == -1:
            return

        pixmap, min_v, max_v, self.offset_x, self.offset_z = OCTHelper.get_pixmap(self._N[angle_id], 512, 512,
                                                                                  channel_id)
        self.ri_pixmap.setPixmap(pixmap)
        self.legend.setVals(min_v, max_v)


    def draw_rays(self):
        for i in self.ri_original_rays:
            self.ri_preview_scene.removeItem(i)
        for i in self.ri_rays:
            self.ri_preview_scene.removeItem(i)

        self.ri_original_rays = []
        self.ri_rays = []

        angle_id = 0
        if self.angle_count > 1:
            angle_id = self.form.angleList.currentIndex()

        if angle_id == -1:
            return

        if self.form.raysShow.isChecked():
            ray_count = int(self.form.rays.text())

            ray_step = self.ri_map.X / (ray_count + 1)
            ray_start_x = (np.array(range(ray_count)) + 1) * ray_step

            if not self.form.originalTraceHide.isChecked():
                brush = QBrush(QColor(255, 255, 255))
                pen = QPen(brush, 1)
                for ray_x in ray_start_x:
                    line = QGraphicsLineItem(ray_x * self.ri_scale + self.offset_x, 0, ray_x * self.ri_scale + self.offset_x, 512)
                    line.setPen(pen)
                    self.ri_original_rays.append(line)
                    self.ri_preview_scene.addItem(line)

            brush = QBrush(QColor(0, 0, 0))
            pen = QPen(brush, 1)

            for ray_x in ray_start_x:
                ray = self._rays[angle_id].process_ray(ray_x, 1)

                path = QPainterPath()
                path.moveTo(ray[0, 1] * self.ri_scale + self.offset_x, 0)
                for step_i in range(1, self.ri_map.Z):
                    ray_z = ray[step_i, 0] * self.ri_scale
                    ray_x = ray[step_i, 1] * self.ri_scale
                    if ray_z > 0 and 0 <= ray_x + self.offset_x < 512:
                        path.lineTo(ray_x + self.offset_x, ray_z)
                path_item = QGraphicsPathItem(path)
                path_item.setPen(pen)
                self.ri_rays.append(path_item)
                self.ri_preview_scene.addItem(path_item)

    def get_bscan(self, angle_id: int, antialiasing: bool = False):
        bscan_mode = self.form.BScanBase.currentIndex()
        if bscan_mode == 0:
            bscan = abs(self._N[angle_id][:, :, 1])
        elif bscan_mode == 1:
            if len(self.form.riFile.text()) == 0:
                bscan = abs(self._N[angle_id][:, :, 1])
            else:
                img0 = cv2.imread(self.form.riFile.text())
                bscan = (1 + img0[:, :, 0] / 255) - self.base_ri
        elif bscan_mode == 2:
            bscan = self.get_dots_rect()
        else:
            bscan = self.get_dots_circle()

        return self._rays[angle_id].process_bscan(bscan, antialiasing)

    def draw_bscan(self):
        angle_id = 0
        if self.angle_count > 1:
            angle_id = self.form.angleList.currentIndex()

        if angle_id == -1:
            return

        bscan = self.get_bscan(angle_id, self.form.antialiasingBScan.isChecked())
        pixmap, min_v, max_v, offset_x, offset_z = OCTHelper.get_pixmap(bscan, 512, 512, 0, True)
        self.form.bscanPreview.setPixmap(pixmap)
        if self.oct_asset_list.count() == 0:
            self.draw_source_bscan()

    def process_map(self):
        self.set_params()
        if self.set_ri_map():
            self.draw_all()
            self.form.tabGen.setEnabled(False)
            self.form.tabOverview.setEnabled(True)
            self.form.tabReconstruct.setEnabled(True)
            self.form.tabWidget.setCurrentIndex(0)
        else:
            self.draw_step()
            self.form.tabWidget.setCurrentIndex(1)
            self.form.tabGen.setEnabled(True)
            self.form.tabOverview.setEnabled(False)
            self.form.tabReconstruct.setEnabled(False)

    def draw_all(self):
        self.draw_ri_map()
        self.draw_rays()
        self.draw_bscan()
        self.set_bscan_controls()
        self.draw_reconstructed_sample()

    def draw_step(self):
        pixmap, min_v, max_v, self.offset_x, self.offset_z = OCTHelper.get_pixmap(self.ri_map.print(self.ri_scale), 512, 512, 0)
        self.ri_gen_pixmap.setPixmap(pixmap)
        self.legend_gen.setVals(min_v, max_v)

        pixmap, min_v, max_v, offset_x, offset_z = OCTHelper.get_pixmap(self.ri_map.error, 512, 512, 0, True)
        self.form.RIErrorPreview.setPixmap(pixmap)

    def run_step(self):
        self.ri_map.iterate(float(self.form.alpha.text()))
        self.draw_step()

    def run_steps(self):
        N = i = int(self.form.stepsCount.text())
        while i > 0:
            print("Steps lefts: "+str(i))
            self.ri_map.iterate(float(self.form.alpha.text()))
            self.draw_step()

            fileName = "logs/step_" + str(N - i) + ".png"
            self.form.RIGenView.grab().save(fileName, "PNG")
            fileName = "logs/error_" + str(N - i) + ".png"
            self.form.RIErrorPreview.grab().save(fileName, "PNG")

            i = i - 1
            self.form.stepsCount.setText(str(i))

    def set_params(self):
        self.sigma = float(self.form.sigmaEdit.text())
        self.nodes = int(self.form.nodesEdit.text())
        self.x_step = int(self.form.XStepEdit.text())
        self.z_step = float(self.form.ZStepEdit.text())
        self.sample_center_x = float(self.form.XCenterEdit.text())
        self.sample_center_z = float(self.form.ZCenterEdit.text())
        self.sample_w = float(self.form.XSizemm.text())
        self.sample_h = float(self.form.ZSizemm.text())
        self.base_ri = float(self.form.baseRI.text())

        self.angle_count = int(self.form.angleCount.text())
        if self.angle_count > 1:
            self.angle_step = 360.0 / self.angle_count
        else:
            self.angle_step = 0
        self.angle_dir = 1.0 if self.form.angleDirection.currentIndex() == 0 else -1.0

    def reset_params(self):
        self.form.sigmaEdit.setText(str(self.sigma))
        self.form.nodesEdit.setText(str(self.nodes))
        self.form.XStepEdit.setText(str(self.x_step))
        self.form.XCenterEdit.setText(str(self.sample_center_x))
        self.form.ZCenterEdit.setText(str(self.sample_center_z))
        self.form.XSizemm.setText(str(self.sample_w))
        self.form.ZSizemm.setText(str(self.sample_h))
        self.form.baseRI.setText(str(self.base_ri))
        self.form.angleCount.setText(str(self.angle_count))
        self.form.angleDirection.setCurrentIndex(0 if self.angle_dir > 0 else 1)

    def save_ri(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self.window, "Save RI map", ".",
                                                  "PNG Files (*.png)", options=options)

        self.form.RIPreview.grab().save(fileName, "PNG")

    def save_bscan_sample(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self.window, "Save BScan reconstruction", ".",
                                                  "PNG Files (*.png)", options=options)

        self.form.bscanPreview.grab().save(fileName, "PNG")

    def save_sample_reconstruction(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self.window, "Save sample reconstruction", ".",
                                                  "PNG Files (*.png)", options=options)

        self.form.samplePreview.grab().save(fileName, "PNG")

    def draw_reconstructed_sample(self):
        if self.form.mergeResults.isChecked():
            return self.draw_merged_reconstructed_sample()

        if len(self._rays) > 0:

            angle_id = 0
            bscan_id = 0

            if self.oct_asset_list.count() > 0:
                if self.angle_count > 1:
                    if not self.angle_count == self.oct_asset_list.count():
                        self.error_dialog.showMessage('RI angles and OCT B-Scans count mismatch')
                        return
                    angle_id = self.form.BScansAngle.currentIndex()
                bscan_id = self.form.BScansAngle.currentIndex()
                bscan = self.oct_bscan_list[bscan_id, :, :]
                [Z, X] = bscan.shape
                #bscan = self.oct_asset_list.get_asset(bscan_id)
                #[X, Z] = bscan.shape
            else:
                if self.angle_count > 1:
                    angle_id = self.form.BScansAngle.currentIndex()
                if angle_id == -1:
                    return
                bscan = self.get_bscan(angle_id)
                [Z, X] = bscan.shape

            heatmap = self.form.showHeatmap.isChecked()
            rays = self._rays[angle_id]
            points = rays.get_reconstructed_points(bscan)
            img = OCTHelper.draw_bscan_points(X, X, points, self.form.antialiasingReconstruction.isChecked(), heatmap)
            pixmap, min_v, max_v, offset_x, offset_z = OCTHelper.get_pixmap(img, 512, 512, 0, not heatmap)
            self.form.samplePreview.setPixmap(pixmap)

    def draw_merged_reconstructed_sample(self):
        if len(self._rays) > 0 and (self.angle_count > 0 or self.oct_asset_list.count() > 1):
            points = []
            sample_center_z = self.ri_map.center_z
            sample_center_x = self.ri_map.center_x

            if self.oct_asset_list.count() > 0:
                if self.angle_count > 1:
                    if not self.angle_count == self.oct_asset_list.count():
                        self.error_dialog.showMessage('RI angles and OCT B-Scans count mismatch')
                        return

                rays = self._rays[0]
                bscan = self.oct_asset_list.get_asset(0)
                [X, Z] = bscan.get_image_size()

                imgs = np.zeros((self.oct_asset_list.count(), X, X), float)

                for bscan_id in range(self.oct_asset_list.count()):
                    if self.angle_count > 1:
                        rays = self._rays[bscan_id]
                    bscan = self.oct_asset_list.get_asset(bscan_id)
                    angle_points = rays.get_reconstructed_points(bscan, self.angle_dir * bscan.angle, sample_center_z,
                                                                 sample_center_x)
                    angle_bscan = OCTHelper.draw_bscan_points(X, X, angle_points, self.form.antialiasingReconstruction.isChecked())
                    imgs[bscan_id, :, :] = angle_bscan[:, :]
                    #points = points + angle_points
            else:
                bscan = self.get_bscan(0)
                Z = bscan.shape[0]
                X = bscan.shape[1]

                imgs = np.zeros((self.angle_count, X, X), float)

                for angle_id in range(self.angle_count):
                    rays = self._rays[angle_id]
                    bscan = self.get_bscan(angle_id)
                    angle_points = rays.get_reconstructed_points(bscan, self.angle_dir * self.angle_step * angle_id, sample_center_z,
                                                           sample_center_x)
                    angle_bscan = OCTHelper.draw_bscan_points(X, X, angle_points, self.form.antialiasingReconstruction.isChecked())
                    imgs[angle_id, :, :] = angle_bscan[:, :]
                    #points = points + angle_points

            heatmap = self.form.showHeatmap.isChecked()

            img = np.zeros([X, X])

            for _x in range(X):
                for _z in range(X):
                    img[_z, _x] = imgs[:, _z, _x].mean()

            #img = OCTHelper.draw_bscan_points(Z, X, points, False)
            pixmap, min_v, max_v, offset_x, offset_z = OCTHelper.get_pixmap(img, 512, 512, 0, not heatmap)
            self.form.samplePreview.setPixmap(pixmap)

    def draw_angle_sample(self):
        if self.angle_count > 1:
            if self.old_angle_idx != self.form.angleList.currentIndex():
                self.old_angle_idx = self.form.angleList.currentIndex()
                self.draw_ri_map()
                self.draw_rays()
                self.draw_bscan()
        else:
            self.old_angle_idx = 0

    def get_dots_rect(self):
        dot_step = 8
        bscan = np.zeros((self.ri_map.Z, self.ri_map.X, 1), np.uint8)

        x = dot_step - int(round(self.ri_map.center_x - (self.ri_map.center_x // dot_step)*dot_step))
        _z = dot_step - int(round(self.ri_map.center_z - (self.ri_map.center_z // dot_step)*dot_step))

        while x < self.ri_map.X:
            z = _z
            while z < self.ri_map.Z:
                bscan[z, x, 0] = 255
                z = z + dot_step
            x = x + dot_step
        return bscan

    def get_dots_circle(self):
        dot_step = 8
        bscan = np.zeros((self.ri_map.Z, self.ri_map.X, 1), np.uint8)

        r = dot_step
        bscan[self.ri_map.center_z, self.ri_map.center_x, 0] = 255
        while self.ri_map.center_x + r < self.ri_map.X or 0 <= self.ri_map.center_x - r or self.ri_map.center_z + r < self.ri_map.Z or 0 <= self.ri_map.center_z - r:
            dots = math.floor(2 * math.pi * r / dot_step)
            if dots > 0:
                angle_step = 2 * math.pi / dots
                for i in range(dots):
                    z = round(r * math.sin(angle_step * i)) + self.ri_map.center_z
                    x = round(r * math.cos(angle_step * i)) + self.ri_map.center_x
                    if 0 <= x < self.ri_map.X and 0 <= z < self.ri_map.Z:
                        bscan[z, x, 0] = 255
            r = r + dot_step

        return bscan

    def load_assets(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.window, "Load dataset", ".",
                                                  "All Files (*)", options=options)
        if fileName:
            self.form.BScansPath.setText(fileName)
            self.load_dataset()

    def load_dataset(self):
        self.oct_asset_list = OCTAssetList()
        dataset_fname = self.form.BScansPath.text()
        if len(dataset_fname) == 0 or self.oct_asset_list.load(dataset_fname) == False:
            self.error_dialog.showMessage('Invalid datasource name')
        else:
            self.oct_asset_list.set_oct_scale(self.sample_h, self.sample_h)
            self.oct_bscan_list = self.oct_asset_list.get_bscans()
            self.form.BScansAngle.clear()
            for asset in self.oct_asset_list.assets:
                self.form.BScansAngle.addItem(str(asset.angle))
            self.form.BScansAngle.setCurrentIndex(0)

    def draw_source_bscan(self):
        asset_id = 0

        if self.oct_asset_list.count() > 0:
            asset_id = self.form.BScansAngle.currentIndex()
            if asset_id == -1:
                return
            #pixmap = self.oct_asset_list.get_asset(asset_id).get_pixmap(512, 512)
            pixmap, _min_v, _max_v, _offset_x, _offset_z = OCTHelper.get_pixmap(self.oct_bscan_list[asset_id, :, :], 512, 512, 0, True)
        else:
            if self.angle_count > 0:
                asset_id = self.form.BScansAngle.currentIndex()
            if asset_id == -1:
                return
            bscan = self.get_bscan(asset_id)
            pixmap, _min_v, _max_v, _offset_x, _offset_z = OCTHelper.get_pixmap(bscan, 512, 512, 0, True)

        self.form.bscanSource.setPixmap(pixmap)
        if not self.form.mergeResults.isChecked():
            self.draw_reconstructed_sample()


    def set_bscan_controls(self):
        if self.oct_asset_list.count() == 0:
            self.form.BScansAngle.clear()
            if self.angle_count > 0:
                for i in range(self.angle_count):
                    self.form.BScansAngle.addItem(format(i * self.angle_step, '.2f'))

    def set_ri_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.window, "Load RI source", ".",
                                                  "All Files (*)", options=options)
        if fileName:
            self.form.riFile.setText(fileName)


if __name__ == '__main__':
    print(os.getcwd())

    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = OCTRIMapUI()
    ui.init(window)
    window.show()

    sys.exit(app.exec_())