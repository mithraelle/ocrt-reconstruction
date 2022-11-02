import os
import re
import numpy as np
import cv2
import scipy.io
import math
import OCTFeature
from PyQt5.QtGui import QPixmap, QImage, QColor

import time

import OCTHelper

def from_image(file: str, angle: float):
    img = cv2.imread(file)
    return OCTAsset(img, angle)


def from_mat_array(data: [float], angle: float):
    data = np.round((data + 1) * 128).clip(0, 255).astype("uint8")
    data = np.transpose(data)
    data = np.dstack([data, data, data])
    return OCTAsset(data, angle)


class OCTAsset:
    imageData = []
    angle: float = 0

    def __init__(self, image_data, angle: float):
        self.imageData = image_data
        self.angle = angle

    def get_image_size(self):
        h, w = self.imageData.shape[:2]
        return [w, h]

    def get_pixmap(self, target_w: int = 0, target_h: int = 0):
        pixmap, min_v, max_v, offset_x, offset_z = OCTHelper.get_pixmap(self.imageData, target_w, target_h, 0, True)
        return pixmap


def get_oct_asset_key(e: OCTAsset):
    return e.angle


class OCTAssetList:
    assets = []

    scale_x = 1
    scale_z = 1

    center_x = 0
    center_y = 0

    def add(self, oct_asset: OCTAsset):
        self.assets.append(oct_asset)

    def get_asset(self, idx: int):
        if 0 <= idx < len(self.assets):
            return self.assets[idx]
        return False

    def get_asset_by_angle(self, angle: float):
        tolerance = 360 / len(self.assets)
        for asset in self.assets:
            if abs(asset.angle - angle) <= tolerance:
                return asset
        return False

    def load(self, dataset_filename: str):
        if dataset_filename.lower().find(".jpg") != -1:
            return self.load_jpgs(os.path.dirname(dataset_filename))
        if dataset_filename.lower().find(".png") != -1:
            return self.load_pngs(os.path.dirname(dataset_filename))
        elif dataset_filename.lower().find(".bmp") != -1:
            return self.load_bmps(os.path.dirname(dataset_filename))
        elif dataset_filename.lower().find(".mat") != -1:
            return self.load_mat(dataset_filename)

        return False

    def load_jpgs(self, img_dir: str):
        self.assets.clear()
        files = os.listdir(img_dir)
        for file in files:
            src = os.path.join(img_dir, file)
            if os.path.isfile(src):
                angle = re.findall(r'(\d+)\.jpg', file, re.IGNORECASE)
                if len(angle) > 0:
                    self.add(from_image(src, float(angle[0])))
        self.assets.sort(key=get_oct_asset_key)

    def load_pngs(self, img_dir: str):
        self.assets.clear()
        files = os.listdir(img_dir)
        for file in files:
            src = os.path.join(img_dir, file)
            if os.path.isfile(src):
                angle = re.findall(r'(\d+)\.png', file, re.IGNORECASE)
                if len(angle) > 0:
                    self.add(from_image(src, float(angle[0])))
        self.assets.sort(key=get_oct_asset_key)

    def load_bmps(self, img_dir: str):
        self.assets.clear()
        files = os.listdir(img_dir)
        for file in files:
            src = os.path.join(img_dir, file)
            if os.path.isfile(src):
                angle = re.findall(r'(\d+)\.bmp', file, re.IGNORECASE)
                if len(angle) > 0:
                    self.add(from_image(src, float(angle[0])))
        self.assets.sort(key=get_oct_asset_key)

    def load_mat(self, filename: str):
        self.assets.clear()
        mat = scipy.io.loadmat(filename)
        scans_count = mat['Bscans'].shape[0]
        print(mat['Bscans'].shape)
        angle = 360 / scans_count
        for i in range(0, scans_count):
            self.add(from_mat_array(mat['Bscans'][i, :, :], angle * i))

    def evaluate_feature_pairs(self, feature_detector):
        samples = math.floor(len(self.assets) / 2)

        feature_match_list = OCTFeature.OCTFeatureMatchSetList()

        total_time = 0
        for sample_i in range(len(self.assets)):
            img0 = self.get_asset(sample_i).imageData
            next_sample = (sample_i + 1) if sample_i + 1 < len(self.assets) else 0
            img1 = self.get_asset(next_sample).imageData

            st = time.time()
            matches, kp0, kp1 = feature_detector(img0, img1)
            total_time = total_time + (time.time()-st)
            feature_set = OCTFeature.OCTFeatureMatchSet(sample_i, next_sample)
            for i in matches:
                if len(i) == 2:
                    idx0 = i[0].queryIdx
                    idx1 = i[0].trainIdx

                    query_point = np.asarray(kp0[idx0].pt)
                    train_point = np.asarray(kp1[idx1].pt)
                    if i[0].distance < 0.7 * i[1].distance:
                        feature_set.add_feature(query_point, train_point)

            feature_match_list.append(feature_set)

        print("Total Feature Detection Time: ", total_time)
        print("Per Pair Feature Detection Time: ", total_time / len(self.assets))

        return feature_match_list

    def get_image_size(self):
        return self.get_asset(0).get_image_size()

    def set_oct_scale(self, z: float, x: float):
        self.scale_x = x
        self.scale_z = z

    def set_image_center(self, x: float, y: float):
        self.center_x = x
        self.center_y = y

    def get_bscans(self):
        img_size = self.get_image_size()
        res_w = img_size[0]
        if self.scale_x != 0 and self.scale_z != 0:
            res_h = round(img_size[0] * self.scale_z / self.scale_x)
        else:
            res_h = img_size[1]

        imgs = np.zeros((self.count(), res_h, res_w), float)
        i = 0
        for asset in self.assets:
            if self.scale_x != 0 and self.scale_z != 0:
                imgs[i, :, :] = cv2.resize(asset.imageData, (res_w, res_h), interpolation=cv2.INTER_CUBIC)[:, :, 0]
            else:
                imgs[i, :, :] = asset.imageData[:, :, 0]

            max_v = imgs[i, :, :].max().astype(float)
            imgs[i, :, :] = imgs[i, :, :] / max_v
            i = i + 1
        return imgs

    def get_average_image(self):
        img_size = self.get_image_size()
        scale = min(self.scale_z, self.scale_x)
        rescale_x = self.scale_x / scale
        rescale_y = self.scale_z / scale
        res_w = img_size[0] * rescale_x
        res_h = img_size[1] * rescale_y

        res = np.array((res_w, res_h, len(self.assets)), np.uint8)
        i = 0
        for asset in self.assets:
            img = cv2.resize(asset.imageData, (res_w, res_h), interpolation=cv2.INTER_CUBIC)
            M = cv2.getRotationMatrix2D((self.center_x, self.center_y), asset.angle, 1)
            img = cv2.warpAffine(img, M, (res_h, res_w))
            res[:, :, i] = img[:, :, 0]
            i += 1

    def count(self):
        return len(self.assets)