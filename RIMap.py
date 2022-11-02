import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from random import *

import OCTHelper
from OCTAsset import OCTAsset, OCTAssetList

OCRT_IMAGE_SIZE = 512

class RIMap:
    x_step = 4
    z_step = 1
    N = 1
    sigma = 8

    base_ri = 1
    center_z = 0
    center_x = 0
    rays = []

    X = 0
    Z = 0

    def set_params(self, x_step, z_step, N, sigma, base_ri, center_x, center_z):
        self.x_step = x_step
        self.z_step = z_step
        self.N = N
        self.sigma = sigma

        self.base_ri = base_ri

        self.center_z = center_z
        self.center_x = center_x

    def rk4_step_calc(self, z, x, t):
        dxdz = x[:, 1]
        _x = t[0, 0] * x[:, 0] + t[0, 1] * z + t[0, 2]
        _z = t[1, 0] * x[:, 0] + t[1, 1] * z + t[1, 2]
        n = np.array(list(map(self.get_ref_index_params, _x, _z)))

        dndz = t[0, 0] * n[:, 1] + t[0, 1] * n[:, 2]
        dndx = t[1, 0] * n[:, 1] + t[1, 1] * n[:, 2]

        dxdz2 = (dndx * (1. + dxdz ** 2) - dndz * dxdz) / n[:, 0]
        return np.stack((dxdz, dxdz2), -1)

    def get_ref_index_params(self, x, z):
        #_x = round(x)
        #_z = round(z)

        if self.N > 0:
            n = self.N - 1
            i_min = math.floor(x - n)
            i_max = math.ceil(x + n) + 1
            j_min = math.floor(z - n)
            j_max = math.ceil(z + n) + 1
        else:
            i_min = 0
            i_max = self.X
            j_min = 0
            j_max = self.Z
        #i_min = max(0, _x - self.N)
        #i_max = min(self.X, _x + self.N + 1)
        #j_min = max(0, _z - self.N)
        #j_max = min(self.Z, _z + self.N + 1)

        if x >= self.X or z >= self.Z or x < 0 or z < 0:
            return [self.base_ri, 0, 0]

        f = 0
        g = 0
        df = np.array([0.0, 0.0])
        dg = np.array([0.0, 0.0])
        sigma2 = self.sigma * self.sigma
        double_sigma2 = 2 * sigma2
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                l = math.exp(-((z - j) ** 2 + (x - i) ** 2) / double_sigma2) + 2e-7
                if i >= self.X or j >= self.Z or i < 0 or j < 0:
                    n = self.base_ri
                else:
                    n = self._N[j, i]
                delta_f = n * l
                delta_g = l
                f = f + delta_f
                g = g + delta_g
                dl = np.array([j - z, i - x]) / sigma2
                df = df + delta_f * dl
                dg = dg + delta_g * dl

        dn = (df * g - f * dg) / (g ** 2)
        return [f / g, dn[0], dn[1]]

    def print(self, scale: float = 1.0, angle: float = 0.0):
        w = int(round(self.Z * scale))
        h = int(round(self.X * scale))
        _N = np.zeros([w, h, 3])
        t = cv2.getRotationMatrix2D((self.center_x, self.center_z), angle, 1)
        for i in range(w):
            for j in range(h):
                x = j / scale
                z = i / scale
                #_v = np.dot(t, (x, z, 1))
                #_N[i, j] = self.get_ref_index_params(_v[0], _v[1])
                _vx = t[0, 0] * x + t[0, 1] * z + t[0, 2]
                _vz = t[1, 0] * x + t[1, 1] * z + t[1, 2]
                _N[i, j] = self.get_ref_index_params(_vx, _vz)
        return _N

    def get_init_rays(self):
        rays = np.zeros([self.X, 3])
        rays[:, 1] = range(self.X)[:]
        return rays

    def propagate(self, angle: float = 0.0):
        steps = math.floor(self.Z / self.z_step) - 1

        init_rays = self.get_init_rays()
        rays = np.zeros([steps+1, init_rays.shape[0], 3])
        rays[0, :, :] = init_rays
        #rays = self.get_init_rays()
        #rays = np.expand_dims(rays, axis=0)

        t = cv2.getRotationMatrix2D([self.center_x, self.center_z], angle, 1)

        for i in range(steps):
            if i % 1000 == 0:
                print(round(100*i/steps, 3), '%')

            z0 = rays[i, :, 0]
            x0 = rays[i, :, 1:]

            _x0 = t[0, 0] * x0[:, 0] + t[0, 1] * z0 + t[0, 2]
            _z0 = t[1, 0] * x0[:, 0] + t[1, 1] * z0 + t[1, 2]

            n = np.array(list(map(self.get_ref_index_params, _x0, _z0)))
            z_step = self.z_step / n[:, 0]
            step2 = z_step / 2

            hk1 = self.rk4_step_calc(z0, x0, t)
            hk1[:, 0] = hk1[:, 0] * z_step[:]
            hk1[:, 1] = hk1[:, 1] * z_step[:]

            z2 = z0 + step2
            x2 = x0 + hk1 / 2
            hk2 = self.rk4_step_calc(z2, x2, t)
            hk2[:, 0] = hk2[:, 0] * z_step[:]
            hk2[:, 1] = hk2[:, 1] * z_step[:]

            x3 = x0 + hk2 / 2
            hk3 = self.rk4_step_calc(z2, x3, t)
            hk3[:, 0] = hk3[:, 0] * z_step[:]
            hk3[:, 1] = hk3[:, 1] * z_step[:]

            z4 = z0 + z_step[:]
            x4 = x0 + hk3
            hk4 = self.rk4_step_calc(z4, x4, t)
            hk4[:, 0] = hk4[:, 0] * z_step[:]
            hk4[:, 1] = hk4[:, 1] * z_step[:]

            xi = x0 + (hk1 + 2 * hk2 + 2 * hk3 + hk4) / 6.
            z4 = np.expand_dims(z4, -1)
            zx = np.concatenate((z4, xi), axis=1)
            #rays = np.append(rays, np.expand_dims(zx, axis=0), axis=0)
            rays[i+1, :, :] = zx

        return RayPropagationResult(self.Z, self.X, rays, self.z_step)


class OCRTMedium(RIMap):
    def set_params(self, x_step, z_step, N, sigma, base_ri, center_x, center_z):
        super().set_params(x_step, z_step, N, sigma, base_ri, center_x, center_z)
        self.Z = 512
        self.X = 512

    def get_ref_index_params(self, x, z):
        return [self.base_ri, 0, 0]

    def get_init_rays(self):
        ray_count = math.floor(self.X / self.x_step) + 1
        rays = np.zeros([ray_count, 3])
        rays[:, 1] = range(ray_count)[:]
        rays[1:, 1] = rays[1:, 1] * self.x_step
        return rays


class OCRTLens(RIMap):
    #delta = 0.0001
    delta = 1

    _z1 = -54#-69
    _x1 = 75
    _z2 = 216#181
    _x2 = 75
    _r1 = 160
    _r2 = 160

    max_x = 150
    start_x = 0
    original_x = 75
    original_z = 81#56

    def set_params(self, x_step, z_step, N, sigma, base_ri, center_x, center_z):
        super().set_params(x_step, z_step, N, sigma, base_ri, center_x, center_z)

        self.center_x = self.original_x
        self.center_z = self.original_z

        self.start_x = self.x_step - int(round(self.center_x - (self.center_x // self.x_step)*self.x_step))
        if self.start_x % self.x_step == 0:
            self.start_x = 0
        self.center_x = self.center_x + self.start_x

        self._x1 = self.center_x
        self._x2 = self.center_x
        self.Z = 512
        self.X = self.max_x + 2*self.start_x
        self.delta = self.z_step

    def get_n(self, x, z):
        n = self.base_ri
        if self.start_x <= x < self.start_x + self.max_x:
            z1 = self._z1 + math.sqrt(self._r1 ** 2 - (x - self._x1) ** 2)
            z2 = self._z2 - math.sqrt(self._r2 ** 2 - (x - self._x2) ** 2)
            if z1 >= z >= z2:
                n = 1.4
        return n

    def get_ref_index_params(self, x, z):
        n = self.get_n(x, z)
        delta2 = 2 * self.delta
        dndz = (self.get_n(x, z+self.delta) - self.get_n(x, z-self.delta)) / delta2
        dndx = (self.get_n(x+self.delta, z) - self.get_n(x-self.delta, z)) / delta2
        return [n, dndz, dndx]

    def print(self, scale: float = 1.0, angle: float = 0.0):
        img = np.zeros((512, self.X, 3), np.float)
        img[:, :, 0] = self.base_ri

        t = cv2.getRotationMatrix2D((self.center_x, self.center_z), angle, 1)

        for z in range(512):
            for x in range(self.start_x, self.max_x + self.start_x):
                _v = np.dot(t, (x, z, 1))
                img[z, x, :] = self.get_ref_index_params(_v[0], _v[1])[:]
        """
        for x in range(self.start_x, self.max_x + self.start_x):
            z1 = self._z1 + math.sqrt(self._r1 ** 2 - (x - self._x1) ** 2)
            z = int(round(z1))
            img[z, x, 1:3] = self.get_ref_index_params(x, z1)[1:3]

            z2 = self._z2 - math.sqrt(self._r2 ** 2 - (x - self._x2) ** 2)
            z = int(round(z2))
            img[z, x, 1:3] = self.get_ref_index_params(x, z2)[1:3]
        """
        return img

    def get_init_rays(self):
        ray_count = math.floor(self.X / self.x_step) + 1
        rays = np.zeros([ray_count, 3])
        rays[:, 1] = range(ray_count)[:]
        rays[1:, 1] = rays[1:, 1] * self.x_step
        return rays


class OCRTCapillary(OCRTLens):
    _z1 = 180
    _x1 = 200
    _z2 = 180
    _x2 = 200
    _r1 = 164
    _r2 = 122

    max_x = 400
    start_x = 0
    original_x = 200
    original_z = 180

    glass_ri = 1.55
    oil_ri = 1.42

    def get_n(self, x, z):
        n = self.base_ri
        if self.start_x <= x < self.start_x + self.max_x:
            if math.sqrt((x - self._x2) ** 2 + (z - self._z2) ** 2) < self._r2:
                n = self.oil_ri
            elif math.sqrt((x - self._x1) ** 2 + (z - self._z1) ** 2) < self._r1:
                n = self.glass_ri
        return n

    def print(self, scale: float = 1.0, angle: float = 0.0):
        img = np.zeros((512, self.X, 3), np.float)
        img[:, :, 0] = self.base_ri

        t = cv2.getRotationMatrix2D((self.center_x, self.center_z), angle, 1)

        for z in range(512):
            for x in range(self.start_x, self.max_x + self.start_x):
                _v = np.dot(t, (x, z, 1))
                img[z, x, :] = self.get_ref_index_params(_v[0], _v[1])[:]

        for x in range(self._x1 - self._r1, self._x1 + self._r1):
            z1 = self._z1 + math.sqrt(self._r1 ** 2 - (x - self._x1) ** 2)
            z = int(round(z1))
            img[z, x, 1:3] = self.get_ref_index_params(x, z1)[1:3]

            z2 = self._z1 - math.sqrt(self._r1 ** 2 - (x - self._x1) ** 2)
            z = int(round(z2))
            img[z, x, 1:3] = self.get_ref_index_params(x, z2)[1:3]

        for x in range(self._x2 - self._r2, self._x2 + self._r2):
            z1 = self._z2 + math.sqrt(self._r2 ** 2 - (x - self._x2) ** 2)
            z = int(round(z1))
            img[z, x, 1:3] = self.get_ref_index_params(x, z1)[1:3]

            z2 = self._z2 - math.sqrt(self._r2 ** 2 - (x - self._x2) ** 2)
            z = int(round(z2))
            img[z, x, 1:3] = self.get_ref_index_params(x, z2)[1:3]


        return img


class OCRTImageMap(RIMap):
    image_file: str

    def set_image(self, img_path: str):
        self.image_file = img_path

    def set_params(self, x_step, z_step, N, sigma, base_ri, center_x, center_z):
        super().set_params(x_step, z_step, N, sigma, base_ri, center_x, center_z)

        img0 = cv2.imread(self.image_file)
        [h, w] = img0.shape[:2]

        if self.center_x == 0:
            self.center_x = math.floor(w / 2)

        if self.center_z == 0:
            self.center_z = math.floor(h / 2)

        nodes_left = math.floor(self.center_x / self.x_step)
        start_x = round(self.center_x - nodes_left * self.x_step)
        nodes_right = math.floor((w - self.center_x) / self.x_step)

        nodes_top = math.floor(self.center_z / self.x_step)
        start_z = round(self.center_z - nodes_top * self.x_step)
        nodes_bottom = math.floor((h - self.center_z) / self.x_step)

        self.X = nodes_left + nodes_right
        if start_x > 0:
            self.X = self.X + 1
        if nodes_right * self.x_step + self.center_x > w:
            self.X = self.X + 1

        self.Z = nodes_top + nodes_bottom
        if start_z > 0:
            self.Z = self.Z + 1
        if nodes_bottom * self.x_step + self.center_z > h:
            self.Z = self.Z + 1

        self._N = np.zeros((self.Z, self.X), float)
        self._N[:, :] = self.base_ri

        for x in range(self.X):
            for z in range(self.Z):
                i = start_x + x * self.x_step
                j = start_z + z * self.x_step
                if i < w and j < h:
                    self._N[z, x] = 1 + (img0[j, i, 0] / 255)

        self.center_z = nodes_top
        self.center_x = nodes_left

class OCRTIterativeMap(RIMap):
    bscan_list: []
    direction: float = 1.0
    angle: float = 0.0
    _rays: []
    error: []
    _Nold: []
    errorOld: []

    def set_asset(self, asset_path: str, scan_z: float, scan_x: float, direction: float):
        asset_list = OCTAssetList()
        asset_list.load(asset_path)
        asset_list.set_oct_scale(scan_z, scan_x)
        self.bscan_list = asset_list.get_bscans()
        self.direction = direction
        self.angle = 360 / asset_list.count()
        self._Nold = []
        self.errorOld = []


    def set_params(self, x_step, z_step, N, sigma, base_ri, center_x, center_z):
        super().set_params(x_step, z_step, N, sigma, base_ri, center_x, center_z)

        [h, w] = self.bscan_list.shape[1:3]

        if self.center_x == 0:
            self.center_x = math.floor(w / 2)

        if self.center_z == 0:
            self.center_z = math.floor(h / self.base_ri / 2)

        nodes_left = math.floor(self.center_x / self.x_step)
        start_x = round(self.center_x - nodes_left * self.x_step)
        nodes_right = math.floor((w - self.center_x) / self.x_step)

        nodes_top = math.floor(self.center_z / self.x_step)
        start_z = round(self.center_z - nodes_top * self.x_step)
        nodes_bottom = math.floor((h - self.center_z) / self.x_step)

        self.X = nodes_left + nodes_right
        if start_x > 0:
            self.X = self.X + 1
        if nodes_right * self.x_step + self.center_x > w:
            self.X = self.X + 1

        self.Z = nodes_top + nodes_bottom
        if start_z > 0:
            self.Z = self.Z + 1
        if nodes_bottom * self.x_step + self.center_z > h:
            self.Z = self.Z + 1

        self._N = np.zeros((self.Z, self.X), float)
        self._N[:, :] = self.base_ri

        self.center_z = nodes_top
        self.center_x = nodes_left
        self.evaluate_error()

    def evaluate_error(self):
        [n, _Z, _X] = self.bscan_list.shape
        self._rays = []
        _reconstructed = np.zeros((n, _Z, _X), float)
        for i in range(n):
            angle = self.angle * i
            self._rays.append(self.propagate(self.direction * angle))
            _reconstructed[i, :, :] = OCTHelper.draw_bscan_points(_Z, _X,
                self._rays[i].get_reconstructed_points(self.bscan_list[i, :, :], self.direction * angle, self.center_z,
                                                                 self.center_x))
        self.error = np.zeros([_Z, _X], float)

        for _x in range(_X):
            for _z in range(_Z):
                self.error[_z, _x] = np.std(_reconstructed[:, _z, _x])

    def adjustN(self, z, x, de, dN, alpha):
        if self.N > 0:
            n = self.N - 1
            i_min = math.floor(x - n)
            i_max = math.ceil(x + n) + 1
            j_min = math.floor(z - n)
            j_max = math.ceil(z + n) + 1
        else:
            i_min = 0
            i_max = self.X
            j_min = 0
            j_max = self.Z

        if x >= self.X or z >= self.Z or x < 0 or z < 0:
            return

        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                if i >= self.X or j >= self.Z or j < 0 or i < 0:
                    continue
                self._N[j, i] = self._N[j, i] - alpha * de * dN[j, i]

    def iterate(self, alpha: float):
        if len(self._Nold) == 0:
            self._Nold = self._N
            self._N = self._N + alpha*np.random.random((self.Z, self.X))
        else:
            de = self.error - self.errorOld
            dN = np.sign(self._N - self._Nold)
            self._Nold = np.array(self._N)
            [_Z, _X] = de.shape
            scale = self.X / _X
            for _x in range(_X):
                for _z in range(_Z):
                    self.adjustN(_z * scale, _x * scale, de[_z, _x], dN, alpha)

        self.errorOld = self.error
        self.evaluate_error()


class OCRTCapillaryV2(OCRTCapillary):
    _z1 = 177
    _x1 = 196
    _z2 = 177
    _x2 = 196
    _r1 = 165
    _r2 = 121

    max_x = 400
    start_x = 0

    original_x = 196
    original_z = 177

    glass_ri = 1.54

class RayPropagationResult:
    X: int
    Z: int
    z_step: float
    rays = []

    def __init__(self, Z: int, X: int, rays, z_step: float):
        self.X = X
        self.Z = Z
        self.rays = rays
        self.z_step = z_step

    def process_ray(self, x, z_scale):
        steps = math.ceil(self.Z / z_scale) + 1
        max_i = self.rays.shape[1]
        max_j = self.rays.shape[0]
        max_x = self.rays[0, max_i-1, 1]
        z = 0

        ray = np.zeros([steps, 2])
        delta_j = max_j / steps

        if max_x < x or x < 0:
            ray[:, 1] = x
            ray[:, 0] = range(steps)
            ray[:, 0] = ray[:, 0] * z_scale
            return ray

        ray[0, 1] = x

        for i in range(max_i-1):
            if abs(self.rays[0, i, 1] - x) < 0.000001: #exact on the ray
                for step in range(1, steps):
                    z = step * delta_j
                    left_j = math.floor(z)
                    right_j = math.ceil(z)
                    if left_j == right_j or right_j == max_j: #exact on the node or right to the last ray
                        ray[step] = self.rays[left_j, i, 0:2]
                    else:
                        c = z - left_j
                        ray[step] = self.rays[left_j, i, 0:2] + c * (
                                    self.rays[right_j, i, 0:2] - self.rays[left_j, i, 0:2])
                return ray
            elif self.rays[0, i, 1] < x < self.rays[0, i+1, 1]:
                left_i = i
                right_i = i + 1
                c_x = (x - self.rays[0, left_i, 1]) / (self.rays[0, right_i, 1] - self.rays[0, left_i, 1])
                for step in range(1, steps):
                    z = step * delta_j
                    left_j = math.floor(z)
                    right_j = math.ceil(z)
                    if left_j == right_j or right_j == max_j: #exact on the node or right to the last ray
                        ray[step] = self.rays[left_j, left_i, 0:2] + c_x * (
                                    self.rays[left_j, right_i, 0:2] - self.rays[left_j, left_i, 0:2])
                    else: #between rays and iteration nodes
                        c_z = z - left_j
                        left = self.rays[left_j, left_i, 0:2] + c_x * (
                                    self.rays[left_j, right_i, 0:2] - self.rays[left_j, left_i, 0:2])
                        right = self.rays[right_j, left_i, 0:2] + c_x * (
                                    self.rays[right_j, right_i, 0:2] - self.rays[right_j, left_i, 0:2])
                        ray[step] = left + c_z * (right - left)
                return ray
        return ray

    def process_bscan(self, bscan, antialiasing: bool = False):
        Z = bscan.shape[0]
        X = bscan.shape[1]
        res = np.zeros([Z, X])

        scale = self.X / X

        for x in range(X):
            ray = self.process_ray(x*scale, scale)
            for z in range(Z):
                c = 0
                _z = ray[z, 0] / scale
                _x = ray[z, 1] / scale
                if 0 <= round(_x) < X and 0 <= round(_z) < Z:
                    if not antialiasing:
                        c = bscan[round(_z), round(_x)]
                    else:
                        c = OCTHelper.get_pixel(_x, _z, bscan)
                    res[z, x] = c + res[z, x]
        return res

    def reconstruct_bscan(self, bscan, antialiasing: bool = False):
        Z = bscan.shape[0]
        X = bscan.shape[1]
        res = np.zeros([Z, X])

        x_scale = self.X / X

        for x in range(X):
            ray = self.process_ray(x*x_scale, 1)
            for z in range(Z):
                _z = round(ray[z, 0])
                _x = round(ray[z, 1] / x_scale)
                if 0 <= _x < X and 0 <= _z < Z and res[_z, _x] < 256:
                    c = bscan[z, x]
                    res[_z, _x] = c + res[_z, _x]
        return res

    def get_reconstructed_points(self, bscan, angle: float = 0, center_z: float = 0, center_x: float = 0, normalize: bool = True):
        if normalize:
            oct = isinstance(bscan, OCTAsset)
            if oct:
                [X, Z] = bscan.get_image_size()
                max_v = bscan.imageData[:, :, 0].max().astype(float)
                norm_bscan = bscan.imageData[:, :, 0] / max_v
            else:
                Z = bscan.shape[0]
                X = bscan.shape[1]
                max_v = bscan.max()
                norm_bscan = bscan / max_v
        else:
            Z = bscan.shape[0]
            X = bscan.shape[1]
            norm_bscan = bscan

        res = []
        x_scale = self.X / X

        t = []
        if angle != 0:
            t = cv2.getRotationMatrix2D([center_x / x_scale, center_z / x_scale], angle, 1)

        for x in range(X):
            ray = self.process_ray(x*x_scale, x_scale)
            max_z = len(ray)
            for z in range(Z):
                if z >= max_z:
                    continue
                c = norm_bscan[z, x]

                _z = ray[z, 0] / x_scale
                _x = ray[z, 1] / x_scale

                if angle != 0:
                    _x, _z = np.dot(t, (_x, _z, 1))

                if c > 0 and 0 <= _x < X and 0 <= _z < Z:
                    res.append([_z, _x, c])
        return res
