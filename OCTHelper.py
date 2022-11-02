import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QColor
import cv2
import math

def get_qcolor(c, rgb: bool = True):
    a = c / 255.0
    if rgb:
        color = QColor.fromRgbF(a, a, a)
    else:
        color = QColor.fromHsv((1-a) * 240.0, 255, 255)

    return color


def get_color(c, rgb: bool = True):
    return get_qcolor(c, rgb).rgb()


def get_pixmap(img, target_w: int, target_h: int, channel: int = 0, rgb: bool = False):
    img_w = img.shape[1]
    img_h = img.shape[0]

    if len(img.shape) < 3:
        channel = 0
    else:
        img = img[:, :, channel]

    min_v = img.min()
    max_v = img.max()
    if not min_v == max_v:
        img = np.round((img - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    else:
        img = np.round(img / min_v * 255.0).astype(np.uint8)
    img = np.dstack([img, img, img])

    res_w = img_w
    res_h = img_h
    img_ar = img_w / img_h
    offset_w = 0
    offset_h = 0
    resize = True

    if target_w == 0 and target_h == 0:
        target_w = img_w
        target_h = img_h
        resize = False

    if resize:
        target_ar = target_w / target_h

        if img_ar >= target_ar:
            res_w = target_w #min(target_w, img_w)
            res_h = res_w / img_ar
        else:
            res_h = target_h #min(target_h, img_h)
            res_w = res_h * img_ar

        res_w = round(res_w)
        res_h = round(res_h)

        offset_w = (target_w - res_w) // 2
        offset_h = (target_h - res_h) // 2

        to_draw = cv2.resize(img, (res_w, res_h), interpolation=cv2.INTER_AREA)
    else:
        to_draw = img

    img = QImage(target_w, target_h, QImage.Format_RGB32)
    for x in range(res_w):
        for y in range(res_h):
            c = to_draw[y][x][channel]
            color = get_color(c, rgb)
            img.setPixel(x + offset_w, y + offset_h, color)

    return QPixmap.fromImage(img), min_v, max_v, offset_w, offset_h


def draw_bscan_points(Z: int, X: int, points: [], antialiazing: bool = False, sum: bool = False):
    res = np.zeros([Z, X])

    c = len(points)
    for i in range(c):
        if antialiazing:
            if 0 <= points[i][1] < X and 0 <= points[i][0] < Z:
                set_pixel(points[i][2], points[i][1], points[i][0], res, sum)
        else:
            _z = round(points[i][0])
            _x = round(points[i][1])
            if 0 <= _x < X and 0 <= _z < Z:
                if sum:
                    res[_z, _x] = points[i][2] + res[_z, _x]
                else:
                    res[_z, _x] = max(points[i][2], res[_z, _x])
    return res

def getD(x0, y0, x1, y1):
    return 1 - 0.5*((x0-x1)**2 + (y0-y1)**2)

def set_pixel(v, x, y, img, sum_v = True):
    [max_y, max_x] = img.shape
    x_l = math.floor(x)
    x_r = math.ceil(x)
    y_l = math.floor(y)
    y_r = math.ceil(y)

    if x_r == max_x:
        x_r = x_l

    if y_r == max_y:
        y_r = y_l

    if x_l == x_r:
        if y_l == y_r:
            if sum_v:
                if img[y_l, x_l] < 1.0:
                    img[y_l, x_l] = img[y_l, x_l] + v
                    if img[y_l, x_l] > 1.0:
                        img[y_l, x_l] = 1.0
            else:
                img[y_l, x_l] = max(img[y_l, x_l], v)
        else:
            set_pixel(v * getD(x, y, x_l, y_l), x_l, y_l, img, sum_v)
            set_pixel(v * getD(x, y, x_l, y_r), x_l, y_r, img, sum_v)
    else:
        if y_l == y_r:
            set_pixel(v * getD(x, y, x_l, y_l), x_l, y_l, img, sum_v)
            set_pixel(v * getD(x, y, x_r, y_l), x_r, y_l, img, sum_v)
        else:
            set_pixel(v * getD(x, y, x_r, y_l), x_l, y_l, img, sum_v)
            set_pixel(v * getD(x, y, x_r, y_l), x_r, y_l, img, sum_v)
            set_pixel(v * getD(x, y, x_r, y_l), x_l, y_r, img, sum_v)
            set_pixel(v * getD(x, y, x_r, y_l), x_r, y_r, img, sum_v)



def get_pixel(x, y, img):
    [max_y, max_x] = img.shape
    x_l = math.floor(x)
    x_r = math.ceil(x)
    y_l = math.floor(y)
    y_r = math.ceil(y)

    if x_r == max_x:
        x_r = x_l

    if y_r == max_y:
        y_r = y_l

    v = 0

    if x_l == x_r:
        if y_l == y_r:
            v = img[y_l, x_r]
        else:
            v = img[y_l, x_l] + (img[y_r, x_l] - img[y_l, x_l]) * (y - y_l)
    else:
        if y_l == y_r:
            v = img[y_l, x_l] + (img[y_l, x_r] - img[y_l, x_l]) * (x - x_l)
        else:
            c_l = img[y_l, x_l] + (img[y_r, x_l] - img[y_l, x_l]) * (y - y_l)
            c_r = img[y_l, x_r] + (img[y_r, x_r] - img[y_l, x_r]) * (y - y_l)
            v = c_l + (c_r - c_l) * (x - x_l)

    return v
