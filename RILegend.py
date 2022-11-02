from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtCore import QRectF, Qt, QRect
from PyQt5.QtGui import QPixmap, QBrush, QColor, QPen, QPainterPath, QTextItem, QLinearGradient

import numpy as np
import OCTHelper


class RILegendItem(QGraphicsItem):
    min_v: float = 1
    max_v: float = 2

    def boundingRect(self):
        return QRectF(0, 0, 512, 49)

    def paint(self, painter, option, widget):
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        grad1 = QLinearGradient(16, 0, 480, 0)
        # add intermediate colors to mimic hue mixing
        for i in np.linspace(0, 1, 10):
            grad1.setColorAt(i, OCTHelper.get_qcolor(i*255.0, False))
        painter.setBrush(QBrush(grad1))
        painter.drawRect(16, 0, 480, 20)

        pos = 16
        v = self.min_v
        step = (self.max_v - self.min_v) / 4.0
        for i in range(5):
            painter.drawLine(pos, 20, pos, 28)
            if i == 0:
                flag = 0
                text_pos = 0
            elif i < 4:
                flag = Qt.AlignHCenter
                text_pos = -50
            else:
                flag = Qt.AlignRight
                text_pos = -100
            painter.drawText(QRect(pos + text_pos, 30, 100, 20), flag, format(v, '.4f'))
            pos = pos + 120
            v = v + step

    def setVals(self, min_v: float, max_v: float):
        self.min_v = min_v
        self.max_v = max_v
        self.update()
