from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsEllipseItem
from PyQt5.QtGui import QPixmap, QBrush, QColor, QPen, QCursor
from PyQt5.QtCore import Qt
from OCTAsset import OCTAsset, OCTAssetList
from OCTFeature import OCTFeatureMatchSet, OCTFeatureMatch


class OCTFeatureMatchGraphicsItem(QGraphicsLineItem):
    mark_size: int = 2
    x1: float
    x2: float
    y1: float
    y2: float

    activeColor: QColor = QColor(0, 255, 0)
    disabledColor: QColor = QColor(100, 100, 100)

    match: OCTFeatureMatch

    def __init__(self, match: OCTFeatureMatch, offset_x: float, offset_y: float, mark_size: int):
        self.match = match
        self.x1 = match.x0[0]
        self.y1 = match.x0[1]
        self.x2 = match.x1[0] + offset_x
        self.y2 = match.x1[1] + offset_y

        super().__init__(self.x1, self.y1, self.x2, self.y2)
        self.setMarkSize(mark_size)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)

        brush = QBrush(self.activeColor)
        self.setPen(QPen(brush, 1))
        self.setMatchColor()

        self.setCursor(Qt.PointingHandCursor)

    def setMarkSize(self, mark_size: int = 2):
        self.mark_size = mark_size

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):
        painter.setPen(self.pen())
        painter.drawLine(self.x1, self.y1, self.x2, self.y2)
        painter.drawEllipse(self.x1 - self.mark_size // 2, self.y1 - self.mark_size // 2, self.mark_size,
                            self.mark_size)
        painter.drawEllipse(self.x2 - self.mark_size // 2, self.y2 - self.mark_size // 2, self.mark_size,
                            self.mark_size)

    def hoverEnterEvent(self, event):
        pen = self.pen()
        pen.setColor(QColor(255, 0, 0))
        self.setPen(pen)
        self.update()

    def hoverLeaveEvent(self, event):
        self.setMatchColor()
        self.update()

    def setMatchColor(self):
        pen = self.pen()
        if self.match.get_state():
            pen.setColor(self.activeColor)
        else:
            pen.setColor(self.disabledColor)
        self.setPen(pen)

    def mousePressEvent(self, event):
        self.match.set_state(not self.match.get_state())


class OCTFeatureMatchesView:
    scene: QGraphicsScene
    bscan0: QGraphicsPixmapItem
    bscan1: QGraphicsPixmapItem

    offset_x: int
    offset_y: int
    scan_w: int
    scan_h: int
    view_w: int
    view_h: int

    match_e = []

    def __init__(self):
        self.scene = QGraphicsScene()
        self.bscan0 = self.scene.addPixmap(QPixmap())
        self.bscan1 = self.scene.addPixmap(QPixmap())

    def clear(self):
        for i in self.match_e:
            self.scene.removeItem(i)
        self.scene.clear()
        self.bscan0 = self.scene.addPixmap(QPixmap())
        self.bscan1 = self.scene.addPixmap(QPixmap())
        self.bscan1.setOffset(self.offset_x, self.offset_y)
        self.scene.addLine(self.offset_x, self.offset_y, self.scan_w, self.scan_h, QColor(255, 0, 0))

    def set_view_size(self, assets_list: OCTAssetList):
        [self.scan_w, self.scan_h] = assets_list.get_image_size()
        ar = self.scan_w / self.scan_h

        if ar >= 2:
            self.view_w = self.scan_w
            self.view_h = self.scan_h * 2
            self.offset_x = 0
            self.offset_y = self.scan_h
        else:
            self.view_w = self.scan_w * 2
            self.view_h = self.scan_h
            self.offset_x = self.scan_w
            self.offset_y = 0

        self.bscan0.setOffset(0, 0)
        self.bscan1.setOffset(self.offset_x, self.offset_y)
        self.scene.addLine(self.offset_x, self.offset_y, self.scan_w, self.scan_h, QColor(255, 0, 0))

        return self.view_w, self.view_h

    def set_b_scans(self, img0: OCTAsset, img1: OCTAsset):
        self.clear()
        self.bscan0.setPixmap(img0.get_pixmap())
        self.bscan1.setPixmap(img1.get_pixmap())
        self.scene.update()

    def draw_feature_matches(self, matches: OCTFeatureMatchSet):
        mark_size = 4
        offset_x = self.bscan1.offset().x()
        offset_y = self.bscan1.offset().y()

        self.match_e = []

        for i in matches.features:
            line = OCTFeatureMatchGraphicsItem(i, offset_x, offset_y, mark_size)
            self.scene.addItem(line)
            self.match_e.append(line)

    def update(self):
        for i in self.match_e:
            i.setMatchColor()
            i.update()
        self.scene.update()
