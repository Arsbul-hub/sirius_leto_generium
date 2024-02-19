import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QSizePolicy, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageViewer(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(1000, 702)
        self.setWindowTitle("Просмотр обработанного снимка")
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.x, self.y = 0, 0

    def open_image(self, image_np):
        self.image_np = image_np
        height, width, _ = image_np.shape
        self.x, self.y = width, height
        bytesPerLine = 3 * width
        image = QImage(self.image_np.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaledToWidth(self.width())
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # масштабировать содержимое QLabel под размер окна
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # игнорировать политику размеров QLabel

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.startingPos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            dx = event.pos().x() - self.startingPos.x()
            dy = event.pos().y() - self.startingPos.y()
            self.label.move(self.label.x() + dx, self.label.y() + dy)
            self.startingPos = event.pos()

    def wheelEvent(self, event):
        angleDelta = event.angleDelta() / 8
        angleY = angleDelta.y()
        if angleY > 0:

            self.x *= 1.1
            self.y *= 1.1
        else:
            self.x *= 0.9
            self.y *= 0.9
        self.label.resize(int(self.x), int(self.y))
        height, width, _ = self.image_np.shape
        bytesPerLine = 3 * width
        image = QImage(self.image_np.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pix = QPixmap(image)
        pix = pix.scaledToWidth(int(self.x))
        pix = pix.scaledToHeight(int(self.y))
        self.label.setPixmap(pix)
