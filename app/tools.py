import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage


def convert_to_np(image_path):
    pil_image = Image.open(image_path)
    return np.array(pil_image)


def convert_to_pixmap(image):
    # height, width, channel = image.shape
    # bytesPerLine = 3 * width
    # qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    # return QPixmap(qimage)
    # height, width, _ = image.shape
    # qimage = QImage(image.data, width, height, width * 3, QImage.Format_RGB888)
    #
    # # Создание объекта QPixmap из объекта QImage
    # return QPixmap.fromImage(qimage)
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    return QPixmap(QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888))


def find_intersection(rect1, rect2):
    x1_rect1, y1_rect1, x2_rect1, y2_rect1 = rect1
    x1_rect2, y1_rect2, x2_rect2, y2_rect2 = rect2

    x_top_left = max(x1_rect1, x1_rect2)
    y_top_left = max(y1_rect1, y1_rect2)
    x_bottom_right = min(x2_rect1, x2_rect2)
    y_bottom_right = min(y2_rect1, y2_rect2)

    if (x_top_left < x_bottom_right) and (y_top_left < y_bottom_right):
        return x_top_left, y_top_left, x_bottom_right, y_bottom_right
    else:
        return None
