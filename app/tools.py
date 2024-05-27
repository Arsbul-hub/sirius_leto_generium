import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtGui import QPixmap, QImage

WESTERN_BLOT_TYPE = 0
ELECTROPHORESIS_TYPE = 1
PROTEIN_TYPE = 2


def convert_pil_to_np(pil_image):
    return np.array(pil_image)


def load_image_as_np(image_path):
    pil_image = Image.open(image_path)
    return np.array(pil_image)[:, :, ::-1].copy()


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


def get_box_area(box, w, h):
    x1, y1, x2, y2 = box
    dx = x2 * w - x1 * w
    dy = y2 * h - y1 * h
    return dx * dy


def get_boxes_area(boxes, w, h):
    area_sum = 0
    for box in boxes:
        area_sum += get_box_area(box, w, h)
    return area_sum


def convert_image_to_generium_output(pil_image):
    w, h = pil_image.size
    offset_x = w * .18
    offset_y = h * .18
    new = Image.new("L", (int(w + offset_x), int(h + offset_y)), 255)
    s_w = int(w * .13)
    s_h = int(h * .13)
    new.paste(pil_image, (s_w, s_h))
    d1 = ImageDraw.Draw(new)
    font = ImageFont.truetype('app/fonts/arial.ttf', int(w * .05) + int(h * .01))
    d1.text((s_w + int(w * .05), s_h / 2 - int(h * .05) / 2), "3", fill=0, font=font)
    d1.text((s_w + int(w * .05) + w / 4 - int(w * .05) / 2, s_h / 2 - int(h * .05) / 2), "<", fill=0, font=font)
    d1.text((s_w + w / 2 - int(w * .05) / 2, s_h / 2 - int(h * .05) / 2), "PI", fill=0, font=font)
    d1.text((s_w - int(w * .05) + (w * 3) / 4 - int(w * .05) / 2, s_h / 2 - int(h * .05) / 2), ">", fill=0, font=font)
    d1.text((s_w - int(w * .05) + w - int(offset_x) / 2, s_h / 2 - int(h * .05) / 2), "10", fill=0, font=font)
    d1.text((s_w / 2 - int(w * .05), s_h + int(h * .1)), "HMW", fill=0, font=font)
    d1.text((s_w / 2 - int(w * .05), s_h + h - int(h * .1)), "LMW", fill=0, font=font)

    return new
