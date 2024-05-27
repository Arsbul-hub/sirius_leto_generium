from random import randint

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap

print("Tensorflow module loaded")

from app.tools import *


class Analyzer:
    CLASES = "Одиночный белок", "Группа белков"

    def __init__(self, model1=None, model2=None):
        self.labels_map = [""]
        self.electrophoresis_model = model1
        self.western_blot_model = model2
        self.western_blot_image = None
        self.electrophoresis_image = None
        self.western_blot_image_resized = None
        self.electrophoresis_image_resized = None
        self.electrophoresis_image_tensor = None
        self.western_blot_image_tensor = None

    def load_models(self, model1, model2):
        self.electrophoresis_model = model1
        self.western_blot_model = model2

    def set_labels_map(self, *args):
        for name in args:
            self.labels_map.append(name)

    def load_images(self, electrophoresis_image_path, western_blot_image_path):
        self.electrophoresis_image = load_image_as_np(electrophoresis_image_path)
        self.western_blot_image = load_image_as_np(western_blot_image_path)

        self.electrophoresis_image_tensor = tf.convert_to_tensor(np.expand_dims(self.electrophoresis_image, 0),
                                                                 dtype=tf.uint8)
        self.western_blot_image_tensor = tf.convert_to_tensor(np.expand_dims(self.western_blot_image, 0),
                                                              dtype=tf.uint8)

    def load_models_detections(self):
        self.electrophoresis_model.load_detections(self.electrophoresis_image_tensor)
        self.western_blot_model.load_detections(self.western_blot_image_tensor)

    def analyze_any(self, trigger_threshold=.5):
        e_boxes = []
        e_scores = []
        e_classes = []
        w_boxes = []
        w_scores = []
        w_classes = []
        btypes = []
        h_e, w_e, _ = self.electrophoresis_image.shape
        for i in range(self.electrophoresis_model.classes.shape[1]):
            electrophoresis_proteins_class = self.electrophoresis_model.classes[0][i]
            e_y1, e_x1, e_y2, e_x2 = self.electrophoresis_model.boxes[0][i]
            electrophoresis_proteins_score = self.electrophoresis_model.scores[0][i]
            # intersections = self.find_intersection((w_x1, w_y1, w_x2, w_y2), (e_x1, e_y1, e_x2, e_y2))
            # if intersections is not None:
            if electrophoresis_proteins_score >= trigger_threshold:
                e_boxes.append((float(e_x1), float(e_y1), float(e_x2), float(e_y2)))
                e_classes.append(self.CLASES[int(electrophoresis_proteins_class) - 1])
                e_scores.append(float(electrophoresis_proteins_score))  # convert float32 to float
                btypes.append(ELECTROPHORESIS_TYPE)
        for j in range(len(self.western_blot_model.classes[0])):
            western_blot_proteins_class = self.western_blot_model.classes[0][j]
            w_y1, w_x1, w_y2, w_x2 = self.western_blot_model.boxes[0][j]
            western_blot_proteins_score = self.western_blot_model.scores[0][j]

            # intersections = self.find_intersection((w_x1, w_y1, w_x2, w_y2), (e_x1, e_y1, e_x2, e_y2))
            # if intersections is not None:
            if western_blot_proteins_score >= trigger_threshold:
                w_boxes.append((float(w_x1), float(w_y1), float(w_x2), float(w_y2)))
                w_classes.append(self.CLASES[int(western_blot_proteins_class) - 1])
                w_scores.append(float(western_blot_proteins_score))  # convert float32 to float
                btypes.append(WESTERN_BLOT_TYPE)
        return e_boxes, e_classes, e_scores, w_boxes, w_classes, w_scores

    def analyze(self, trigger_threshold=.5, x_offset_hint=0, y_offset_hint=0):
        boxes = []
        scores = []
        classes = []
        e_boxes = []
        e_scores = []
        e_classes = []
        w_boxes = []
        w_scores = []
        w_classes = []
        btypes = []
        h_e, w_e, _ = self.electrophoresis_image.shape
        h_w, w_w, _ = self.western_blot_image.shape
        for i in range(len(self.electrophoresis_model.classes[0])):
            electrophoresis_proteins_class = self.electrophoresis_model.classes[0][i]
            electrophoresis_proteins_box = self.electrophoresis_model.boxes[0][i]

            electrophoresis_proteins_score = self.electrophoresis_model.scores[0][i]
            e_y1, e_x1, e_y2, e_x2 = electrophoresis_proteins_box
            if electrophoresis_proteins_score >= trigger_threshold:

                for j in range(len(self.western_blot_model.classes[0])):
                    western_blot_proteins_class = self.western_blot_model.classes[0][j]
                    western_blot_proteins_box = self.western_blot_model.boxes[0][j]
                    western_blot_proteins_score = self.western_blot_model.scores[0][j]

                    w_y1, w_x1, w_y2, w_x2 = western_blot_proteins_box
                    if western_blot_proteins_score >= trigger_threshold:

                        intersections = find_intersection((float(e_x1), float(e_y1), float(e_x2), float(e_y2)),
                                                          (float(w_x1) + x_offset_hint,
                                                           float(w_y1) + y_offset_hint,
                                                           float(w_x2) + x_offset_hint,
                                                           float(w_y2) + y_offset_hint
                                                           ))
                        if intersections is not None:
                            e_boxes.append((float(e_x1), float(e_y1), float(e_x2), float(e_y2)))
                            e_classes.append(self.CLASES[int(electrophoresis_proteins_class) - 1])

                            e_scores.append(float(electrophoresis_proteins_score))  # convert float32 to float
                            # btypes.append(ELECTROPHORESIS_TYPE)

                            w_boxes.append((float(w_x1), float(w_y1), float(w_x2), float(w_y2)))
                            w_classes.append(self.CLASES[int(western_blot_proteins_class) - 1])
                            w_scores.append(float(western_blot_proteins_score))  # convert float32 to float
                            # btypes.append(WESTERN_BLOT_TYPE)

                            boxes.append(intersections)
                            classes.append(self.CLASES[1])  # proteins
                            score = (np.clip(float(electrophoresis_proteins_score) + float(western_blot_proteins_score),
                                             0, 1)) / 2
                            scores.append(float(score))  # convert float32 to float
                            btypes.append(PROTEIN_TYPE)
        return boxes, classes, scores, e_boxes, e_classes, e_scores, w_boxes, w_classes, w_scores
        # print(len(self.electrophoresis_model.classes[0]))
        # print(len(self.electrophoresis_model.boxes[0]))

    def visualize(self, boxes, classes, scores, image_np, x_offset_hint=0, y_offset_hint=0,
                  viz_allow=PROTEIN_TYPE):

        height, width, _ = image_np.shape
        h, w, __ = self.western_blot_image.shape
        for c, s, b in zip(classes, scores, boxes):
            x1, y1, x2, y2 = b
            border_color = 0, 0, 255

            border_radius = round(width * height * .000_0002)  # % от площади
            if border_radius == 0:
                border_radius = 1

            font_size = round(width * height * .000_00007)

            if viz_allow == ELECTROPHORESIS_TYPE:
                border_color = 255, 100, 100

            elif viz_allow == WESTERN_BLOT_TYPE:
                border_color = 200, 200, 100

            elif viz_allow == PROTEIN_TYPE:
                border_color = 100, 255, 100

            x1 = int(x1 * width)
            x2 = int(x2 * width)
            y1 = int(y1 * height)
            y2 = int(y2 * height)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), border_color, border_radius)
            label = f"Белок"
            # cv2.putText(image_np, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_COMPLEX, font_size, border_color, int(border_radius))
        return image_np

    def analyze_info(self, boxes, e_boxes, w_boxes, all_e_boxes, all_w_boxes):
        w_e, h_e, _ = self.electrophoresis_image.shape
        w_w, h_w, _ = self.western_blot_image.shape
        # [(0.334951788187027, 0.3467009961605072, 0.40479597449302673, 0.42856183648109436)] boxes
        n_e = get_boxes_area(all_e_boxes, w_e, h_e)
        n_w = get_boxes_area(all_w_boxes, w_w, h_w)
        n = get_boxes_area(boxes, w_e, h_e)
        # Количество совпадающих пятен
        n_common = len(boxes)

        # Площадь совпадающих пятен (предположим, что совпадающие пятна имеют одинаковую площадь)
        common_area = n

        # Доля совпадающих пятен от всех пятен на 2D-электрофорезе (по площади)

        ratio_e_area = common_area / n_e

        # Доля совпадающих пятен от всех пятен на Вестерн-блоте (по площади)
        ratio_w_area = common_area / n_w

        # Доля совпадающих пятен от всех пятен на 2D-электрофорезе (по количеству)
        ratio_e_count = n_common / len(all_e_boxes)

        # Доля совпадающих пятен от всех пятен на Вестерн-блоте (по количеству)
        ratio_w_count = n_common / len(all_w_boxes)

        # Доля совпадающих пятен от всех пятен на обоих фото (по количеству)
        ratio_common_count = n_common / (len(all_e_boxes) + len(all_w_boxes))

        # Доля совпадающих пятен от всех пятен на обоих фото (по площади)
        ratio_common_area = common_area / (n_e + n_w)

        # Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по количеству)

        ratio_only_e_count = len(e_boxes) / len(all_e_boxes)

        # Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по площади)

        ratio_only_e_area = get_boxes_area(e_boxes, w_e, h_e) / n_e

        rate_error_by_count = ((len(all_e_boxes) + len(all_w_boxes)) - n_common) / (len(all_e_boxes) + len(all_w_boxes))
        rate_error_by_area = ((n_e + n_w) - common_area) / (n_e + n_w)
        return {
            "ratio_e_area": ratio_e_area,
            "ratio_w_area": ratio_w_area,
            "ratio_e_count": ratio_e_count,
            "ratio_w_count": ratio_w_count,
            "ratio_common_count": ratio_common_count,
            "ratio_common_area": ratio_common_area,
            "ratio_only_e_count": ratio_only_e_count,
            "ratio_only_e_area": ratio_only_e_area,
            "rate_error_by_count": rate_error_by_count,
            "rate_error_by_area": rate_error_by_area,
            "common_area": common_area,
        }


class Model:
    def __init__(self, path):
        self.model = None
        self.path = path
        self.detection = None
        self.boxes = []
        self.classes = []
        self.scores = []

    def load_model(self):
        self.model = tf.saved_model.load(self.path)

    def load_detections(self, input_tensor):
        self.detection = self.model(input_tensor)
        self.boxes = self.detection['detection_boxes'].numpy()
        self.classes = self.detection['detection_classes'].numpy().astype(int)
        self.scores = self.detection['detection_scores'].numpy()

    def get_detections(self):
        boxes = []
        scores = []
        classes = []
        for i in range(len(self.classes[0])):
            boxes.append(self.boxes[0][i])
            classes.append(self.classes[0][i])
            scores.append(self.scores[0][i])
        return boxes, classes, scores
