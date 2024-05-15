from random import randint

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap

print("Tensorflow module loaded")

from app.tools import *


class Analyzer:

    def __init__(self, model1=None, model2=None):
        self.labels_map = [""]
        self.electrophoresis_model = model1
        self.western_blot_model = model2
        self.western_blot_image = None
        self.electrophoresis_image = None
        self.electrophoresis_image_tensor = None
        self.western_blot_image_tensor = None

    def load_models(self, model1, model2):
        self.electrophoresis_model = model1
        self.western_blot_model = model2

    def set_labels_map(self, *args):
        for name in args:
            self.labels_map.append(name)

    def load_images(self, electrophoresis_image_path, western_blot_image_path):
        self.electrophoresis_image = convert_to_np(electrophoresis_image_path)
        self.western_blot_image = convert_to_np(western_blot_image_path)
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
            e_y1_hint, e_x1_hint, e_y2_hint, e_x2_hint = self.electrophoresis_model.boxes[0][i]
            electrophoresis_proteins_score = self.electrophoresis_model.scores[0][i]
            if electrophoresis_proteins_score > .30:
                e_x1 = int(e_x1_hint * w_e)
                e_y1 = int(e_y1_hint * h_e)
                e_x2 = int(e_x2_hint * w_e)
                e_y2 = int(e_y2_hint * h_e)
                # intersections = self.find_intersection((w_x1, w_y1, w_x2, w_y2), (e_x1, e_y1, e_x2, e_y2))
                # if intersections is not None:
                if electrophoresis_proteins_score >= trigger_threshold:
                    e_boxes.append((e_x1, e_y1, e_x2, e_y2))
                    e_classes.append(self.labels_map[electrophoresis_proteins_class])
                    e_scores.append(electrophoresis_proteins_score.item())  # convert float32 to float
                    btypes.append(ELECTROPHORESIS_TYPE)
        for j in range(len(self.western_blot_model.classes[0])):
            western_blot_proteins_class = self.western_blot_model.classes[0][j]
            w_y1_hint, w_x1_hint, w_y2_hint, w_x2_hint = self.western_blot_model.boxes[0][j]
            western_blot_proteins_score = self.western_blot_model.scores[0][j]
            if western_blot_proteins_score > .30:
                w_x1 = int(w_x1_hint * w_e)
                w_y1 = int(w_y1_hint * h_e)
                w_x2 = int(w_x2_hint * w_e)
                w_y2 = int(w_y2_hint * h_e)
                # intersections = self.find_intersection((w_x1, w_y1, w_x2, w_y2), (e_x1, e_y1, e_x2, e_y2))
                # if intersections is not None:
                if western_blot_proteins_score >= trigger_threshold:
                    w_boxes.append((w_x1, w_y1, w_x2, w_y2))
                    w_classes.append(self.labels_map[western_blot_proteins_class])
                    w_scores.append(western_blot_proteins_score.item())  # convert float32 to float
                    btypes.append(WESTERN_BLOT_TYPE)
        return e_boxes, e_classes, e_scores, w_boxes, w_classes, w_scores, btypes

    def analyze(self, trigger_threshold=.5, x_offset_hint=0, y_offset_hint=0):
        boxes = []
        scores = []
        classes = []
        btypes = []
        h_e, w_e, _ = self.electrophoresis_image.shape
        h_w, w_w, _ = self.western_blot_image.shape
        for i in range(len(self.electrophoresis_model.classes[0])):
            electrophoresis_proteins_class = self.electrophoresis_model.classes[0][i]
            electrophoresis_proteins_box = self.electrophoresis_model.boxes[0][i]
            electrophoresis_proteins_score = self.electrophoresis_model.scores[0][i]
            e_y1, e_x1, e_y2, e_x2 = electrophoresis_proteins_box
            if electrophoresis_proteins_score >= trigger_threshold:
                boxes.append((e_x1, e_y1, e_x2, e_y2))
                classes.append(electrophoresis_proteins_class)

                scores.append(electrophoresis_proteins_score.item())  # convert float32 to float
                btypes.append(ELECTROPHORESIS_TYPE)
                for j in range(len(self.western_blot_model.classes[0])):
                    western_blot_proteins_class = self.western_blot_model.classes[0][j]
                    western_blot_proteins_box = self.western_blot_model.boxes[0][j]
                    western_blot_proteins_score = self.western_blot_model.scores[0][j]

                    w_y1, w_x1, w_y2, w_x2 = western_blot_proteins_box
                    if western_blot_proteins_score > trigger_threshold:
                        boxes.append((w_x1, w_y1, w_x2, w_y2))
                        classes.append(western_blot_proteins_class)
                        scores.append(western_blot_proteins_score.item())  # convert float32 to float
                        btypes.append(WESTERN_BLOT_TYPE)

                        intersections = find_intersection((e_x1, e_y1, e_x2, e_y2),
                                                          (w_x1 + x_offset_hint,
                                                           w_y1 + y_offset_hint,
                                                           w_x2 + x_offset_hint,
                                                           w_y2 + y_offset_hint
                                                           ))
                        if intersections is not None:
                            boxes.append(intersections)
                            classes.append(2)  # proteins
                            score = (electrophoresis_proteins_score.item() + western_blot_proteins_score.item()) / 2
                            scores.append(score)  # convert float32 to float
                            btypes.append(PROTEIN_TYPE)
        return boxes, classes, scores, btypes
        # print(len(self.electrophoresis_model.classes[0]))
        # print(len(self.electrophoresis_model.boxes[0]))

    def visualize(self, boxes, classes, scores, btypes, image, x_offset_hint=0, y_offset_hint=0,
                  viz_allow=[PROTEIN_TYPE]):
        image_np = np.array(image)
        height, width, _ = image_np.shape
        h, w, __ = self.western_blot_image.shape
        for c, s, b, t in zip(classes, scores, boxes, btypes):
            x1, y1, x2, y2 = b
            border_color = 0, 0, 255

            border_radius = round(width * height * .000_0006)  # % от площади
            if border_radius == 0:
                border_radius = 1

            font_size = round(width * height * .000_00001)

            if t not in viz_allow:
                continue
            if t == ELECTROPHORESIS_TYPE and ELECTROPHORESIS_TYPE in viz_allow:
                border_color = 100, 255, 100
                x1 = int(x1 * width)
                x2 = int(x2 * width)
                y1 = int(y1 * height)
                y2 = int(y2 * height)

            elif t == WESTERN_BLOT_TYPE and WESTERN_BLOT_TYPE in viz_allow:
                border_color = 200, 200, 100
                x1 = int((x1 + x_offset_hint) * width)
                x2 = int((x2 + x_offset_hint) * width)
                y1 = int((y1 + y_offset_hint) * height)
                y2 = int((y2 + y_offset_hint) * height)

            if t == PROTEIN_TYPE and PROTEIN_TYPE in viz_allow:
                border_color = 255, 100, 100

                x1 = int(x1 * width)
                x2 = int(x2 * width)
                y1 = int(y1 * height)
                y2 = int(y2 * height)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), border_color, border_radius)
            label = f"Class: {c}, Score: {s:.2f}"
            cv2.putText(image_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, border_color, int(border_radius))
        return image_np


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
