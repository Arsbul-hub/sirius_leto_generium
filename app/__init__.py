import datetime
import json
import sys  # sys нужен для передачи argv в QApplication
import time
from functools import partial
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QApplication

from app import design

from PIL import ImageQt
from app.imageWindow import ImageViewer
from app.ThreadManager import ThreadManager
from app.tools import *


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())
        self.screensWidget.setCurrentIndex(0)

        self.thread_manager = ThreadManager()
        self.current_loading_model.setText("Загрузка модуля обработки...\nШаг (1 из 3)")
        self.thread_manager.add_task("prepare", self.load_tensorflow)
        self.analyzer = None
        self.electrophoresis_image_path = None
        self.western_blot_image_path = None
        self.loadElectrophoresisImageButton.clicked.connect(self.load_electrophoresis)
        self.loadWesternBlotImageButton.clicked.connect(self.load_western_blot)
        self.startLoadingButton.clicked.connect(self.load_images)
        self.marginHorizontalSlider.valueChanged.connect(self.update_margin_on_spinners)
        self.marginVerticalSlider.valueChanged.connect(self.update_margin_on_spinners)
        self.marginVerticalSliderValue.valueChanged.connect(self.update_margin_on_sliders)
        self.marginHorizontalSliderValue.valueChanged.connect(self.update_margin_on_sliders)

    def load_tensorflow(self):
        from app.analyzer import Analyzer, Model

        self.analyzer = Analyzer(Model("app\\models\\electrophoresis_model\\saved_model"),
                                 Model("app\\models\\western_blot_model\\saved_model"))
        self.analyzer.set_labels_map("protein", "proteins", "border")
        self.current_loading_model.setText("Загрузка модели электофореза\nШаг (2 из 3)")
        self.analyzer.electrophoresis_model.load_model()

        self.current_loading_model.setText("Загрузка модели вестерн-блота\nШаг (3 из 3)")
        self.analyzer.western_blot_model.load_model()
        self.screensWidget.setCurrentIndex(1)

    def open_visualization_in_window(self, image_np):
        self.set_notify("Открытие изображения...")

        dlg = ImageViewer()
        dlg.open_image(image_np)
        dlg.exec_()
        self.clear_notify()

    def update_margin_on_sliders(self):
        self.marginVerticalSlider.setValue(self.marginVerticalSliderValue.value())
        self.marginHorizontalSlider.setValue(self.marginHorizontalSliderValue.value())
        self.thread_manager.add_task("visualization", self.update_visualization_image)

    def update_margin_on_spinners(self):

        self.marginVerticalSliderValue.setValue(self.marginVerticalSlider.value())
        self.marginHorizontalSliderValue.setValue(self.marginHorizontalSlider.value())
        self.thread_manager.add_task("visualization", self.update_visualization_image)

    def load_electrophoresis(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение опыта электрофореза",
            "",
            "Изображения (*.png *.jpg)"
        )
        if filename and ok:
            path = Path(filename)
            self.electrophoresis_image_path = str(path.absolute())
            self.electrophoresisImageName.setText(path.name)
            pixmap = QPixmap(self.electrophoresis_image_path)
            pixmap = pixmap.scaledToWidth(self.electrophoresisImage.width())
            pixmap = pixmap.scaledToHeight(self.electrophoresisImage.height())
            self.electrophoresisImage.setPixmap(pixmap)
            if self.electrophoresis_image_path is not None and self.western_blot_image_path is not None:
                self.startLoadingButton.setEnabled(True)
                self.visualisationImage.setText("Загрузка изображения, подождите...")
                self.thread_manager.add_task("visualization", self.update_visualization_image)
                self.visualisationImageControlls.setEnabled(True)

    def load_western_blot(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение опыта вестерн-блота",
            "",
            "Изображения (*.png *.jpg)"
        )
        if filename and ok:
            path = Path(filename)
            self.western_blot_image_path = str(path.absolute())
            self.westernBlotImageName.setText(path.name)

            pixmap = QPixmap(self.western_blot_image_path)
            pixmap = pixmap.scaledToWidth(self.westernBlotImage.width())
            pixmap = pixmap.scaledToHeight(self.westernBlotImage.height())
            self.westernBlotImage.setPixmap(pixmap)
            if self.western_blot_image_path is not None and self.electrophoresis_image_path is not None:
                self.startLoadingButton.setEnabled(True)
                self.visualisationImage.setText("Загрузка изображения, подождите")
                self.thread_manager.add_task("visualization", self.update_visualization_image)
                self.visualisationImageControlls.setEnabled(True)

    def update_visualization_image(self):

        pil_image = self.load_visualization_image()
        image = QImage(pil_image.tobytes(), pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
        pixmap = QPixmap(image)
        pixmap = pixmap.scaledToWidth(self.visualisationImage.width())
        pixmap = pixmap.scaledToHeight(self.visualisationImage.height())
        self.visualisationImage.setPixmap(pixmap)
        self.openOutputImageButton.clicked.connect(lambda: self.open_image(pil_image))

    def load_visualization_image(self):
        image1 = Image.open(self.electrophoresis_image_path)
        image2 = Image.open(self.western_blot_image_path)
        # image2.resize(image1.size)
        # image2 = image2.resize(image1.size)
        # blend = Image.blend(image1, image2, alpha=0.5)
        blend = image1.copy()

        # Смещение второго изображения
        x_offset = int(self.marginHorizontalSlider.value() / 100 * image2.width)  # Смещение вправо на 40 пикселей
        y_offset = int(self.marginVerticalSlider.value() / 100 * image2.height)
        mask = image2.convert('L').point(lambda x: min(x, 170))  # Создание маски для второго изображения
        image2.putalpha(mask)  # Установка альфа-канала второго изображения из маски

        # Наложение второго изображения смещенным на новое изображение
        blend.paste(image2, (x_offset, y_offset), mask=image2)
        return blend

    def load_images(self):
        self.set_notify("Анализ данных, подождите...")
        self.startLoadingButton.setEnabled(False)
        self.analyzer.load_images(self.electrophoresis_image_path, self.western_blot_image_path)
        self.analyzer.load_models_detections()
        self.load_all_images()
        self.clear_notify()
        # self.screensWidget.setCurrentIndex(1)

    def switch_next(self):
        self.screensWidget.setCurrentIndex(1)

    def switch_previous(self):
        self.screensWidget.setCurrentIndex(0)

    def load_all_images(self):

        # e_boxes, e_classes, e_scores = self.analyzer.electrophoresis_model.get_detections()
        # w_boxes, w_classes, w_scores = self.analyzer.western_blot_model.get_detections()

        # self.thread_manager.add_task("analyze", self.analyzer.analyze, read_analyze_data, timer=1)

        boxes, classes, scores, btypes = self.analyzer.analyze(x_offset_hint=self.marginHorizontalSlider.value() / 100,
                                                               y_offset_hint=self.marginVerticalSlider.value() / 100)
        e_boxes, e_classes, e_scores, w_boxes, w_classes, w_scores, any_btypes = self.analyzer.analyze_any()

        out_visualized = self.analyzer.visualize(boxes, classes, scores, btypes, self.load_visualization_image(),
                                                 self.marginHorizontalSlider.value() / 100,
                                                 self.marginVerticalSlider.value() / 100)

        # e_pixmap = convert_to_pixmap(electrophoresis_visualized)

        # e_pixmap = e_pixmap.scaledToWidth(self.electrophoresisImage.width())
        # e_pixmap = e_pixmap.scaledToHeight(self.electrophoresisImage.height())
        # w_pixmap = convert_to_pixmap(western_blot_visualized)
        # w_pixmap = w_pixmap.scaledToWidth(self.westernBlotImage.width())
        # w_pixmap = w_pixmap.scaledToHeight(self.westernBlotImage.height())

        o_pixmap = convert_to_pixmap(out_visualized)
        o_pixmap = o_pixmap.scaledToWidth(self.visualisationImage.width())
        o_pixmap = o_pixmap.scaledToHeight(self.visualisationImage.height())

        # self.electrophoresisImage.setPixmap(QPixmap(e_pixmap))
        # self.westernBlotImage.setPixmap(QPixmap(w_pixmap))
        # pil_image = blended_image
        # image = QImage(pil_image.tobytes(), pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
        # pixmap = QPixmap(image)
        # pixmap = pixmap.scaledToWidth(self.visualisationImage.width())
        # pixmap = pixmap.scaledToHeight(self.visualisationImage.height())

        self.visualisationImage.setPixmap(o_pixmap)

        # self.openWesternBlotImageButton.clicked.connect(lambda: self.open_image(western_blot_visualized))
        # self.openElectrophoresisImageButton.clicked.connect(lambda: self.open_image(electrophoresis_visualized))

        self.startLoadingButton.setEnabled(True)

        self.openOutputImageButton.disconnect()
        self.openOutputImageButton.clicked.connect(lambda: self.open_visualization_in_window(out_visualized))

        h, w, _ = self.analyzer.electrophoresis_image.shape
        self.saveDataButton.disconnect()
        self.saveDataButton.clicked.connect(lambda: self.save_data(base_image_size=(w, h),
                                                                   out_image=out_visualized,
                                                                   boxes=boxes, classes=classes, scores=scores,
                                                                   e_boxes=e_boxes, e_classes=e_classes,
                                                                   e_scores=e_scores,
                                                                   w_boxes=w_boxes, w_classes=w_classes,
                                                                   w_scores=w_scores))
        self.load_info(boxes=boxes, e_boxes=e_boxes, w_boxes=w_boxes)

        # electrophoresis_visualized = visualize(e_boxes, e_classes, e_scores, self.analyzer.electrophoresis_image)
        #
        # western_blot_visualized = visualize(w_boxes, w_classes, w_scores, self.analyzer.western_blot_image)
        # blended_image = np.array(self.load_visualization_image())

    def load_info(self, **kwargs):
        self.info_1.setText(f"Распознанные области: {len(kwargs['boxes'])}")
        self.info_2.setText(f"Распозннаные области на снимках электрофореза: {len(kwargs['e_boxes'])}")
        self.info_3.setText(f"Распозннаные области на снимках вестерн-блота: {len(kwargs['w_boxes'])}")
        ratio = len(kwargs['w_boxes']) / len(kwargs['e_boxes'])
        self.info_4.setText(
            f"Отношения количество белков вестерн-блота к электрофореза ({len(kwargs['w_boxes'])} к {len(kwargs['e_boxes'])}): {ratio:.3f}")

    def save_data(self, **kwargs):
        filename = QFileDialog.getExistingDirectory(
            self,
            "Select a Path",
            ""
        )

        if filename:
            path = Path(filename)
            today = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            document_path = path.joinpath(f"Результаты анализа {today}").absolute()
            self.set_notify("Сохранение...")
            with open(f"{document_path}.txt", "w") as document:
                text = ""
                text += f"Распознанные области: {len(kwargs['boxes'])}\n"
                text += f"Распозннаные области на снимках электрофореза: {len(kwargs['e_boxes'])}\n"
                text += f"Распозннаные области на снимках вестерн-блота: {len(kwargs['w_boxes'])}\n"
                ratio = len(kwargs['w_boxes']) / len(kwargs['e_boxes'])
                text += f"Отношения количество белков вестерн-блота к электрофореза ({len(kwargs['w_boxes'])} к {len(kwargs['e_boxes'])}): {ratio:.3f}\n"
                document.write(text)
            json_path = path.joinpath(f"Результаты анализа {today}").absolute()
            with open(f"{json_path}.json", "w") as document:
                ratio = len(kwargs['w_boxes']) / len(kwargs['e_boxes'])
                data = {
                    "base_image_size": kwargs["base_image_size"],
                    "detections_type": "Opencv detections",
                    "boxes_data_type": "Ratio",
                    "detected_areas": len(kwargs['boxes']),
                    "detected_areas_electrophoresis": len(kwargs['e_boxes']),
                    "detected_areas_western_blot": len(kwargs['w_boxes']),
                    "proteins_ratio": ratio,
                    "boxes": kwargs["boxes"],
                    "classes": kwargs["classes"],
                    "scores": kwargs["scores"],
                    "electrophoresis_boxes": kwargs["e_boxes"],
                    "electrophoresis_classes": kwargs["e_classes"],
                    "electrophoresis_scores": kwargs["e_scores"],
                    "western_blot_boxes": kwargs["w_boxes"],
                    "western_blot_classes": kwargs["w_classes"],
                    "western_blot_scores": kwargs["w_scores"],
                }
                json.dump(data, document)
            self.thread_manager.add_task("save image",
                                         lambda: Image.fromarray(kwargs["out_image"]).save(
                                             path.joinpath(f"Снимок {today}.png").absolute()),
                                         lambda: self.clear_notify)

    def set_notify(self, message):
        self.clear_notify()
        self.analyzeStatus.setText(message)

    def clear_notify(self):
        self.analyzeStatus.clear()

    def closeEvent(self, event):
        self.thread_manager.stop_loop()
        event.accept()
        sys.exit(0)
