import datetime
import json
from pathlib import Path

import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from app import design
from app.ThreadManager import ThreadManager
from app.imageWindow import ImageViewer
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
        self.electrophoresis_image = None
        self.western_blot_image = None
        self.out_visualized_image = None
        self.out_visualized_electrophoresis_image = None
        self.is_analyzing_by_area = False
        self.analyzed_info = None
        self.data_for_saving = None
        self.loadElectrophoresisImageButton.clicked.connect(self.load_electrophoresis)
        self.loadWesternBlotImageButton.clicked.connect(self.load_western_blot)
        self.startLoadingButton.clicked.connect(self.load_images)
        self.marginHorizontalSlider.valueChanged.connect(self.update_margin_on_spinners)
        self.marginVerticalSlider.valueChanged.connect(self.update_margin_on_spinners)
        self.marginVerticalSliderValue.valueChanged.connect(self.update_margin_on_sliders)
        self.marginHorizontalSliderValue.valueChanged.connect(self.update_margin_on_sliders)
        self.openOutputImageButton.clicked.connect(
            lambda: self.open_visualization_in_window(self.out_visualized_image))
        self.openOutputElectrophoresisButton.clicked.connect(
            lambda: self.open_visualization_in_window(self.out_visualized_electrophoresis_image))
        self.openOutputWesternBlotButton.clicked.connect(
            lambda: self.open_visualization_in_window(self.out_visualized_western_blot_image))
        self.saveDataButton.clicked.connect(lambda: self.save_data(**self.data_for_saving))
        self.allowAreaCalculationCheckbox.stateChanged.connect(self.update_info_in_window)

    def closeEvent(self, event):
        msg = QMessageBox(self)
        msg.setWindowTitle("Внимание")
        msg.setIcon(QMessageBox.Question)
        msg.setText("Вы действительно хотите выйти?")

        yes = msg.addButton("Да", QMessageBox.YesRole)
        no = msg.addButton("Отмена", QMessageBox.RejectRole)
        msg.setDefaultButton(no)
        msg.exec_()
        if msg.clickedButton() == yes:
            self.thread_manager.stop_loop()
            event.accept()
        else:
            msg.close()
            event.ignore()

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
            self.electrophoresisImage.setStyleSheet("background-color: none")
            pixmap = QPixmap(self.electrophoresis_image_path)
            pixmap = pixmap.scaledToWidth(self.electrophoresisImage.width())
            pixmap = pixmap.scaledToHeight(self.electrophoresisImage.height())
            self.electrophoresisImage.setPixmap(pixmap)

            if self.electrophoresis_image_path is not None and self.western_blot_image_path is not None:
                self.visualisationImage.setText(None)
                self.set_notify("Загрузка изображения, подождите...")
                self.update_images()
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
            self.westernBlotImage.setStyleSheet("background-color: none")
            pixmap = QPixmap(self.western_blot_image_path)
            pixmap = pixmap.scaledToWidth(self.westernBlotImage.width())
            pixmap = pixmap.scaledToHeight(self.westernBlotImage.height())
            self.westernBlotImage.setPixmap(pixmap)
            if self.western_blot_image_path is not None and self.electrophoresis_image_path is not None:
                self.visualisationImage.setText(None)
                self.set_notify("Загрузка изображения, подождите...")
                self.update_images()
                self.thread_manager.add_task("visualization", self.update_visualization_image)
                self.visualisationImageControlls.setEnabled(True)

    def update_images(self):
        self.electrophoresis_image = Image.open(self.electrophoresis_image_path)
        self.western_blot_image = Image.open(self.western_blot_image_path)

    def update_visualization_image(self):

        resized_image = self.get_resized_visualization_image()
        self.visualisationImage.setStyleSheet("background-color: none")
        # Image.fromarray(resized_image).show()
        o_pixmap = convert_to_pixmap(resized_image)

        o_pixmap = o_pixmap.scaledToWidth(self.visualisationImage.width())
        o_pixmap = o_pixmap.scaledToHeight(self.visualisationImage.height())

        self.visualisationImage.setPixmap(o_pixmap)
        self.startLoadingButton.setEnabled(True)
        self.openOutputImageButton.clicked.connect(lambda: self.open_image(self.get_original_visualization_image()))
        self.clear_notify()

    def get_original_visualization_image(self):
        electrophoresis_image = load_image_as_np(self.electrophoresis_image_path)
        western_blot_image = load_image_as_np(self.western_blot_image_path)
        height, width, channels = western_blot_image.shape
        # Смещение второго изображения
        x_offset = int(
            self.marginHorizontalSlider.value() / 100 * height)  # Смещение по горизонтали на n пикселей
        y_offset = int(
            self.marginVerticalSlider.value() / 100 * height)  # Смещение по вертикали на n пикселей

        # Создание нового изображения
        new_western_blot_image = np.zeros_like(western_blot_image)

        # Определение границ копирования с учетом смещения
        x_start = max(0, x_offset)
        x_end = min(width, width + x_offset)
        y_start = max(0, y_offset)
        y_end = min(height, height + y_offset)

        # Копирование сдвинутых пикселей в новое изображение
        new_western_blot_image[y_start:y_end, x_start:x_end] = western_blot_image[
                                                               max(0, -y_offset):min(height, height - y_offset),
                                                               max(0, -x_offset):min(width, width - x_offset)]

        # Заполнение оставшихся областей черным цветом
        if x_offset >= 0 and y_offset >= 0:
            new_western_blot_image[:y_offset, :x_offset] = 0
        elif x_offset < 0 and y_offset < 0:
            new_western_blot_image[y_offset:, x_offset:] = 0

        new_western_blot_image = cv2.resize(new_western_blot_image, electrophoresis_image.shape[:2][::-1])
        blend = np.zeros_like(electrophoresis_image)

        blend[:, :] = cv2.addWeighted(electrophoresis_image, 0.5, new_western_blot_image, 0.5, 0)

        return blend

    def get_resized_visualization_image(self):

        if not self.electrophoresis_image:
            self.electrophoresis_image = Image.open(self.electrophoresis_image_path)
        if not self.western_blot_image:
            self.western_blot_image = Image.open(self.western_blot_image_path)

        if self.electrophoresis_image.width > 1500:
            self.electrophoresis_image = self.electrophoresis_image.resize(
                (1500, int(1500 * (self.electrophoresis_image.height / self.electrophoresis_image.width))))
        elif self.electrophoresis_image.height > 1500:
            self.electrophoresis_image = self.electrophoresis_image.resize(
                (int(1500 * (self.electrophoresis_image.width / self.electrophoresis_image.height)), 1500))
        if self.western_blot_image.width > 1500:
            self.western_blot_image = self.western_blot_image.resize(
                (1500, int(1500 * (self.western_blot_image.height / self.western_blot_image.width))))
        elif self.western_blot_image.height > 1500:
            self.western_blot_image = self.western_blot_image.resize(
                (int(1500 * (self.western_blot_image.width / self.western_blot_image.height)), 1500))

        western_blot_image = convert_pil_to_np(self.western_blot_image)
        electrophoresis_image = convert_pil_to_np(self.electrophoresis_image)

        height, width, channels = western_blot_image.shape
        # Смещение второго изображения
        x_offset = int(
            self.marginHorizontalSlider.value() / 100 * height)  # Смещение по горизонтали на n пикселей
        y_offset = int(
            self.marginVerticalSlider.value() / 100 * height)  # Смещение по вертикали на n пикселей

        # Создание нового изображения
        new_western_blot_image = np.zeros_like(western_blot_image)

        # Определение границ копирования с учетом смещения
        x_start = max(0, x_offset)
        x_end = min(width, width + x_offset)
        y_start = max(0, y_offset)
        y_end = min(height, height + y_offset)

        # Копирование сдвинутых пикселей в новое изображение
        new_western_blot_image[y_start:y_end, x_start:x_end] = western_blot_image[
                                                               max(0, -y_offset):min(height, height - y_offset),
                                                               max(0, -x_offset):min(width, width - x_offset)]

        # Заполнение оставшихся областей черным цветом
        if x_offset >= 0 and y_offset >= 0:
            new_western_blot_image[:y_offset, :x_offset] = 0
        elif x_offset < 0 and y_offset < 0:
            new_western_blot_image[y_offset:, x_offset:] = 0

        new_western_blot_image = cv2.resize(new_western_blot_image, electrophoresis_image.shape[:2][::-1])
        blend = np.zeros_like(electrophoresis_image)

        blend[:, :] = cv2.addWeighted(electrophoresis_image, 0.5, new_western_blot_image, 0.5, 0)

        return blend

    def load_images(self):
        self.set_notify("Анализ данных, подождите...")
        self.startLoadingButton.setEnabled(False)
        self.saveDataButton.setEnabled(False)
        self.openOutputImageButton.setEnabled(False)
        self.loadWesternBlotImageButton.setEnabled(False)
        self.openOutputWesternBlotButton.setEnabled(False)
        self.loadElectrophoresisImageButton.setEnabled(False)
        self.openOutputElectrophoresisButton.setEnabled(False)
        self.offsetsControlls.setEnabled(False)
        def start_analyze():
            self.analyzer.load_images(self.electrophoresis_image_path, self.western_blot_image_path)
            self.analyzer.load_models_detections()

        def stop_analyze():
            self.load_all_images()

        self.thread_manager.add_task("analyze", start_analyze, stop_analyze)

        # self.screensWidget.setCurrentIndex(1)

    def switch_next(self):
        self.screensWidget.setCurrentIndex(1)

    def switch_previous(self):
        self.screensWidget.setCurrentIndex(0)

    def load_all_images(self):

        # e_boxes, e_classes, e_scores = self.analyzer.electrophoresis_model.get_detections()
        # w_boxes, w_classes, w_scores = self.analyzer.western_blot_model.get_detections()

        # self.thread_manager.add_task("analyze", self.analyzer.analyze, read_analyze_data, timer=1)

        boxes, classes, scores, e_boxes, e_classes, e_scores, w_boxes, w_classes, w_scores = self.analyzer.analyze(
            x_offset_hint=self.marginHorizontalSlider.value() / 100,
            y_offset_hint=self.marginVerticalSlider.value() / 100)
        all_e_boxes, all_e_classes, all_e_scores, all_w_boxes, all_w_classes, all_w_scores = self.analyzer.analyze_any()
        if len(all_e_boxes) == 0:
            self.set_error("Программе не удалось распознать белки на снимке электрофореза!")
        elif len(all_w_boxes) == 0:
            self.set_error("Программе не удалось распознать белки на снимке вестерн-блота!")
        else:
            original_image = self.get_original_visualization_image()
            self.out_visualized_image = self.analyzer.visualize(boxes, classes, scores, original_image,
                                                                self.marginHorizontalSlider.value() / 100,
                                                                self.marginVerticalSlider.value() / 100)

            self.out_visualized_electrophoresis_image = self.analyzer.visualize(all_e_boxes, all_e_classes,
                                                                                all_e_scores,
                                                                                self.analyzer.electrophoresis_image,
                                                                                self.marginHorizontalSlider.value() / 100,
                                                                                self.marginVerticalSlider.value() / 100,
                                                                                ELECTROPHORESIS_TYPE)
            self.out_visualized_western_blot_image = self.analyzer.visualize(all_w_boxes, all_w_classes, all_w_scores,
                                                                             self.analyzer.western_blot_image,
                                                                             self.marginHorizontalSlider.value() / 100,
                                                                             self.marginVerticalSlider.value() / 100,
                                                                             WESTERN_BLOT_TYPE)

            # e_pixmap = convert_to_pixmap(electrophoresis_visualized)

            # e_pixmap = e_pixmap.scaledToWidth(self.electrophoresisImage.width())
            # e_pixmap = e_pixmap.scaledToHeight(self.electrophoresisImage.height())
            # w_pixmap = convert_to_pixmap(western_blot_visualized)
            # w_pixmap = w_pixmap.scaledToWidth(self.westernBlotImage.width())
            # w_pixmap = w_pixmap.scaledToHeight(self.westernBlotImage.height())
            # Итоговая визуализация
            o_pixmap = convert_to_pixmap(self.out_visualized_image)
            o_pixmap = o_pixmap.scaledToWidth(self.visualisationImage.width())
            o_pixmap = o_pixmap.scaledToHeight(self.visualisationImage.height())
            self.visualisationImage.setPixmap(o_pixmap)
            # Визуальзация только белков электрофореза
            o_pixmap = convert_to_pixmap(self.out_visualized_electrophoresis_image)
            o_pixmap = o_pixmap.scaledToWidth(self.electrophoresisImage.width())
            o_pixmap = o_pixmap.scaledToHeight(self.electrophoresisImage.height())
            self.electrophoresisImage.setPixmap(o_pixmap)
            # Визуализация только белков вестерн-блота
            o_pixmap = convert_to_pixmap(self.out_visualized_western_blot_image)
            o_pixmap = o_pixmap.scaledToWidth(self.westernBlotImage.width())
            o_pixmap = o_pixmap.scaledToHeight(self.westernBlotImage.height())
            self.westernBlotImage.setPixmap(o_pixmap)
            # self.electrophoresisImage.setPixmap(QPixmap(e_pixmap))
            # self.westernBlotImage.setPixmap(QPixmap(w_pixmap))
            # pil_image = blended_image
            # image = QImage(pil_image.tobytes(), pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
            # pixmap = QPixmap(image)
            # pixmap = pixmap.scaledToWidth(self.visualisationImage.width())
            # pixmap = pixmap.scaledToHeight(self.visualisationImage.height())

            # self.openWesternBlotImageButton.clicked.connect(lambda: self.open_image(western_blot_visualized))
            # self.openElectrophoresisImageButton.clicked.connect(lambda: self.open_image(electrophoresis_visualized))

            self.startLoadingButton.setEnabled(True)
            self.openOutputImageButton.setEnabled(True)
            self.openOutputElectrophoresisButton.setEnabled(True)
            self.openOutputWesternBlotButton.setEnabled(True)
            self.allowAreaCalculationCheckbox.setEnabled(True)
            self.saveDataButton.setEnabled(True)
            self.loadWesternBlotImageButton.setEnabled(True)
            self.loadElectrophoresisImageButton.setEnabled(True)
            self.offsetsControlls.setEnabled(True)
            self.load_info(
                boxes=boxes,
                classes=classes,
                scores=scores,
                e_boxes=e_boxes,
                e_classes=e_classes,
                e_scores=e_scores,
                w_boxes=w_boxes,
                w_classes=w_classes,
                w_scores=w_scores,
                all_e_boxes=all_e_boxes,
                all_e_classes=all_e_classes,
                all_e_scores=all_e_scores,
                all_w_boxes=all_w_boxes,
                all_w_classes=all_w_classes,
                all_w_scores=all_w_scores,
            )
            self.clear_notify()
            # electrophoresis_visualized = visualize(e_boxes, e_classes, e_scores, self.analyzer.electrophoresis_image)
            #
            # western_blot_visualized = visualize(w_boxes, w_classes, w_scores, self.analyzer.western_blot_image)
            # blended_image = np.array(self.load_visualization_image())

    def update_info_in_window(self, analyze_by_area):

        if not analyze_by_area:
            self.info_1.setText(
                f"Доля совпадающих пятен от всех пятен, присутствующих на обоих фото (по количеству): {self.analyzed_info['ratio_common_count'] * 100:.2f}%")
            self.info_2.setText(
                f"Доля совпадающих пятен от всех пятен, присутствующих на 2D-электрофорезе (по количеству): {self.analyzed_info['ratio_e_count'] * 100:.2f}%")
            self.info_3.setText(
                f"Доля совпадающих пятен от всех пятен, присутствующих на Вестерн-блоте (по количеству): {self.analyzed_info['ratio_w_count'] * 100:.2f}%")
            self.info_4.setText(
                f"Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по количеству): {self.analyzed_info['ratio_only_e_count'] * 100:.2f}%")
            self.info_5.setText(
                f"Погрешность (по количеству): {self.analyzed_info['rate_error_by_count'] * 100:.2f}%")
        else:
            self.info_1.setText(
                f"Доля совпадающих пятен от всех пятен, присутствующих на обоих фото (по площади): {self.analyzed_info['ratio_common_area'] * 100:.2f}%")
            self.info_2.setText(
                f"Доля совпадающих пятен от всех пятен, присутствующих на 2D-электрофорезе (по площади): {self.analyzed_info['ratio_e_area'] * 100:.2f}%")
            self.info_3.setText(
                f"Доля совпадающих пятен от всех пятен, присутствующих на Вестерн-блоте (по площади): {self.analyzed_info['ratio_w_area'] * 100:.2f}%")
            self.info_4.setText(
                f"Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по площади): {self.analyzed_info['ratio_only_e_area'] * 100:.2f}%")
            self.info_5.setText(
                f"Погрешность (по площади): {self.analyzed_info['rate_error_by_area'] * 100:.2f}%")

    def load_info(self, **kwargs):
        # try:
        self.analyzed_info = self.analyzer.analyze_info(kwargs["boxes"], kwargs["e_boxes"], kwargs["w_boxes"],
                                                        kwargs["all_e_boxes"], kwargs["all_w_boxes"])
        self.update_info_in_window(self.allowAreaCalculationCheckbox.checkState())
        h_e, w_e, _ = self.analyzer.electrophoresis_image.shape
        h_w, w_w, _ = self.analyzer.western_blot_image.shape
        self.data_for_saving = {"electrophoresis_image_size": (w_e, h_e),
                                "western_blot_image_size": (w_e, h_e),
                                "out_image": self.out_visualized_image,
                                "boxes": kwargs["boxes"],
                                "scores": kwargs["scores"],
                                "coincidence_electrophoresis_boxes": kwargs["e_boxes"],
                                "coincidence_electrophoresis_classes": kwargs["e_classes"],
                                "coincidence_electrophoresis_scores": kwargs["e_scores"],
                                "coincidence_western_blot_boxes": kwargs["w_boxes"],
                                "coincidence_western_blot_classes": kwargs["w_classes"],
                                "coincidence_western_blot_scores": kwargs["w_scores"],
                                "all_electrophoresis_boxes": kwargs["all_e_boxes"],
                                "all_electrophoresis_classes": kwargs["all_e_classes"],
                                "all_electrophoresis_scores": kwargs["all_e_scores"],
                                "all_western_blot_boxes": kwargs["all_w_boxes"],
                                "all_western_blot_classes": kwargs["all_w_classes"],
                                "all_western_blot_scores": kwargs["all_w_scores"],
                                "ratio_common_count": self.analyzed_info['ratio_common_count'] * 100,
                                "ratio_common_area": self.analyzed_info['ratio_common_area'] * 100,
                                "ratio_e_count": self.analyzed_info['ratio_e_count'] * 100,
                                "ratio_e_area": self.analyzed_info['ratio_e_area'] * 100,
                                "ratio_w_count": self.analyzed_info['ratio_w_count'] * 100,
                                "ratio_w_area": self.analyzed_info['ratio_w_area'] * 100,
                                "ratio_only_e_count": self.analyzed_info['ratio_only_e_count'] * 100,
                                "ratio_only_e_area": self.analyzed_info['ratio_only_e_area'] * 100,
                                "rate_error_by_count": self.analyzed_info['rate_error_by_count'] * 100,
                                "rate_error_by_area": self.analyzed_info['rate_error_by_area'] * 100,
                                "classes_codes": {ELECTROPHORESIS_TYPE: "electrophoresis",
                                                  WESTERN_BLOT_TYPE: "western_blot", PROTEIN_TYPE: "proteins"}
                                }
        self.saveDataButton.setEnabled(True)
        # except:
        #     self.set_error("Произошла ошибка распознавания объектов на снимках!")
        # Печать результатов
        # print(
        #     f"Доля совпадающих пятен от всех пятен, присутствующих на 2D-электрофорезе (по площади): {ratio_e_area * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")
        # print(
        #     f"Доля совпадающих пятен от всех пятен, присутствующих на 2D-электрофорезе (по количеству): {ratio_e_count * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")
        # print(
        #     f"Доля совпадающих пятен от всех пятен, присутствующих на Вестерн-блоте (по площади): {ratio_w_area * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")
        # print(
        #     f"Доля совпадающих пятен от всех пятен, присутствующих на Вестерн-блоте (по количеству): {ratio_w_count * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")
        # print(
        #     f"Доля совпадающих пятен от всех пятен, присутствующих на обоих фото (по количеству): {ratio_common_count * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")
        # print(
        #     f"Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по количеству): {ratio_only_e_count * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")
        # print(
        #     f"Доля совпадающих пятен от всех пятен, присутствующих на обоих фото (по площади): {ratio_common_area * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")
        # print(
        #     f"Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по площади): {ratio_only_e_area * 100:.2f} %; e: {w_e * h_e}; w: {w_w * h_w}")

        # ratio = round((len(kwargs['boxes']) / (len(kwargs['e_boxes']) * len(kwargs['w_boxes']))) * 100, 1)
        # self.info_4.setText(
        #     f"Доля белков, присутствующих на обоих фото: ({len(kwargs['boxes'])} к {len(kwargs['e_boxes']) * len(kwargs['w_boxes'])}): {ratio:.3f}%")

    def save_data(self, **kwargs):

        filename = QFileDialog.getExistingDirectory(
            self,
            "Выберите путь сохранения данных обработки снимков",
            ""
        )

        if filename:
            path = Path(filename)
            today = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            document_path = path.joinpath(f"Результаты анализа {today}").absolute()
            self.set_notify("Сохранение...")
            w_e, h_e = kwargs['electrophoresis_image_size']
            with open(f"{document_path}.txt", "w") as document:
                text = f"""
                "Размер снимка 2D - электрофореза: {kwargs['electrophoresis_image_size'][0]}x{kwargs['electrophoresis_image_size'][1]}"
                "Размер снимка вестерн-блота: {kwargs['western_blot_image_size'][0]}x{kwargs['western_blot_image_size'][1]}"
                "Количество белков на снимке 2D - электрофореза: {len(kwargs['all_electrophoresis_boxes'])}"
                "Площадь белков на снимке 2D - электрофореза: {round(get_boxes_area(kwargs['all_electrophoresis_boxes'], w_e, h_e), 2)}"
                "Количество белков на снимке вестерн-блота: {len(kwargs['all_western_blot_boxes'])}"
                "Площадь белков на снимке вестерн-блота: {round(get_boxes_area(kwargs['all_western_blot_boxes'], w_e, h_e), 2)}"
                "Количество совпадающих пятен: {len(kwargs['boxes'])}"
                "Площадь совпадающих пятен: {round(get_boxes_area(kwargs['boxes'], w_e, h_e), 2)}"
                "Доля совпадающих пятен от всех пятен, присутствующих на обоих фото (по количеству): {round(kwargs['ratio_common_count'] * 100, 2)}%"
                "Доля совпадающих пятен от всех пятен, присутствующих на 2D-электрофорезе (по количеству): {round(kwargs['ratio_e_count'] * 100, 2)}%"
                "Доля совпадающих пятен от всех пятен, присутствующих на Вестерн-блоте (по количеству): {round(kwargs['ratio_w_count'] * 100, 2)}%"
                "Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по количеству): {round(kwargs['ratio_only_e_count'] * 100, 2)}%"
                "Погрешность (по количеству): {round(kwargs['rate_error_by_count'] * 100, 2)}%"
                "Доля совпадающих пятен от всех пятен, присутствующих на обоих фото (по площади): {round(kwargs['ratio_common_area'] * 100, 2)}%"
                "Доля совпадающих пятен от всех пятен, присутствующих на 2D-электрофорезе (по площади): {round(kwargs['ratio_e_area'] * 100, 2)}%"
                "Доля совпадающих пятен от всех пятен, присутствующих на Вестерн-блоте (по площади): {round(kwargs['ratio_w_area'] * 100, 2)}%"
                "Доля пятен, присутствующих ТОЛЬКО на 2D-электрофорезе от всех пятен на 2D-электрофорезе (по площади): {round(kwargs['ratio_only_e_area'] * 100, 2)}%"
                "Погрешность (по площади): {round(kwargs['rate_error_by_area'] * 100, 2)}%"
                """
                document.write(text)
            json_path = path.joinpath(f"Результаты анализа {today}").absolute()
            with open(f"{json_path}.json", "w") as document:
                data = self.data_for_saving
                data.pop("out_image")
                json.dump(data, document)
            self.thread_manager.add_task("save image",
                                         lambda: convert_image_to_generium_output(
                                             Image.fromarray(kwargs["out_image"])).save(
                                             path.joinpath(f"Снимок {today}.png").absolute()),
                                         self.clear_notify)

    def set_notify(self, message):
        self.analyzeStatus.setStyleSheet("color: black")
        self.analyzeStatus.setText(message)

    def set_error(self, message):
        self.analyzeStatus.setStyleSheet("color: red")
        self.analyzeStatus.setText(message)

    def clear_notify(self):
        self.analyzeStatus.clear()
        self.analyzeStatus.setStyleSheet("color: black")
