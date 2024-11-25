import numpy as np
import cv2

class Indexes:
    '''
    Класс для работы с вегетативными индексами
    '''
    def __init__(self):
        # имена доступных индексов для рассчёта
        # TODO: добавить индекс REP
        self.available_index_names = [
            "NDWI",
            "NDMI",
            "NDVI",
            "SR",
            # "REP",
            "EVI",
            "EVI2",
            "ARVI",
            "SAVI",
            "GOSAVI",
            "GARI",
            "VARI",
        ]

    def get_available_index_names(self):
        return self.available_index_names

    def get_index(self, index_name, img):
        """
        Рассчёт индекса по его имени
        index_name: имя индекса из списка доступных
        img: 15-ти или 14-ти канальное иозбражения со спутника Sentinel-2, в стандартной последовательности band-ов

        return: одноканальное изображение-индекс, все none переведены в нули
        """
        assert index_name in self.available_index_names, "Index name not available to calculate!" \
                                                         " Check 'available_index_names'."

        if index_name == "NDWI":
            NIR_index = img[:, :, 2:3]
            RED_index = img[:, :, 7:8]
            INDEX_img = (-NIR_index + RED_index) / (NIR_index + RED_index)

        if index_name == "NDMI":
            NIR_index = img[:, :, 7:8]
            RED_index = img[:, :, 10:11]
            INDEX_img = (NIR_index - RED_index) / (NIR_index + RED_index)

        if index_name == "NDVI":
            NIR_index = img[:, :, 7:8]
            RED_index = img[:, :, 3:4]
            INDEX_img = (NIR_index - RED_index) / (NIR_index + RED_index)

        if index_name == "SR":
            NIR_index = img[:, :, 7:8]
            RED_index = img[:, :, 3:4]
            INDEX_img = NIR_index / RED_index

        if index_name == "EVI":
            NIR_index = img[:, :, 7:8]
            RED_index = img[:, :, 3:4]
            BLUE_index = img[:, :, 1:2]
            INDEX_img = 2.5 * (NIR_index - RED_index) / (NIR_index + 6 * RED_index - 7.5 * BLUE_index + 1.0)

        if index_name == "EVI2":
            NIR_index = img[:, :, 7:8]
            RED_index = img[:, :, 3:4]
            INDEX_img = 2.5 * (NIR_index - RED_index) / (NIR_index + RED_index + 1.0)

        if index_name == "ARVI":
            B8A = img[:, :, 8:9]
            B04 = img[:, :, 3:4]
            B02 = img[:, :, 1:2]
            y = 2
            INDEX_img = (B8A - B04 - y * (B04 - B02)) / (B8A + B04 - y * (B04 - B02))

        if index_name == "SAVI":
            NIR_index = img[:, :, 7:8]
            RED_index = img[:, :, 3:4]
            INDEX_img = 1.5 * (NIR_index - RED_index) / (NIR_index + RED_index + 0.5)

        if index_name == "GOSAVI":
            NIR_index = img[:, :, 7:8]
            GREEN_index = img[:, :, 2:3]
            INDEX_img = (NIR_index - GREEN_index) / (NIR_index + GREEN_index + 0.16)

        if index_name == "GARI":
            NIR_index = img[:, :, 7:8]
            RED_index = img[:, :, 3:4]
            GREEN_index = img[:, :, 2:3]
            BLUE_index = img[:, :, 1:2]
            INDEX_img = (NIR_index - (GREEN_index - (BLUE_index - RED_index))) / (
                        NIR_index + (GREEN_index - (BLUE_index - RED_index)))

        if index_name == "VARI":
            RED_index = img[:, :, 3:4]
            GREEN_index = img[:, :, 2:3]
            BLUE_index = img[:, :, 1:2]
            INDEX_img = (GREEN_index - RED_index) / (GREEN_index + RED_index - BLUE_index)

        return np.nan_to_num(INDEX_img)

    def create_all_indexes_from_img(self, img):
        """
        Рассчёт всех индексов для снимка
        img: 15-ти или 14-ти канальное иозбражения со спутника Sentinel-2, в стандартной последовательности band-ов

        return: словарь, где по ключу - имени индекса лежит значени - изображние-индекс
        """
        index_by_names = {}
        for index_name in self.available_index_names:
            index_by_names[index_name] = self.get_index(index_name, img)
        return index_by_names

    def get_color_from_index(self,  index):
        """
        Отображение индекса в зелёном диапазоне для наглядной визуализации
        """
        norm_index = cv2.normalize(index, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        h, w = norm_index.shape
        color_index = np.zeros((h, w, 3))
        color_index[:, :, 1] = norm_index
        return color_index


# ПРИМЕР ИСПОЛЬЗОВАНИЯ!
if __name__ == "__main__":
    indexes = Indexes()

    # Вывод на консоль допустимых имён индексов, для которых предусмотрена реализация
    print(indexes.get_available_index_names())

    # читаем 15тиканальное изображение
    from tifffile import imread, imwrite
    path_to_original_img = "tiff\\665_2024-09-08_08-17.tiff"

    # Извлечём индекс SR из снимка
    original_img = imread(path_to_original_img)
    SR_index = indexes.get_index("SR", original_img)

    # сохраним этот индекс в tiff и в цвете
    imwrite("F:\\AIagriculture\\tiff_image", SR_index)
    cv2.imwrite("F:\\AIagriculture\\tiff_image", indexes.get_color_from_index(SR_index))

    # Извлечём все доступные индексы из изображения и сохраним каждый по названию индекса
    all_indexes_from_one_image = indexes.create_all_indexes_from_img(original_img)
    for index_name, index_image in all_indexes_from_one_image.items():
        imwrite(f"tiff/test_{index_name}.tiff", index_image)