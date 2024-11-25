import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.transform import resize


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_shape = None
        self.profile = None
        self.image = None

    def load_image(self):
        with rasterio.open(self.image_path) as src:
            self.image = src.read()  
            self.profile = src.profile
            self.image_shape = self.image.shape

    def normalize(self):
        image_for_display = self.image[0, :, :]
        return (image_for_display - image_for_display.min()) / (image_for_display.max() - image_for_display.min())

class MacroProcessor:
    def __init__(self, mask_path, image_shape):
        self.mask_path = mask_path
        self.image_shape = image_shape
        self.mask = None

    def load_mask(self):
        self.mask = np.load(self.mask_path)

    def resize_mask(self):
        if self.mask.shape != self.image_shape[1:]:
            self.mask = resize(self.mask, (self.image_shape[1], self.image_shape[2]), order=0, preserve_range=True).astype(int)
    
    def apply_mask(self, image_for_display):
         return label2rgb(self.mask, image=image_for_display, bg_label=0, alpha=0.6)

class Visualizer:
    def __init__(self, output_path):
        self.output_path = output_path
    
    def show(self, mask_colored):
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_colored)
        plt.title("Image with Mask Overlay", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.show()

class OverlayImage:
    def __init__(self, image_path, mask_path, output_image_path):
        self.image_processor = ImageProcessor(image_path)
        self.mask_processor = None
        self.visualizer = Visualizer(output_image_path)
    
    def processor(self):
        self.image_processor.load_image()
        image_for_display = self.image_processor.normalize()
        self.mask_processor = MacroProcessor(mask_path, self.image_processor.image_shape)
        self.mask_processor.load_mask()
        self.mask_processor.resize_mask()
        mask_colored = self.mask_processor.apply_mask(image_for_display)
        self.visualizer.show(mask_colored)


image_path = "D:/профагро проект/изображения для  разметки профагро(1209 поле)/10_2017-03-12_08-19.png"  
mask_path = "npy_mask/task-2-annotation-5-by-1-tag-сloud-0.npy"   
output_image_path = "output/overlayed_image.png"  

overlay = OverlayImage(image_path, mask_path, output_image_path)
overlay.processor()

print(f"Изображение с наложенной маской сохранено в {output_image_path}")
