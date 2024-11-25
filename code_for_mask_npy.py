import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.transform import resize

image_path = "D:/профагро проект/изображения для  разметки профагро(1209 поле)/118_2018-05-21_08-22.png" 
mask_path = "npy_mask/task-26-annotation-28-by-1-tag-сloud-0.npy"   
output_image_path = "output/overlayed_image.png"  


mask = np.load(mask_path)


with rasterio.open(image_path) as src:
    image = src.read()  
    profile = src.profile


image_for_display = image[0, :, :]  
image_for_display = (image_for_display - image_for_display.min()) / (image_for_display.max() - image_for_display.min())

print(image_for_display)

if mask.shape != image.shape[1:]:

     mask = resize(mask, (image.shape[1], image.shape[2]), order=0, preserve_range=True).astype(int)


mask_colored = label2rgb(mask, image=image_for_display, bg_label=0, alpha=0.6)

print(mask_colored)

plt.figure(figsize=(10, 10))
plt.imshow(mask_colored)
plt.title("Image with Mask Overlay", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.savefig(output_image_path)
plt.show()

print(f"Изображение с наложенной маской сохранено в {output_image_path}")
