import numpy as np
import skimage.io
import skimage.util
import skimage.exposure
import skimage.color
import os

tiff_image = skimage.io.imread('tiff\\665_2024-09-08_08-17.tiff')

image_mask = np.load('masks\\task-328-annotation-329-by-1-tag-shade-0.npy')

print("TIFF Image Shape:", tiff_image.shape)
print("Image Mask Shape:", image_mask.shape)

assert tiff_image.shape[:2] == image_mask.shape, "Размеры масок и изображения не совпадают"
expanded_mask = np.repeat(image_mask[:, :, np.newaxis], tiff_image.shape[2], axis=2)

photo_name = '665_2024-09-08_08-17'
output_directory = os.path.join('channels_photo', photo_name)

os.makedirs(output_directory, exist_ok=True)

for channel_index in range(tiff_image.shape[2]):
    channel_image = tiff_image[:, :, channel_index]

    masked_image = np.where(expanded_mask[:, :, channel_index], 0, channel_image)

    enhanced_image = skimage.exposure.equalize_adapthist(masked_image)

    output_path = os.path.join(output_directory, f'channel_{channel_index}_masked.png')
    skimage.io.imsave(output_path, skimage.util.img_as_ubyte(enhanced_image))

    print(f'Канал {channel_index} обработан.')

