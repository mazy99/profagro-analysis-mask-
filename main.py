import numpy as np
import skimage.io
import skimage.util
import skimage.exposure

# Load the PNG image
png_image = skimage.io.imread('rgb\\665_2024-09-08_08-17.png')

# Load the mask
image_mask = np.load('masks\\task-328-annotation-329-by-1-tag-cloud-0.npy')

print("PNG Image Shape:", png_image.shape)
print("Image Mask Shape:", image_mask.shape)

# Ensure the mask dimensions match the image dimensions
assert png_image.shape[:2] == image_mask.shape, "Размеры масок и изображения не совпадают"

# Apply the mask to the entire image
masked_image = np.where(image_mask[:, :, np.newaxis], 0, png_image)

# Enhance the image
enhanced_image = skimage.exposure.equalize_adapthist(masked_image)

# Save the result
skimage.io.imsave('masked_enhanced_image.png', skimage.util.img_as_ubyte(enhanced_image))

print('Изображение обработано и сохранено.')
