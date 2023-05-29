import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2
from skimage import color, restoration

image = cv2.imread('images/image_original_colored.png')
image = plt.imread('images/image_original_colored.png')

plt.imshow(image)
plt.show()

# Print error message if image is null
if image is None:
    print('Could not read image')

rng = np.random.default_rng()
# image = image.astype(float)  # / 255.0

# image_grayscale = color.rgb2gray(image)
psf = np.ones((5, 5)) / 25

# Grayscale version
image_grayscale = color.rgb2gray(image)
image_grayscale = conv2(image_grayscale, psf, 'same')
image_grayscale += 0.1 * image_grayscale.std() * rng.standard_normal(image_grayscale.shape)
deconvolved_grayscale, _ = restoration.unsupervised_wiener(image_grayscale, psf)

# Color version
deconvolved = np.zeros_like(image)
for i in range(3):
    image[:, :, i] = conv2(image[:, :, i], psf, 'same')
    image[:, :, i] += 0.1 * image[:, :, i].std() * rng.standard_normal(image[:, :, i].shape)

    deconvolved[:, :, i], _ = restoration.unsupervised_wiener(image[:, :, i], psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)

ax[0].imshow(image, vmin=0, vmax=1)
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()