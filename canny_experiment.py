""" Окна оптимизированы для работы в Windows среде. Если работать через WSL, то окна могут быть разбросаны.
"""

from __future__ import print_function
import os
import cv2 as cv
import numpy as np

input_folder = "data/temp"

thresh_1_default = 54
thresh_2_default = 80
apertureSize_default = 3
blur_size_default = 3

threshold_1 = thresh_1_default
threshold_2 = thresh_2_default
apertureSize = apertureSize_default
blur_size = blur_size_default

blur = []
gray = []

def thresh_1_callback(val):
    global threshold_1
    threshold_1 = val
    update()

def thresh_2_callback(val):
    global threshold_2
    threshold_2 = val
    update()

def aperture_size_callback(val):
    global apertureSize
    if val < 3:
        val = 3
    elif val > 7:
        val = 7
    elif val % 2 != 1:
        val += 1

    apertureSize = val
    update()

def blur_kernel_callback(val):
    global blur_size
    if val < 3:
        val = 3
    elif val > 15:
        val = 15
    elif val % 2 != 1:
        val += 1

    blur_size = val
    on_blur()
    update()

def on_blur():
    global blur
    global src
    blur = []
    for img in gray:
        blur.append(cv.blur(img, (blur_size,blur_size)))


def update():
    ## [Canny]
    # Detect edges using Canny
    canny_output = []
    for image in blur:
        canny_output.append(cv.Canny(image, threshold_1, threshold_2, apertureSize=apertureSize))
    show_images('canny', canny_output, 1)
    # [Canny]

def show_images(name, images, column):
    for i, image in enumerate(images):
        show_image(f"{name}_{i}", image, i % 3, column + 2*(i//3))

def show_image(name, image, row, column):
    cv.imshow(name, image)
    cv.moveWindow(name, 360*column, 150 + 300*row)

## [setup]
# Load source image
img_files = [f.path for f in os.scandir(input_folder) if f.is_file()]
src = []

image_files_for_show = img_files

if len(image_files_for_show) > 6:
    image_files_for_show = image_files_for_show[:6]

for i, image_name in enumerate(image_files_for_show):
    img = cv.imread(image_name)
    if img is None:
        print('Could not open or find the image:', image_name)
        exit(0)
    src.append(img)
show_images('source', src, 0)

# Convert image to gray and blur it
for img in src:
    gray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
on_blur()
## [setup]

## [createWindow]
max_thresh = 255
track_window = 'Params'
cv.namedWindow(track_window, cv.WINDOW_NORMAL)
cv.imshow(track_window, np.zeros((1,1500,3) , dtype=np.uint8))
cv.createTrackbar('Aperture Size:', track_window, apertureSize_default, 7, aperture_size_callback)
cv.createTrackbar('Blur size:', track_window, blur_size_default, 15, blur_kernel_callback)
cv.createTrackbar('Thresh 1:', track_window, thresh_1_default, max_thresh, thresh_1_callback)
cv.createTrackbar('Thresh 2:', track_window, thresh_2_default, max_thresh, thresh_2_callback)
cv.moveWindow(track_window, 0, 0)
cv.resizeWindow(track_window, 1500, 50) 


thresh_1_callback(thresh_1_default)
thresh_2_callback(thresh_2_default)
aperture_size_callback(apertureSize_default)
blur_kernel_callback(blur_size_default)

cv.waitKey()