import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def GaussianFilter():
    sigma = 2
    size = 6*sigma+1
    if size % 2 == 0:
        size += 1
    center = size // 2
    x = np.arange(size) - center
    y = np.arange(size) - center
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2+y**2) / (2 * sigma**2))
    kernel = kernel/kernel.sum()
    return kernel

def convolution(img, filterG):
    filter_h, filter_w = filterG.shape
    start_h, start_w = (img.shape[0] - filter_h) // 2, (img.shape[1] - filter_w) // 2
    pad_filter = np.zeros(img.shape[:2])
    pad_filter[start_h : start_h + filter_h, start_w : start_w + filter_w] = filterG

    filt_fft = np.fft.fft2(pad_filter)

    result = np.zeros(img.shape)
    for color in range(img.shape[2]):
        img_fft = np.fft.fft2(img[:, :, color])
        result[:, :, color] = np.fft.fftshift(np.fft.ifft2(img_fft * filt_fft)).real
    return result

def subsampling(img):
    newImg = np.zeros((img.shape[0]//2, img.shape[1]//2, img.shape[2]))
    for i in range(newImg.shape[0]):
        for j in range(newImg.shape[1]):
            newImg[i][j] = img[2*i][2*j]
    return newImg
    
def upsampling(img, previous_layer):
    row, column, ch = img.shape
    size = previous_layer.shape
    r = size[0] // row
    c = size[1] // column
    
    newImg = np.zeros(size)
    for i in range(row):
        for j in range(column):
            newImg[i*r:i*r+r+1, j*c:j*c+c+1] = img[i, j]
    return newImg

def img_to_spectrum(img):
    result = np.zeros(img.shape)
    if len(img.shape) == 3:
        result = np.fft.fft2(img, axes=(0, 1))
    else:
        result = np.fft.fft2(img)
    spectrum = np.log(1 + np.abs(np.fft.fftshift(result)))
    img_min, img_max = spectrum.min(), spectrum.max()
    return (spectrum - img_min) / (img_max - img_min) 

def image_pyramid(img, layer):
    for i in range(layer):
        previous_layer = img
        img = convolution(img, GaussianFilter())
        save_and_display(img, f'result/Gaussian_layer_{i}.jpg')

        if i == layer:
            Laplacian = img
        else:
            img = subsampling(img)
            upImg = upsampling(img, previous_layer)
            Laplacian = previous_layer - upImg
        save_and_display(Laplacian, f'result/Laplacian_layer_{i}.jpg')

def save_and_display(image, filename):
    if not os.path.exists('result'):
        os.makedirs('result')
    cv2.imwrite(filename, image)    
    spectrum = img_to_spectrum(image)
    plt.imshow(spectrum)
    plt.title(filename)
    plt.axis('off')
    plt.show()

# main
layer = 5
img = cv2.imread(f'data/test.jpg')
image_pyramid(img, layer)