import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def GaussianFilter():
    sigma = 2
    size = (13, 13)
    x, y = np.mgrid[-(math.floor(size[0]/2)):(math.ceil(size[0]/2)), 
                    -(math.floor(size[0]/2)):(math.ceil(size[0]/2))]
    kernel = np.exp(-(x**2+y**2) / (2 * sigma**2))
    kernel = kernel/kernel.sum()
    return kernel

def padding(img, size=(1, 1)):
    r_pad, c_pad = size
    row, column, ch = img.shape
    img_pad = np.zeros((row + 2 * r_pad, column + 2 * c_pad, ch), dtype=img.dtype)
    
    img_pad[r_pad:-r_pad, c_pad:-c_pad] = img
    img_pad[:r_pad, c_pad:-c_pad] = img[0:1, :, :] 
    img_pad[-r_pad:, c_pad:-c_pad] = img[-1:, :, :] 
    img_pad[:, :c_pad] = img_pad[:, c_pad:c_pad+1]
    img_pad[:, -c_pad:] = img_pad[:, -c_pad-1:-c_pad] 
    img_pad[:r_pad, :c_pad] = img[0, 0] 
    img_pad[:r_pad, -c_pad:] = img[0, -1]
    img_pad[-r_pad:, :c_pad] = img[-1, 0]
    img_pad[-r_pad:, -c_pad:] = img[-1, -1] 
    
    return img_pad

def convolution(img, kernel):
    k_y, k_x = kernel.shape
    p_y = (k_y - 1) // 2
    p_x = (k_x - 1) // 2
    img = padding(img, (p_y, p_x))
    row, column, ch = img.shape
    output = np.copy(img)
    for y in range(p_y, row - p_y):
        for x in range(p_x, column - p_x):
            for c in range(ch):
                img_window = img[(y - p_y):(y + p_y) + 1, (x - p_x):(x +p_x) + 1, c]
                output[y, x, c] = np.sum(img_window * kernel)   
    
    output = output[p_y:-p_y, p_x:-p_x]
    return output

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

def spectrum_transfer(img):
    row, column, ch = img.shape
    for c in range(ch):
        dft = np.fft.fft2(img[:, :, c])
        spectrum = np.log(np.abs(np.fft.fftshift(dft)))
    return spectrum

def image_pyramid(img, layer):
    Gaussian = img
    for i in range(layer):
        # Gaussian pyramid
        previous_layer = Gaussian
        Gaussian = convolution(previous_layer, GaussianFilter())
        Gaussian = subsampling(Gaussian)
        cv2.imwrite(f'output/task2/data_output/Gaussian_layer_{i}.jpg', Gaussian)

        spectrum = spectrum_transfer(Gaussian)
        plt.imshow(spectrum)
        plt.axis('off')
        plt.savefig(f'output/task2/data_output/Gaussian_spectrum_{i}.jpg', bbox_inches='tight', pad_inches=0)

        # Laplacian pyramid
        highImg = previous_layer
        lowImg = upsampling(Gaussian, previous_layer)
        Laplacian = highImg - lowImg
        cv2.imwrite(f'output/task2/data_output/Laplacian_layer_{i}.jpg', Laplacian)

        spectrum = spectrum_transfer(Laplacian)
        plt.imshow(spectrum)
        plt.axis('off')
        plt.savefig(f'output/task2/data_output/Laplacian_spectrum_{i}.jpg', bbox_inches='tight', pad_inches=0)

# main
layer = 4
img = cv2.cvtColor(cv2.imread("data/task1and2_hybrid_pyramid/3_cat.bmp",), cv2.COLOR_RGB2GRAY)
# img = cv2.cvtColor(cv2.imread("my_data/test.jpg",), cv2.COLOR_RGB2GRAY)
img = img.reshape(img.shape[0],img.shape[1],1)
image_pyramid(img, layer)