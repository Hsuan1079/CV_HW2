import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# (-1)^(x+y)
def fftshift(image):
    rows, cols = image.shape
    shifted_image = np.copy(image)
    for x in range(rows):
        for y in range(cols):
            shifted_image[x, y] = image[x, y] * ((-1) ** (x + y))
    return shifted_image

def apply_gaussian_filter(fft_image, filter_type="low", d=30):
    rows, cols = fft_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if filter_type == "low":
                mask[i, j] = np.exp(-(dist ** 2) / (2 * (d ** 2)))
            elif filter_type == "high":
                mask[i, j] = 1 - np.exp(-(dist ** 2) / (2 * (d ** 2)))

    return fft_image * mask

def apply_ideal_filter(fft_image, filter_type="low", d=30):
    rows, cols = fft_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    
    if filter_type == "low":
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) > d:
                    mask[i, j] = 0
    elif filter_type == "high":
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) < d:
                    mask[i, j] = 0
    
    return fft_image * mask

def resize_images_to_smallest(image1, image2):
    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape
    new_rows = min(rows1, rows2)
    new_cols = min(cols1, cols2)
    
    image1_resized = cv2.resize(image1, (new_cols, new_rows), interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, (new_cols, new_rows), interpolation=cv2.INTER_AREA)
    
    return image1_resized, image2_resized

def gaussian_hybrid(image1, image2):
    # Resize the images to the smallest shape
    image1, image2 = resize_images_to_smallest(image1, image2)

    # Step 1: Shift by multiplying (-1)^(x+y)
    image1_centered = fftshift(image1)
    image2_centered = fftshift(image2)

    # Step 2: Fourier Transform
    fft_image1 = np.fft.fft2(image1_centered)
    fft_image2 = np.fft.fft2(image2_centered)

    # Step 3: Apply Gaussian filters (low-pass to image1, high-pass to image2)
    fft_image1_gaussian = apply_gaussian_filter(fft_image1, "low", d=10)  # Low-pass
    fft_image2_gaussian = apply_gaussian_filter(fft_image2, "high", d=30)  # High-pass

    # Step 4: Combine the two filtered images (hybrid image creation)
    combined_fft = fft_image1_gaussian + fft_image2_gaussian

    # Step 5: Inverse Fourier Transform
    combined_image = np.fft.ifft2(combined_fft)
    combined_image = np.real(combined_image)

    # Step 6: Multiply result by (-1)^(x+y) to shift back
    final_image = fftshift(combined_image)
    return final_image

def ideal_hybrid(image1, image2):
    # Resize the images to the smallest shape
    image1, image2 = resize_images_to_smallest(image1, image2)

    # Step 1: Shift by multiplying (-1)^(x+y)
    image1_centered = fftshift(image1)
    image2_centered = fftshift(image2)

    # Step 2: Fourier Transform
    fft_image1 = np.fft.fft2(image1_centered)
    fft_image2 = np.fft.fft2(image2_centered)

    # Step 3: Apply Gaussian filters (low-pass to image1, high-pass to image2)
    fft_image1_gaussian = apply_ideal_filter(fft_image1, "low", d=10)  # Low-pass
    fft_image2_gaussian = apply_ideal_filter(fft_image2, "high", d=30)  # High-pass

    # Step 4: Combine the two filtered images (hybrid image creation)
    combined_fft = fft_image1_gaussian + fft_image2_gaussian

    # Step 5: Inverse Fourier Transform
    combined_image = np.fft.ifft2(combined_fft)
    combined_image = np.real(combined_image)

    # Step 6: Multiply result by (-1)^(x+y) to shift back
    final_image = fftshift(combined_image)
    return final_image

def create_hybrid_images(folder_path):
    files = sorted(os.listdir(folder_path))

    pairs = {}
    for file in files:
        number = file.split('_')[0]
        if number not in pairs:
            pairs[number] = []
        pairs[number].append(file)
    
    # Process each pair
    for number, file_pair in pairs.items():
        if len(file_pair) == 2:
            image1 = cv2.imread(os.path.join(folder_path, file_pair[0]), 0)
            image2 = cv2.imread(os.path.join(folder_path, file_pair[1]), 0)
            
            if image1 is None or image2 is None:
                print(f"Error loading images: {file_pair}")
                continue

            final_gaussian_image = gaussian_hybrid(image1, image2)
            final_ideal_image = ideal_hybrid(image1, image2)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(final_ideal_image, cmap='gray')
            plt.title('Hybrid Image (Ideal Filter)')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(final_gaussian_image, cmap='gray')
            plt.title('Hybrid Image (Gaussian Filter)')
            plt.axis('off')
            plt.savefig(os.path.join(r'output\task1\data_output', f"hybrid_{number}.png"))
            plt.show()

folder_path = r'data\task1and2_hybrid_pyramid'
create_hybrid_images(folder_path)
