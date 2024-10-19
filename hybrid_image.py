import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper functions
def fftshift(image):
    rows, cols = image.shape
    shifted_image = np.copy(image)
    for x in range(rows):
        for y in range(cols):
            shifted_image[x, y] = image[x, y] * ((-1) ** (x + y))
    return shifted_image

def apply_gaussian_filter(fft_image, filter_type="low", d=30):
    """Applies a Gaussian filter to the Fourier transformed image."""
    rows, cols = fft_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if filter_type == "low":
                # Low-pass Gaussian filter
                mask[i, j] = np.exp(-(dist ** 2) / (2 * (d ** 2)))
            elif filter_type == "high":
                # High-pass Gaussian filter
                mask[i, j] = 1 - np.exp(-(dist ** 2) / (2 * (d ** 2)))

    return fft_image * mask

def show_image(title, image):
    """Helper function to show an image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Resize images to the smallest size of the pair
def resize_images_to_smallest(image1, image2):
    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape
    new_rows = min(rows1, rows2)
    new_cols = min(cols1, cols2)
    
    image1_resized = cv2.resize(image1, (new_cols, new_rows), interpolation=cv2.INTER_AREA)
    image2_resized = cv2.resize(image2, (new_cols, new_rows), interpolation=cv2.INTER_AREA)
    
    return image1_resized, image2_resized

# Main function to create hybrid images for each pair in the folder
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
            print(f"Processing pair: {file_pair[0]} and {file_pair[1]}")
            
            image1 = cv2.imread(os.path.join(folder_path, file_pair[0]), 0)
            image2 = cv2.imread(os.path.join(folder_path, file_pair[1]), 0)
            if image1 is None or image2 is None:
                print(f"Error loading images: {file_pair}")
                continue

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

            # Display or save the final hybrid image
            show_image(f"Hybrid Image (Pair {number})", final_image)
            # Optionally save the image: uncomment the line below
            cv2.imwrite(os.path.join(r'C:\Users\IT_manager\Desktop\2024 Autumn\CV\CV_HW2\output\task1\data_output', f"hybrid_{number}.png"), final_image)

# Example usage
folder_path = r'C:\Users\IT_manager\Desktop\2024 Autumn\CV\CV_HW2\data\task1and2_hybrid_pyramid'  # Replace with the path to your folder
create_hybrid_images(folder_path)  # You can adjust the filter_d for more blurring
