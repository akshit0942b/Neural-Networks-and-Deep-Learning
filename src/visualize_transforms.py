import numpy as np
import cv2
import matplotlib.pyplot as plt
import mnist_loader

def apply_sobel(image):
    """
    Applies Histogram Equalization and a Sobel filter.
    Returns the gradient magnitude image.
    """
    img_2d = image.reshape(28, 28)
    img_uint8 = np.uint8(img_2d * 255)
    
    eq_img = cv2.equalizeHist(img_uint8)
    
    sobelx = cv2.Sobel(eq_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(eq_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    
    # Normalize to [0, 1] for visualization
    max_val = np.max(sobel_mag)
    if max_val > 0:
        sobel_mag = sobel_mag / max_val
        
    return sobel_mag

def apply_gaussian_blur(image, ksize=(5, 5), sigmaX=0):
    """
    Applies Gaussian Blur to the image.
    """
    img_2d = image.reshape(28, 28)
    blurred_img = cv2.GaussianBlur(img_2d, ksize, sigmaX)
    return blurred_img

def visualize_transformations(image, label):
    """
    Plots the original image next to its transformed versions.
    """
    img_2d = image.reshape(28, 28)
    
    # Apply transformations
    sobel_img = apply_sobel(image)
    blurred_img = apply_gaussian_blur(image, ksize=(3, 3))
    heavy_blur_img = apply_gaussian_blur(image, ksize=(5, 5))
    very_heavy_blur_img = apply_gaussian_blur(image, ksize=(9, 9))
    # Setup the matplotlib figure
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.title(f"Original (Label: {label})")
    plt.imshow(img_2d, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Gaussian Blur (3x3)")
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Gaussian Blur (5x5)")
    plt.imshow(heavy_blur_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Gaussian Blur (9x9)")
    plt.imshow(very_heavy_blur_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()  # Pauses execution until you close the plotting window

def main():
    print("Loading MNIST data...")
    # We use load_data() to get the images as flat arrays and labels as integers
    training_data, validation_data, test_data = mnist_loader.load_data()
    
    print("Visualizing transformations on the first 3 training images.")
    # Show transformations for the first 3 images sequentially
    for i in range(10):
        image = training_data[0][i]
        label = training_data[1][i]
        
        print(f"Showing image {i + 1}...")
        visualize_transformations(image, label)
        
    print("Done visualizing!")

if __name__ == "__main__":
    main()
