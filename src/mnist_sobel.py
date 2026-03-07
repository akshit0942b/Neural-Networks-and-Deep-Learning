#### Libraries
# Standard library
from collections import defaultdict

# Third-party libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt # Added for visualization

# My libraries
import mnist_loader

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    
    # Let's visualize the very first image in the training data
    # visualize_pipeline(training_data[0][0], training_data[1][0])
    
    avgs = avg_edge_intensities(training_data)
    
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
                      
    print("Baseline classifier using Sobel & Histogram Equalization.")
    print()
    print(f"{num_correct} of {len(test_data[1])} values correct.")
    print()

def visualize_pipeline(image, label):
    """
    Shows the original, equalized, and Sobel-filtered versions of a single image.
    Close the popup window to let the rest of the program continue.
    """
    img_2d = image.reshape(28, 28)
    img_uint8 = np.uint8(img_2d * 255)
    
    eq_img = cv2.equalizeHist(img_uint8)
    
    sobelx = cv2.Sobel(eq_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(eq_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    
    # Plotting the three stages side-by-side
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original (Label: {label})")
    plt.imshow(img_2d, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Histogram Equalization")
    plt.imshow(eq_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Sobel Filter (Edges)")
    plt.imshow(sobel_mag, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() # This will pause the script until you close the image window


def extract_feature(image):
    """
    Takes a flattened MNIST image, reshapes it, applies Histogram Equalization, 
    then a Sobel filter, and returns the total edge intensity.
    """
    img_2d = image.reshape(28, 28)
    img_uint8 = np.uint8(img_2d * 255)
    
    eq_img = cv2.equalizeHist(img_uint8)
    
    sobelx = cv2.Sobel(eq_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(eq_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    
    return np.sum(sobel_mag)


def avg_edge_intensities(training_data):
    """ 
    Return a defaultdict whose keys are the digits 0 through 9.
    For each digit we compute a value which is the average edge intensity.
    """
    digit_counts = defaultdict(int)
    edge_intensities = defaultdict(float)
    
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        edge_intensities[digit] += extract_feature(image)
        
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():
        avgs[digit] = edge_intensities[digit] / n
        
    return avgs


def guess_digit(image, avgs):
    """
    Return the digit whose average edge intensity in the training data is
    closest to the edge intensity of ``image``.
    """
    feature_val = extract_feature(image)
    distances = {k: abs(v - feature_val) for k, v in avgs.items()}
    return min(distances, key=distances.get)

if __name__ == "__main__":
    main()