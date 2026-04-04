import cv2
import numpy as np
import matplotlib.pyplot as plt

def erode_images(images, kernel_size=(2, 2), iterations=1):
    """
    Applies morphological erosion to a batch of images.
    In MNIST, digits are white (255) on black background (0).
    Erosion will make the white foreground thinner.
    
    :param images: Numpy array of shape (N, 28, 28) or (N, 784)
    :param kernel_size: Tuple for the erosion kernel size. Default is (2, 2) to avoid completely removing thin digits.
    :param iterations: Number of times erosion is applied.
    :return: Eroded images array of the same shape.
    """
    kernel = np.ones(kernel_size, np.uint8)
    original_shape = images.shape
    
    # Reshape if flat (e.g. 784 to 28x28)
    if len(original_shape) == 2:
        num_images, total_pixels = original_shape
        h = int(np.sqrt(total_pixels))
        w = h
        reshaped = images.reshape(-1, h, w)
    else:
        reshaped = images
    
    eroded_images = []
    for img in reshaped:
        # Convert to uint8 domain if normally float [0, 1]
        is_float = img.dtype in [np.float32, np.float64]
        if is_float:
            img_to_process = (img * 255.0).astype(np.uint8)
        else:
            img_to_process = img.astype(np.uint8)
            
        eroded = cv2.erode(img_to_process, kernel, iterations=iterations)
        
        # Convert back back to float if it was float initially
        if is_float:
            eroded = eroded.astype(np.float32) / 255.0
            
        eroded_images.append(eroded)
    
    eroded_images = np.array(eroded_images)
    
    # Reshape back to flat structure if original was flat
    if len(original_shape) == 2:
        return eroded_images.reshape(original_shape)
    return eroded_images

def visualize_erosion(original_images, eroded_images, num_samples=5):
    """
    Displays a grid with original images on top and eroded images on the bottom.
    
    :param original_images: Original dataset chunk
    :param eroded_images: Eroded dataset chunk
    :param num_samples: Number of random images to visualize
    """
    plt.figure(figsize=(num_samples * 2, 4))
    
    # Choose random indices if we have more images than requested
    indices = np.random.choice(len(original_images), min(num_samples, len(original_images)), replace=False)
    
    for i, idx in enumerate(indices):
        # Original Image
        plt.subplot(2, num_samples, i + 1)
        
        # Ensure it's 2D for imshow
        orig_img = original_images[idx]
        if len(orig_img.shape) == 1:
            orig_dim = int(np.sqrt(orig_img.shape[0]))
            orig_img = orig_img.reshape((orig_dim, orig_dim))
        elif len(orig_img.shape) == 3 and orig_img.shape[-1] == 1:
            orig_img = orig_img[:, :, 0]
            
        plt.imshow(orig_img, cmap='gray')
        plt.title(f"Original {idx}")
        plt.axis('off')
        
        # Eroded Image
        plt.subplot(2, num_samples, num_samples + i + 1)
        
        erod_img = eroded_images[idx]
        if len(erod_img.shape) == 1:
            erod_dim = int(np.sqrt(erod_img.shape[0]))
            erod_img = erod_img.reshape((erod_dim, erod_dim))
        elif len(erod_img.shape) == 3 and erod_img.shape[-1] == 1:
            erod_img = erod_img[:, :, 0]
            
        plt.imshow(erod_img, cmap='gray')
        plt.title("Eroded")
        plt.axis('off')
        
    plt.suptitle("Erosion Preprocessing Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import sys
    import os
    # Add parent directory to path so we can import mnist_loader
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import mnist_loader
    
    print("Loading MNIST dataset for testing using local mnist_loader...")
    training_data, _, _ = mnist_loader.load_data()
    x_train = training_data[0] # (50000, 784)
    
    # We will pick a few random images to visualize
    sample_images = x_train[:40]
    
    print("Applying morphological erosion with 2x2 kernel...")
    eroded_samples = erode_images(sample_images, kernel_size=(2, 2), iterations=1)
    
    print("Visualizing results. Close the matplotlib window to exit.")
    visualize_erosion(sample_images, eroded_samples, num_samples=8)
