from PIL import Image
import numpy as np
import cv2


class CLAHE:
    
    def __init__(self, clip_limit=1.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        # Convert PIL image to NumPy array
        image_np = np.array(image)
        
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        # Apply CLAHE to each channel separately if it's a color image
        if len(image_np.shape) == 3:  # Color image
            channels = cv2.split(image_np)
            clahe_channels = [clahe.apply(ch) for ch in channels]
            clahe_image_np = cv2.merge(clahe_channels)
        else:  # Grayscale image
            clahe_image_np = clahe.apply(image_np)
        
        # Convert back to PIL image
        clahe_image = Image.fromarray(clahe_image_np)
        
        return clahe_image