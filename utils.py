import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import io

# Label mapping for test masks: maps {0, 1, 3, 6} → {0, 1, 2, 3}
label_map = {0: 0, 1: 1, 3: 2, 6: 3}

# Class color dictionary (consistent with final predictions)
CLASS_LABELS = {
    0: {'name': 'Background', 'color': (0, 0, 0)},       # Black
    1: {'name': 'Vegetation', 'color': (34, 139, 34)},   # Forest Green
    2: {'name': 'Slum', 'color': (255, 165, 0)},         # Orange
    3: {'name': 'Water', 'color': (30, 144, 255)},       # Dodger Blue
}

# Function to map the mask labels (for ground truth or predictions)
def map_mask_labels(mask):
    """
    Map the test dataset labels to internal labels used by model training.
    """
    mapped_mask = np.zeros_like(mask)
    for old_val, new_val in label_map.items():
        mapped_mask[mask == old_val] = new_val
    return mapped_mask

# Function to preprocess the image for the model
def preprocess_image(uploaded_image):
    # Read the image using PIL (this works well with TIFF images)
    image = Image.open(uploaded_image)
    
    # Convert the image to RGB if it has multiple channels
    image = image.convert("RGB")
    
    # Convert to NumPy array for OpenCV to work with
    image_array = np.array(image)
    
    # Resize the image to (120, 120)
    resized_image = cv2.resize(image_array, (120, 120))
    
    # Normalize the image to [0, 1]
    resized_image = resized_image / 255.0
    
    return resized_image

# Function to preprocess the ground truth mask
def preprocess_mask(uploaded_mask):
    # Read the mask image using PIL
    mask = Image.open(uploaded_mask)
    
    # Convert to NumPy array (PNG masks are often single channel, so no need to convert to RGB)
    mask_array = np.array(mask)
    
    # Check if the mask has more than 1 channel (e.g., RGBA), convert it to single-channel grayscale if needed
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Keep the first channel (if it's RGB/RGBA)
    
    # Resize the mask to (120, 120) using nearest interpolation (important for segmentation masks)
    resized_mask = cv2.resize(mask_array, (120, 120), interpolation=cv2.INTER_NEAREST)
    
    # Map the mask labels according to label_map (for model training)
    mapped_mask = map_mask_labels(resized_mask)
    
    return mapped_mask

# Function to decode the model's predicted segmentation mask to color
def decode_segmentation(mask, class_labels=CLASS_LABELS):
    """
    Convert predicted segmentation mask to a color image.
    """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, label_info in class_labels.items():
        color_mask[mask == class_id] = label_info['color']

    return color_mask
