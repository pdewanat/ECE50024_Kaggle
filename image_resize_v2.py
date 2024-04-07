"""
@author: Pandu Dewanatha
"""

import cv2
import os
import numpy as np
from PIL import Image

def resize_pad(img, size=(224, 224), pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    # keep aspect ratio
    aspect = w/h

    # scaling and padding
    if aspect > 1: # if width > height
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) // 2
        pad_top, pad_bot = pad_vert, pad_vert
        pad_left, pad_right = 0, 0
    elif aspect < 1: # if height > width
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) // 2
        pad_left, pad_right = pad_horz, pad_horz
        pad_top, pad_bot = 0, 0
    else: # if the aspect ratio is already 1
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # If its the correct size
    if scaled_img.shape[0] == 224 and scaled_img.shape[1] == 224:
        return scaled_img
    
    # generate padded image
    padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[pad_color, pad_color, pad_color])

    return padded_img


# Load picture folder
folder_path = "C:/Users/pdewanat/Desktop/ECE 50024/Kaggle/test/test_filtered_resized"
counter = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        file_path = os.path.join(folder_path, filename)
        
        # Open with PIL
        try:
            pil_image = Image.open(file_path).convert('RGB')
            image = np.array(pil_image)  # Convert PIL image to a numpy array
            image = image[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
        except IOError:
            print(f"Failed to load {file_path}.")
            continue
        if image is None or image.size == 0:
            print(f"Failed to load {file_path}.")
            continue

        resized_image = resize_pad(image, size=(224, 224))
        cv2.imwrite(file_path, resized_image)  # Overwrite image
        counter += 1
        
        # I made a counter to let me know every 1000 processed image
        if counter % 1000 == 0:
            print(f"Processed {counter} images.")

print("Done.")