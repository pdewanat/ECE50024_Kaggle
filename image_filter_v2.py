"""
@author: Pandu Dewanatha
"""

import cv2
import os
import numpy as np
from PIL import Image

# Image folder
folder_path = "C:/Users/pdewanat/Desktop/ECE 50024/Kaggle/train_small/test"

# Pre-trained face detection that I found online
model = "C:/Users/pdewanat/Desktop/ECE 50024/Kaggle/deploy.prototxt.txt"
weights = "C:/Users/pdewanat/Desktop/ECE 50024/Kaggle/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(model, weights)

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        file_path = os.path.join(folder_path, filename)
        
        # Open with PIL
        try:
            pil_image = Image.open(file_path).convert('RGB')
            image = np.array(pil_image)  
            image = image[:, :, ::-1].copy()
        except IOError:
            print(f"Failed to load {file_path}.")
            continue
        if image is None or image.size == 0:
            print(f"Failed to load {file_path}.")
            continue
        
        # Face Detection
        (h, w) = image.shape[:2]
        crop = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        net.setInput(crop)
        detect = net.forward()
        face_found = False

        for i in range(0, detect.shape[2]):
            confidence = detect[0, 0, i, 2]
            if confidence > 0.5: ## Change this to make it more/less effective
                face_found = True
                box = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                face = image[start_y:end_y, start_x:end_x]
                
                if face is not None and face.size > 0:
                    cv2.imwrite(file_path, face)
                break

        if not face_found:
            print(f"No face found in {file_path}") #Delete file for training images.

print("Done.")
