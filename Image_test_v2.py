"""


@author: Pandu Dewanatha
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB7

# Prepare test image function
def prepare_image(file):
    img = image.load_img(os.path.join(test_images, file), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_2 = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_2)

# Load Model
model_path = 'C:/Users/pdewanat/Downloads/b0_model.h5'
model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = GlobalAveragePooling2D()(model.output)
output = Dense(100, activation='softmax')(x)  # Assuming num_classes=100
model = Model(inputs=model.input, outputs=output)

# Load weights
model.load_weights(model_path)

# Load category mapping
category_df = pd.read_csv('C:/Users/pdewanat/Desktop/ECE 50024/Kaggle/purdue-face-recognition-challenge-2024/category.csv')
dictionary = dict(zip(category_df['Category Number'], category_df['Category Name']))

# Load test images
test_images = 'C:/Users/pdewanat/Desktop/ECE 50024/Kaggle/test/Test_small'
test_image_filenames = sorted(os.listdir(test_images), key=lambda x: int(os.path.splitext(x)[0]))


# Predict and map predictions to category names
predicted_labels = []
ids = []

for i, filename in enumerate(test_image_filenames):
    print(f"Processing image {filename}")
    prepared_img = prepare_image(filename)
    prediction = model.predict(prepared_img, verbose=0)  # Set verbose to 0 to suppress progress bars
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_labels.append(dictionary.get(predicted_class))
    ids.append(filename)

# Remove jpg extension for excel file
ids = [os.path.splitext(filename)[0] for filename in test_image_filenames]

# Excel setup
submission_df = pd.DataFrame({
    'Id': ids,
    'Category': predicted_labels
})

# Save excel
submission_csv_path = 'C:/Users/pdewanat/Desktop/ECE 50024/Kaggle/submission_test.csv'
submission_df.to_csv(submission_csv_path, index=False)
