from google.colab import files
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load CSV
df = pd.read_csv('/content/train.csv', index_col=0)

# Load the category
category_df = pd.read_csv('/content/category.csv')

# Create dictionary
dictionary = dict(zip(category_df['Category Name'], category_df['Category Number']))

# Split training and validation
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# Mapping
train_df['Category'] = train_df['Category'].map(dictionary)
validate_df['Category'] = validate_df['Category'].map(dictionary)
train_df['Category'] = train_df['Category'].astype(str)
validate_df['Category'] = validate_df['Category'].astype(str)

# Load images
images_dir = '/content/train_small'

# Data augmentation
train_data = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=[0.85, 1.15],
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

validation_data = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load images
train_generator = train_data.flow_from_dataframe(
    dataframe=train_df,
    directory=images_dir,
    x_col="File Name",
    y_col="Category",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_data.flow_from_dataframe(
    dataframe=validate_df,
    directory=images_dir,
    x_col="File Name",
    y_col="Category",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load EfficientNetB# (B0-B7) 
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(dictionary), activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Model setup
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate adjustment and early stops (if necessary)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Balance class weights
weights = compute_class_weight('balanced', classes=np.unique(train_df['Category'].astype(int)), y=train_df['Category'].astype(int))
class_weights = dict(enumerate(weights))

# Training
training = model.fit(
    train_generator,
    epochs=20,  # Adjust based on when you observe the plateau
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping],
    class_weight=class_weights
)

# Save model
model.save('/content/final_model.h5')
files.download('/content/final_model.h5')
