# Google Colab Notebook Setup - Cell by Cell

## Cell 1: Install Libraries
```python
!pip install tensorflow==2.16.0 keras pillow matplotlib seaborn scikit-learn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
import zipfile
from google.colab import files, drive
from PIL import Image
import cv2

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

tf.random.set_seed(42)
np.random.seed(42)
```

## Cell 2: Define Classes
```python
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy", 
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight", 
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

NUM_CLASSES = len(CLASS_NAMES)
print(f"📊 Number of classes: {NUM_CLASSES}")

class_indices = {str(i): class_name for i, class_name in enumerate(CLASS_NAMES)}
```

## Cell 3: Upload Dataset
```python
print("📤 Upload your dataset (ZIP file)")
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('dataset')
        print(f"✅ Extracted {filename}")
```

## Cell 4: Create Dataset
```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def create_dataset_from_directory(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation", 
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES
    )
    
    return train_ds, val_ds

data_directory = "dataset"
train_dataset, val_dataset = create_dataset_from_directory(data_directory)
print("✅ Dataset loaded successfully!")
```

## Cell 5: Data Augmentation
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1)
])

AUTOTUNE = tf.data.AUTOTUNE

def prepare_dataset(ds, augment=False):
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = prepare_dataset(train_dataset, augment=True)
val_ds = prepare_dataset(val_dataset, augment=False)
print("✅ Data preprocessing complete!")
```

## Cell 6: Create Model
```python
def create_model(num_classes=NUM_CLASSES):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model

model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

model.summary()
```

## Cell 7: Training Setup
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]
```

## Cell 8: Train Model
```python
print("🚀 Starting training...")
EPOCHS_INITIAL = 20

history_initial = model.fit(
    train_ds,
    epochs=EPOCHS_INITIAL,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("✅ Training completed!")
```

## Cell 9: Fine-tuning
```python
print("🔧 Fine-tuning model...")

base_model = model.layers[1]
base_model.trainable = True

fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

EPOCHS_FINE_TUNE = 15

history_fine_tune = model.fit(
    train_ds,
    epochs=EPOCHS_INITIAL + EPOCHS_FINE_TUNE,
    initial_epoch=len(history_initial.history['loss']),
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("✅ Fine-tuning completed!")
```

## Cell 10: Evaluate Model
```python
val_loss, val_accuracy, val_top3_accuracy = model.evaluate(val_ds, verbose=0)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Top-3 Accuracy: {val_top3_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
```

## Cell 11: Export for Streamlit
```python
print("📦 Exporting model for Streamlit...")

os.makedirs('streamlit_models', exist_ok=True)

# Save model in multiple formats
model.save('streamlit_models/model_savedmodel', save_format='tf')
model.save('streamlit_models/model_new.keras')
model.save('streamlit_models/model.h5')

# Save class indices
with open('streamlit_models/class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)

# Create disease info CSV
disease_info_data = []
for class_name in CLASS_NAMES:
    if 'healthy' in class_name.lower():
        disease_info_data.append({
            'label': class_name,
            'title': 'Healthy Plant',
            'description': 'No visible disease symptoms. Leaves are green and healthy.',
            'treatment': 'Maintain routine nutrition and irrigation.',
            'prevention': 'Use certified seeds, rotate crops, practice good sanitation.',
            'reference': 'Standard agricultural practices'
        })
    else:
        parts = class_name.split('_')
        crop = parts[0]
        disease = '_'.join(parts[1:]) if len(parts) > 1 else 'Disease'
        
        disease_info_data.append({
            'label': class_name,
            'title': f'{crop} {disease}',
            'description': f'Disease affecting {crop} plants requiring attention.',
            'treatment': 'Consult agricultural extension services.',
            'prevention': 'Use disease-resistant varieties, proper spacing.',
            'reference': 'Agricultural disease management guidelines'
        })

disease_info_df = pd.DataFrame(disease_info_data)
disease_info_df.to_csv('streamlit_models/disease_info.csv', index=False)

print("✅ Export complete!")
```

## Cell 12: Download Files
```python
import shutil
shutil.make_archive('crop_disease_model_complete', 'zip', 'streamlit_models')

print("🎉 Model ready for download!")
print(f"Validation Accuracy: {val_accuracy:.1%}")

files.download('crop_disease_model_complete.zip')

print("🚀 INSTRUCTIONS:")
print("1. Download the ZIP file")
print("2. Extract to your Streamlit app's models/ folder")
print("3. Push to GitHub")
print("4. Your app will use the trained model!")
```
