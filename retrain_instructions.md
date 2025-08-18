# How to Retrain Your Model for Deployment

## Why Retrain?
Your current model was trained with an incompatible TensorFlow version. Retraining with the current environment ensures compatibility.

## Step-by-Step Instructions:

### 1. Install Compatible Environment
```bash
pip install tensorflow==2.16.0 keras pillow numpy pandas scikit-learn matplotlib
```

### 2. Training Script Template
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Create model architecture (example - adjust to your needs)
def create_model(num_classes=15):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False  # Freeze base model
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Compile model
model = create_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load your training data here
# X_train, y_train = load_your_data()

# Train the model
# history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save in compatible format
model.save('models/model_savedmodel', save_format='tf')  # SavedModel format
model.save('models/model_v2.keras')  # New .keras format

print("✅ Model saved in compatible formats!")
```

### 3. Class Indices
Make sure your class indices match the existing `models/class_indices.json`:
- 0: Pepper__bell___Bacterial_spot
- 1: Pepper__bell___healthy
- 2: Potato___Early_blight
- ... (see existing file)

### 4. Test Locally
```python
# Test loading
model = tf.keras.models.load_model('models/model_savedmodel')
print("✅ Model loads successfully!")
```
