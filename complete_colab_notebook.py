# ============================================
# COMPLETE GOOGLE COLAB NOTEBOOK CODE
# Crop Disease Detection Model Training
# ============================================

# Cell 1: Install and Import Required Libraries
# ============================================
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

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Cell 2: Mount Google Drive (Optional - if your data is in Drive)
# ============================================
# Uncomment if you have data in Google Drive
# drive.mount('/content/drive')

# Cell 3: Download and Prepare Dataset
# ============================================
# Option A: If you have your own dataset, upload it
print("📤 Upload your dataset (ZIP file with plant images organized in folders)")
uploaded = files.upload()

# Extract the uploaded dataset
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('dataset')
        print(f"✅ Extracted {filename}")

# Option B: Use a sample dataset (PlantVillage-like structure)
# If you don't have data, we'll create the structure for you to upload images

# Cell 4: Define Dataset Classes (Match your Streamlit app)
# ============================================
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
print("📋 Classes:", CLASS_NAMES)

# Create class indices mapping (for Streamlit app)
class_indices = {str(i): class_name for i, class_name in enumerate(CLASS_NAMES)}

# Cell 5: Data Loading and Preprocessing
# ============================================
IMG_SIZE = (224, 224)  # Standard size for transfer learning
BATCH_SIZE = 32

def create_dataset_from_directory(data_dir):
    """Create TensorFlow dataset from directory structure"""
    
    # Create training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES
    )
    
    # Create validation dataset  
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

# Load your dataset (adjust path as needed)
data_directory = "dataset"  # Change this to your dataset path
train_dataset, val_dataset = create_dataset_from_directory(data_directory)

print("✅ Dataset loaded successfully!")

# Cell 6: Data Augmentation and Optimization
# ============================================
# Data augmentation for better generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1)
])

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE

def prepare_dataset(ds, augment=False):
    # Normalize pixel values to [0,1]
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = prepare_dataset(train_dataset, augment=True)
val_ds = prepare_dataset(val_dataset, augment=False)

print("✅ Data preprocessing complete!")

# Cell 7: Visualize Sample Data
# ============================================
plt.figure(figsize=(15, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {CLASS_NAMES[labels[i]]}")
        plt.axis('off')
plt.suptitle("Sample Training Images")
plt.tight_layout()
plt.show()

# Cell 8: Create the Model Architecture
# ============================================
def create_model(num_classes=NUM_CLASSES):
    """Create transfer learning model using MobileNetV2"""
    
    # Base model (pre-trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    model = tf.keras.Sequential([
        # Input layer
        layers.Input(shape=IMG_SIZE + (3,)),
        
        # Base model
        base_model,
        
        # Custom head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model

# Create the model
model = create_model()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

# Model summary
model.summary()

# Cell 9: Training Configuration and Callbacks
# ============================================
# Define callbacks for better training
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

# Cell 10: Initial Training (Transfer Learning)
# ============================================
print("🚀 Starting initial training (frozen base model)...")

EPOCHS_INITIAL = 20

history_initial = model.fit(
    train_ds,
    epochs=EPOCHS_INITIAL,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

print("✅ Initial training completed!")

# Cell 11: Fine-tuning (Unfreeze some layers)
# ============================================
print("🔧 Fine-tuning model (unfreezing top layers)...")

# Unfreeze the top layers of the base model for fine-tuning
base_model = model.layers[1]  # Get the base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) - 30

# Freeze all the layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

# Continue training
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

# Cell 12: Training Visualization
# ============================================
def plot_training_history(history_initial, history_fine_tune=None):
    """Plot training and validation metrics"""
    
    # Combine histories if fine-tuning was done
    if history_fine_tune:
        acc = history_initial.history['accuracy'] + history_fine_tune.history['accuracy']
        val_acc = history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
        loss = history_initial.history['loss'] + history_fine_tune.history['loss']
        val_loss = history_initial.history['val_loss'] + history_fine_tune.history['val_loss']
    else:
        acc = history_initial.history['accuracy']
        val_acc = history_initial.history['val_accuracy']
        loss = history_initial.history['loss']
        val_loss = history_initial.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    if history_fine_tune:
        plt.axvline(x=len(history_initial.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    if history_fine_tune:
        plt.axvline(x=len(history_initial.history['loss'])-1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history_initial, history_fine_tune)

# Cell 13: Model Evaluation
# ============================================
print("📊 Evaluating model performance...")

# Evaluate on validation set
val_loss, val_accuracy, val_top3_accuracy = model.evaluate(val_ds, verbose=0)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Top-3 Accuracy: {val_top3_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# Generate predictions for confusion matrix
y_pred = []
y_true = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Classification Report
print("📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Cell 14: Test Model with Sample Images
# ============================================
def predict_image(model, image_path, class_names):
    """Predict single image"""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return predicted_class, confidence, predictions[0]

# Test with a few validation images
plt.figure(figsize=(15, 10))
test_count = 0

for images, labels in val_ds.take(1):
    for i in range(min(6, len(images))):
        if test_count >= 6:
            break
            
        # Save test image temporarily
        test_img = images[i].numpy()
        test_img_path = f'temp_test_{i}.jpg'
        tf.keras.utils.save_img(test_img_path, test_img)
        
        # Make prediction
        pred_class, confidence, _ = predict_image(model, test_img_path, CLASS_NAMES)
        true_class = CLASS_NAMES[labels[i]]
        
        # Plot
        plt.subplot(2, 3, test_count + 1)
        plt.imshow(test_img)
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}', 
                 color=color, fontsize=10)
        plt.axis('off')
        
        # Clean up
        os.remove(test_img_path)
        test_count += 1

plt.suptitle('Model Predictions on Validation Images')
plt.tight_layout()
plt.show()

# Cell 15: Export Model for Streamlit Deployment
# ============================================
print("📦 Exporting model for Streamlit deployment...")

# Create export directory
os.makedirs('streamlit_models', exist_ok=True)

# 1. Save as SavedModel format (BEST for Streamlit)
model.save('streamlit_models/model_savedmodel', save_format='tf')
print("✅ Saved as SavedModel format")

# 2. Save as new .keras format
model.save('streamlit_models/model_new.keras')
print("✅ Saved as .keras format")

# 3. Save as HDF5 format (backup)
model.save('streamlit_models/model.h5')
print("✅ Saved as HDF5 format")

# 4. Save class indices (required by Streamlit app)
with open('streamlit_models/class_indices.json', 'w') as f:
    json.dump(class_indices, f, indent=2)
print("✅ Saved class indices")

# 5. Create disease info CSV (basic template - you can customize)
disease_info_data = []
for class_name in CLASS_NAMES:
    if 'healthy' in class_name.lower():
        disease_info_data.append({
            'label': class_name,
            'title': 'Healthy Plant',
            'description': 'No visible disease symptoms. Leaves are green and healthy with normal growth pattern.',
            'treatment': 'Maintain routine nutrition and irrigation; monitor regularly for disease symptoms.',
            'prevention': 'Use certified seed potatoes, rotate crops, practice good sanitation; ensure proper soil drainage.',
            'reference': 'Standard agricultural practices'
        })
    else:
        # Extract disease name for generic info
        parts = class_name.split('_')
        crop = parts[0]
        disease = '_'.join(parts[1:]) if len(parts) > 1 else 'Disease'
        
        disease_info_data.append({
            'label': class_name,
            'title': f'{crop} {disease}',
            'description': f'Disease affecting {crop} plants with characteristic symptoms requiring attention.',
            'treatment': 'Consult agricultural extension services for specific treatment recommendations.',
            'prevention': 'Use disease-resistant varieties, proper spacing, avoid overhead irrigation.',
            'reference': 'Agricultural disease management guidelines'
        })

disease_info_df = pd.DataFrame(disease_info_data)
disease_info_df.to_csv('streamlit_models/disease_info.csv', index=False)
print("✅ Created disease info CSV")

# 6. Save model information
model_info = {
    'tensorflow_version': tf.__version__,
    'keras_version': tf.keras.__version__,
    'model_architecture': 'MobileNetV2 + Custom Head',
    'input_shape': list(IMG_SIZE + (3,)),
    'num_classes': NUM_CLASSES,
    'final_val_accuracy': float(val_accuracy),
    'final_val_loss': float(val_loss),
    'class_names': CLASS_NAMES
}

with open('streamlit_models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("✅ Saved model information")

# Cell 16: Create Download Package
# ============================================
print("📦 Creating download package...")

# Create ZIP file with all necessary files
import shutil
shutil.make_archive('crop_disease_model_complete', 'zip', 'streamlit_models')

print("🎉 Model export complete!")
print("\n📁 Files created in 'streamlit_models/' directory:")
print("   ├── model_savedmodel/     (Primary model - use this)")
print("   ├── model_new.keras       (Backup model)")
print("   ├── model.h5              (Alternative format)")
print("   ├── class_indices.json    (Class mappings)")
print("   ├── disease_info.csv      (Disease information)")
print("   └── model_info.json       (Model metadata)")

print(f"\n🎯 Model Performance Summary:")
print(f"   • Validation Accuracy: {val_accuracy:.1%}")
print(f"   • Validation Loss: {val_loss:.4f}")
print(f"   • Number of Classes: {NUM_CLASSES}")

# Download the complete package
files.download('crop_disease_model_complete.zip')

print("\n🚀 DEPLOYMENT INSTRUCTIONS:")
print("1. Download 'crop_disease_model_complete.zip'")
print("2. Extract the ZIP file")
print("3. Copy all files to your Streamlit app's 'models/' folder")
print("4. Commit and push to GitHub")
print("5. Your Streamlit app will automatically use the trained model!")

# Cell 17: Final Model Test Function
# ============================================
def test_exported_model():
    """Test the exported model to ensure it works"""
    print("🧪 Testing exported model...")
    
    try:
        # Load the SavedModel
        loaded_model = tf.keras.models.load_model('streamlit_models/model_savedmodel')
        
        # Test with dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        predictions = loaded_model.predict(dummy_input, verbose=0)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Predicted class: {CLASS_NAMES[np.argmax(predictions[0])]}")
        print(f"   Confidence: {np.max(predictions[0]):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

# Run the test
test_success = test_exported_model()

if test_success:
    print("\n🎉 SUCCESS! Your model is ready for deployment!")
    print("📱 Your Streamlit app will now give accurate predictions!")
else:
    print("\n⚠️  Model test failed. Please check the export process.")

print("\n" + "="*60)
print("🌿 CROP DISEASE DETECTION MODEL TRAINING COMPLETE! 🌿")
print("="*60)
