#!/usr/bin/env python3
"""
Model conversion script to fix compatibility issues.
This script converts the .keras model to SavedModel format for better compatibility.
"""

import tensorflow as tf
import os

def convert_keras_to_savedmodel():
    """Convert .keras model to SavedModel format for better compatibility."""
    
    # Paths
    keras_path = "models/model.keras"
    savedmodel_path = "models/model_savedmodel"
    
    if not os.path.exists(keras_path):
        print(f"❌ Model file not found: {keras_path}")
        return False
    
    try:
        print("🔄 Loading .keras model...")
        # Load the model without compilation to avoid version issues
        model = tf.keras.models.load_model(keras_path, compile=False)
        print(f"✅ Successfully loaded model: {model.input_shape} -> {model.output_shape}")
        
        # Save as SavedModel format (more compatible)
        print("💾 Converting to SavedModel format...")
        model.save(savedmodel_path, save_format='tf')
        print(f"✅ Model saved to: {savedmodel_path}")
        
        # Test loading the SavedModel
        print("🧪 Testing SavedModel loading...")
        test_model = tf.keras.models.load_model(savedmodel_path)
        print("✅ SavedModel loads successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting model conversion...")
    success = convert_keras_to_savedmodel()
    
    if success:
        print("\n✅ Model conversion completed successfully!")
        print("You can now use the SavedModel format for better compatibility.")
    else:
        print("\n❌ Model conversion failed.")
        print("Please check the error messages above.")
