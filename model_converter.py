#!/usr/bin/env python3
"""
Advanced Model Converter for Crop Disease Detection
Converts your trained model to multiple formats for maximum compatibility
"""

import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

def convert_keras_to_savedmodel():
    """Convert .keras model to SavedModel format"""
    print("🔄 Converting .keras model to SavedModel format...")
    
    try:
        # Try different loading methods
        model_path = "models/model.keras"
        
        print(f"📁 Loading model from: {model_path}")
        
        # Method 1: Standard loading
        try:
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully with standard method")
        except Exception as e:
            print(f"❌ Standard method failed: {str(e)[:100]}...")
            
            # Method 2: With compile=False
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model loaded successfully with compile=False")
            except Exception as e:
                print(f"❌ Compile=False method failed: {str(e)[:100]}...")
                
                # Method 3: Custom objects
                try:
                    custom_objects = {
                        'Functional': tf.keras.Model,
                        'functional': tf.keras.Model
                    }
                    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print("✅ Model loaded successfully with custom objects")
                except Exception as e:
                    print(f"❌ Custom objects method failed: {str(e)[:100]}...")
                    return False
        
        # Save as SavedModel format
        output_path = "models/model_savedmodel"
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
        
        model.save(output_path, save_format='tf')
        print(f"✅ Model saved as SavedModel: {output_path}")
        
        # Test the converted model
        print("🧪 Testing converted model...")
        test_model = tf.keras.models.load_model(output_path)
        
        # Create dummy input to test
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        predictions = test_model.predict(dummy_input, verbose=0)
        print(f"✅ Test successful! Output shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def create_tflite_model():
    """Convert to TensorFlow Lite for even better compatibility"""
    print("\n🔄 Converting to TensorFlow Lite...")
    
    try:
        savedmodel_path = "models/model_savedmodel"
        if not os.path.exists(savedmodel_path):
            print("❌ SavedModel not found. Run conversion first.")
            return False
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = "models/model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved: {tflite_path}")
        return True
        
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        return False

def main():
    print("🚀 Advanced Model Converter")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists("models/model.keras"):
        print("❌ Model file not found: models/model.keras")
        return
    
    # Convert to SavedModel
    success = convert_keras_to_savedmodel()
    
    if success:
        print("\n🎉 Conversion successful!")
        print("📁 Your model is now available in multiple formats:")
        print("   - models/model_savedmodel/ (SavedModel format)")
        
        # Try TFLite conversion
        create_tflite_model()
        
        print("\n🔧 Next steps:")
        print("1. Commit and push these new model files")
        print("2. Your app will automatically use the compatible format")
        print("3. Test with your images - should work much better!")
        
    else:
        print("\n❌ Model conversion failed.")
        print("💡 Your original model might be from an incompatible TensorFlow version.")
        print("🔧 Solutions:")
        print("   1. Retrain your model with current TensorFlow (recommended)")
        print("   2. Use the demo mode for now")
        print("   3. Export model from your training environment in SavedModel format")

if __name__ == "__main__":
    main()
