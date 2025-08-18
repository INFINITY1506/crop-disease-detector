#!/usr/bin/env python3
"""
Model compatibility fix script.
This script helps convert your model to a format compatible with current TensorFlow versions.
"""

import tensorflow as tf
import numpy as np
import os
import json

def fix_model_compatibility():
    """Fix model compatibility by loading and re-saving with current TensorFlow version."""
    
    print("🔧 Model Compatibility Fix Script")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists("models/model.keras"):
        print("❌ Model file not found: models/model.keras")
        return False
    
    try:
        print("🔄 Attempting to load the original model...")
        
        # Try different loading methods
        model = None
        methods = [
            ("tf.keras.models.load_model", lambda: tf.keras.models.load_model("models/model.keras", compile=False)),
            ("tf.keras with safe_mode=False", lambda: tf.keras.models.load_model("models/model.keras", compile=False, safe_mode=False)),
        ]
        
        for method_name, method_func in methods:
            try:
                print(f"  Trying {method_name}...")
                model = method_func()
                print(f"  ✅ Success with {method_name}")
                break
            except Exception as e:
                print(f"  ❌ Failed with {method_name}: {str(e)[:100]}...")
                continue
        
        if model is None:
            print("❌ Could not load the original model with any method.")
            print("💡 The model file appears to be incompatible with current TensorFlow version.")
            print("   Consider retraining the model or using the fallback model in the app.")
            return False
        
        # Model loaded successfully - now re-save it
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total parameters: {model.count_params():,}")
        
        # Save in multiple formats for maximum compatibility
        print("💾 Saving model in compatible formats...")
        
        # 1. SavedModel format (most compatible)
        savedmodel_path = "models/model_fixed_savedmodel"
        model.save(savedmodel_path, save_format='tf')
        print(f"   ✅ Saved as SavedModel: {savedmodel_path}")
        
        # 2. New .keras format with current version
        keras_path = "models/model_fixed.keras"
        model.save(keras_path, save_format='keras')
        print(f"   ✅ Saved as .keras: {keras_path}")
        
        # Test loading the fixed models
        print("🧪 Testing fixed model loading...")
        test_savedmodel = tf.keras.models.load_model(savedmodel_path)
        test_keras = tf.keras.models.load_model(keras_path)
        print("   ✅ Both fixed models load successfully!")
        
        # Create a test prediction to ensure everything works
        print("🧪 Testing model prediction...")
        test_input = np.random.random((1, 192, 192, 3)).astype(np.float32)
        test_pred = model.predict(test_input, verbose=0)
        print(f"   ✅ Model prediction works! Output shape: {test_pred.shape}")
        
        print("\n🎉 Model compatibility fix completed successfully!")
        print("📋 What was created:")
        print(f"   • {savedmodel_path} - SavedModel format (recommended)")
        print(f"   • {keras_path} - Fixed .keras format")
        print("\n💡 Next steps:")
        print("   1. Upload the fixed model files to your repository")
        print("   2. Update your app to use the fixed model")
        print("   3. Redeploy your Streamlit app")
        
        return True
        
    except Exception as e:
        print(f"❌ Model fix failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_model_compatibility()
    if not success:
        print("\n🔧 Alternative solutions:")
        print("   1. Use the fallback model (already implemented in the app)")
        print("   2. Retrain your model with the current TensorFlow version")
        print("   3. Convert your model using TensorFlow Lite or ONNX")
