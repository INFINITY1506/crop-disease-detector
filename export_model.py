#!/usr/bin/env python3
"""
Run this script in your ORIGINAL training environment
to export your model in a compatible format
"""

import tensorflow as tf
import os

def export_trained_model():
    """Export your trained model in multiple compatible formats"""
    print("🔄 Exporting trained model...")
    
    # Load your original model (this should work in your training environment)
    model = tf.keras.models.load_model('models/model.keras')
    print("✅ Original model loaded successfully")
    
    # Export in multiple formats for maximum compatibility
    
    # 1. SavedModel format (most compatible)
    print("💾 Saving as SavedModel...")
    model.save('models/model_savedmodel', save_format='tf')
    
    # 2. HDF5 format
    print("💾 Saving as HDF5...")
    model.save('models/model.h5', save_format='h5')
    
    # 3. Weights only (as backup)
    print("💾 Saving weights...")
    model.save_weights('models/model_weights.h5')
    
    # 4. Model architecture
    print("💾 Saving architecture...")
    with open('models/model_architecture.json', 'w') as f:
        f.write(model.to_json())
    
    print("\n🎉 Export complete!")
    print("📁 Files created:")
    print("   - models/model_savedmodel/ (use this one)")
    print("   - models/model.h5")
    print("   - models/model_weights.h5")
    print("   - models/model_architecture.json")
    
    print("\n📤 Upload these files to replace your current model files")

if __name__ == "__main__":
    export_trained_model()
