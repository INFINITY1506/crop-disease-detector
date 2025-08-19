# ============================================
# RUN THIS CODE IN YOUR GOOGLE COLAB NOTEBOOK
# ============================================

import tensorflow as tf
import os
from google.colab import files

def export_model_for_deployment():
    """Export your trained model in multiple compatible formats"""
    print("🔄 Exporting model for deployment...")
    
    # Load your trained model (adjust path as needed)
    # Replace 'your_model_path' with your actual model variable or path
    model = your_trained_model  # or tf.keras.models.load_model('path_to_your_model')
    
    # Create export directory
    os.makedirs('deployment_models', exist_ok=True)
    
    # 1. SavedModel format (MOST COMPATIBLE for Streamlit)
    print("💾 Saving as SavedModel...")
    model.save('deployment_models/model_savedmodel', save_format='tf')
    
    # 2. New .keras format with current TensorFlow
    print("💾 Saving as .keras (current version)...")
    model.save('deployment_models/model_new.keras')
    
    # 3. HDF5 format (backup)
    print("💾 Saving as HDF5...")
    model.save('deployment_models/model.h5')
    
    # 4. Save model summary and info
    with open('deployment_models/model_info.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"\nTensorFlow version: {tf.__version__}\n")
        f.write(f"Keras version: {tf.keras.__version__}\n")
    
    print("✅ Export complete!")
    return True

# Run the export
export_model_for_deployment()

# Create a ZIP file with all exported models
import shutil
shutil.make_archive('deployment_models', 'zip', 'deployment_models')

# Download the ZIP file
print("📥 Downloading models...")
files.download('deployment_models.zip')

print("🎉 SUCCESS! Download the ZIP file and extract it to your project.")
