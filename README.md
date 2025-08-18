# 🌿 Crop Disease Detection App

A machine learning web application that detects diseases in crop plants using computer vision. The app can identify 15 different conditions across tomatoes, potatoes, and bell peppers, providing treatment and prevention recommendations.

## 🚀 Features

- **Multi-crop Disease Detection**: Supports tomatoes, potatoes, and bell peppers
- **15 Disease Classes**: Detects various diseases including bacterial spot, early blight, late blight, and more
- **Real-time Predictions**: Upload images or use live camera capture
- **Treatment Recommendations**: Provides detailed treatment and prevention advice
- **Confidence Scoring**: Shows prediction confidence and top-3 alternatives
- **Mobile-Friendly**: Responsive design that works on all devices

## 🏗️ Architecture

- **Model**: MobileNetV2-based CNN trained on PlantVillage dataset
- **Backend**: Python with TensorFlow/Keras
- **Frontend**: Streamlit web framework
- **Deployment**: Streamlit Cloud

## 📦 Installation & Local Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd crop-disease-app-1755541742
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

1. **Initialize Git repository** (if not already done)
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Crop disease detection app"
   ```

2. **Create a GitHub repository**
   - Go to [GitHub.com](https://github.com) and create a new repository
   - Name it something like `crop-disease-detector`
   - Don't initialize with README (since you already have one)

3. **Push your code**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your GitHub repository
   - Set the main file path to: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Streamlit Cloud will install dependencies and deploy your app
   - This usually takes 2-5 minutes
   - You'll get a public URL like `https://your-app-name.streamlit.app`

### Step 3: Configure (Optional)

If you encounter any issues, you can:

- **Check logs**: View deployment logs in Streamlit Cloud dashboard
- **Reboot app**: Use the reboot option if the app gets stuck
- **Update secrets**: Add any environment variables if needed

## 📁 Project Structure

```
crop-disease-app-1755541742/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
└── models/
    ├── model.keras                 # Trained ML model
    ├── class_indices.json          # Class label mappings
    └── disease_info.csv            # Disease information database
```

## 🔧 Model Details

- **Architecture**: MobileNetV2 (pre-trained, fine-tuned)
- **Input Size**: 192x192 pixels
- **Classes**: 15 different plant conditions
- **Dataset**: PlantVillage dataset
- **Preprocessing**: MobileNetV2 standard preprocessing

### Supported Classes

**Tomatoes (9 classes):**
- Bacterial Spot, Early Blight, Late Blight, Leaf Mold
- Septoria Leaf Spot, Spider Mites, Target Spot
- Yellow Leaf Curl Virus, Mosaic Virus, Healthy

**Potatoes (3 classes):**
- Early Blight, Late Blight, Healthy

**Bell Peppers (2 classes):**
- Bacterial Spot, Healthy

**Remaining (1 class):**
- Healthy

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure all files in `models/` directory are present
   - Check if TensorFlow version is compatible

2. **Memory Issues**
   - The model file is large (~85MB)
   - Streamlit Cloud has memory limits; the app should work fine

3. **Slow Predictions**
   - First prediction may be slower due to model loading
   - Subsequent predictions should be faster due to caching

4. **File Upload Issues**
   - Supported formats: JPG, JPEG, PNG, WEBP
   - Maximum file size: 200MB (configurable in config.toml)

### Getting Help

If you encounter issues:

1. Check the [Streamlit Community Forum](https://discuss.streamlit.io)
2. Review [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-community-cloud)
3. Check GitHub Issues in your repository

## 📝 Usage Instructions

1. **Upload Method**: Click "Upload Image" tab and select a plant leaf image
2. **Camera Method**: Click "Live Camera" tab and take a photo directly
3. **View Results**: The app will show:
   - Primary prediction with confidence score
   - Disease information and treatment advice
   - Top-3 alternative predictions

## ⚠️ Important Notes

- **Medical Disclaimer**: This app provides probabilistic predictions for educational purposes
- **Professional Advice**: Always confirm with local agricultural extension services for critical decisions
- **Image Quality**: Use clear, well-lit images of individual leaves for best results
- **Internet Required**: The app needs internet connection for deployment and some features

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding more plant diseases

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Farming! 🌱**