# plant-leaf-disease-detection-using-ensemble-learning-and-explainable-AI

This project leverages deep learning models to classify plant leaf diseases using selected classes from the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease). It aims to assist in early and accurate detection of plant diseases to support agriculture and farming practices.

## 📁 Dataset

- **Source**: PlantVillage (via Kaggle)
- **Classes Selected**: A subset of classes ("Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy" )
- **Samples**: 1000 images per class
- **Preprocessing**: Resized images, normalized pixel values, split into train-test sets

##  Models Used

Implemented multiple models to compare performance:
- **Transfer Learning**: EfficientNetB0, ResNet50
- **Ensemble Learning**: Combined outputs of multiple models

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Grad-CAM for Explainable AI


## Tools & Libraries

- Python, TensorFlow / Keras
- OpenCV, NumPy, Matplotlib, scikit-learn
- HTML, CSS and Flask for deployment

## 🚀 Deployment

The final model is deployed for real-time prediction. Upload a leaf image and get the predicted disease class.

To run locally:
```bash
python app.py
