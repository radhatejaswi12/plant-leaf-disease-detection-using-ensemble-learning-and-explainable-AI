from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from matplotlib.pyplot import imread, imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np



class_names =[
    "Apple___Apple_scab",
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
    "Tomato___healthy"
]


label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

num_classes = len(label_encoder.classes_)

# Load EfficientNetB0 without the top classification layers
efficientnet_model = EfficientNetB0(input_shape=(100,100, 3), include_top=False, weights='imagenet')

# Define the new model
inputs = efficientnet_model.input

# Get the output of the last convolutional layer for Grad-CAM
conv_output = efficientnet_model.layers[-1].output

# Add Global Average Pooling
x = GlobalAveragePooling2D()(conv_output)

# Add dense layers with regularization, batch normalization, and dropout
x = Dense(128, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization(renorm=True)(x)
x = Dropout(0.3)(x)

x = Dense(64, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization(renorm=True)(x)
x = Dropout(0.3)(x)

x = Dense(32, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization(renorm=True)(x)
x = Dropout(0.3)(x)

# Add the final classification layer
outputs = Dense(units=num_classes, activation='softmax')(x)

# Combine into the model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
custom_optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the saved weights
model.load_weights('best_model.h5')

print("Model loaded successfully!")



# Preprocess the image
def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Resize the image to target size
    image = cv2.resize(image, (100,100))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Create an RGB edge map (edges in red)
    edges_colored = np.zeros_like(image)
    edges_colored[:, :, 2] = edges
    
    # Overlay the edges onto the original image
    processed_image = cv2.addWeighted(image, 0.8, edges_colored, 0.5, 0)
    image2 = cv2.resize(processed_image, (256,256))

    cv2.imwrite("static/output_image.png",processed_image)
    
    # Normalize the image (scaling pixel values between 0 and 1)
    processed_image = processed_image / 255.0
    
    return np.expand_dims(processed_image, axis=0)





def pred_leaf_disease(img_path):
# Path to the image
            image_path =img_path
            preprocessed_image = preprocess_single_image(image_path)

            # Make prediction
            predictions = model.predict(preprocessed_image)

            # Decode the predicted label
            predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
            confidence = np.max(predictions)

            print(f"Predicted Label: {predicted_label[0]}, Confidence: {confidence * 100:.2f}%")

            return predicted_label[0]





import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def generate_gradcam_heatmap(model, image, class_idx, last_conv_layer_name):
    # Build a model that maps the input image to the activations of the last conv layer and the predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, original_image, alpha=0.6, colormap=cv2.COLORMAP_JET):
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    # Overlay the heatmap on the original image
    overlayed_image = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_image

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def save_explainable_output(img):
            # Ask the user to provide the image path
            image_path =img

            # Create 'static' directory if it doesn't exist
            #os.makedirs("static", exist_ok=True)

            # Load and preprocess the image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(original_image, (100, 100)) / 255.0  # Resize and normalize

            # Get the predicted class index
            class_idx = np.argmax(model.predict(np.expand_dims(image, axis=0)))

            # Generate the Grad-CAM heatmap
            heatmap = generate_gradcam_heatmap(model, image, class_idx, last_conv_layer_name='top_conv')

            # Overlay the heatmap on the original image
            overlayed_image = overlay_heatmap_on_image(heatmap, original_image)

            # Convert RGB back to BGR for saving with cv2
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            overlayed_bgr = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

            # Normalize and convert heatmap to 3-channel image
            heatmap_3ch = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # Save images to 'static' folder
            cv2.imwrite("static/original_image.jpg", original_bgr)
            cv2.imwrite("static/heatmap.jpg", heatmap_3ch)
            cv2.imwrite("static/heatmap_overlay.jpg", overlayed_bgr)

