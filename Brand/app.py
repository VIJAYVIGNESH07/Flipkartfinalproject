import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from datetime import datetime
import streamlit as st

# Set Streamlit page configuration (must be the first command)
st.set_page_config(page_title="Product Detection App", layout="wide")
model_path = os.path.join(os.getcwd(), 'my_model.keras')
json_path = os.path.join(os.getcwd(), 'class_indicies.json')

# Load model
model_path = 'my_model.keras'
class_indices_path = 'class_indices.json'
excel_file = 'product_detection_results.xlsx'

# Load model
@st.cache_resource
def load_prediction_model(model_path):
    try:
        model = load_model(model_path)
        # st.success(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load class indices
@st.cache_resource
def load_class_indices(class_indices_path):
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
            label_map = {v: k for k, v in class_indices.items()}
            # st.success("Class indices loaded successfully.")
            return label_map
    except Exception as e:
        st.error(f"Error loading class indices: {e}")
        return None

model = load_prediction_model(model_path)
label_map = load_class_indices(class_indices_path)

# Preprocess image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image {img_path}: {e}")
        return None

# Predict a single image
def predict_image(img_array, model, label_map):
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = label_map[predicted_class]
        confidence = np.max(predictions[0]) * 100
        return predicted_label, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Save results to Excel
def save_results_to_excel(df, excel_file):
    try:
        df.to_excel(excel_file, index=False)
        st.success(f"Results saved to {excel_file}")
    except Exception as e:
        st.error(f"Error saving Excel file: {e}")

# Streamlit app
st.title("🔍 Product Detection and Tracking")
st.markdown("Upload images of products for classification and tracking. Results will be saved in an Excel file for future reference.")

# Upload folder
uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("Submit for Detection"):
    # Initialize results DataFrame
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file)
        except Exception as e:
            st.warning(f"Error reading Excel file: {e}")
            df = pd.DataFrame(columns=['S. No.', 'Product Name', 'Time', 'Count'])
    else:
        df = pd.DataFrame(columns=['S. No.', 'Product Name', 'Time', 'Count'])

    processed_count = 0
    error_count = 0

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        img_name = uploaded_file.name
        st.write(f"🖼️ Processing image: {img_name}")
        img_array = preprocess_image(uploaded_file)

        if img_array is None:
            st.warning(f"⚠️ Skipping image due to preprocessing error: {img_name}")
            error_count += 1
            continue

        predicted_label, confidence = predict_image(img_array, model, label_map)
        if predicted_label is None:
            st.warning(f"⚠️ Skipping image due to prediction error: {img_name}")
            error_count += 1
            continue

        # Update results DataFrame
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if predicted_label in df['Product Name'].values:
            df.loc[df['Product Name'] == predicted_label, 'Count'] += 1
            df.loc[df['Product Name'] == predicted_label, 'Time'] = current_time
        else:
            new_row = {
                'S. No.': len(df) + 1,
                'Product Name': predicted_label,
                'Time': current_time,
                'Count': 1
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        st.success(f"✅ Predicted: **{predicted_label}**, Confidence: **{confidence:.2f}%**")
        processed_count += 1

    # Save to Excel
    save_results_to_excel(df, excel_file)

   
