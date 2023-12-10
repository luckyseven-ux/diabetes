import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Function to load the selected model
def load_model_file(model_path):
    try:
        loaded_model = load_model(model_path)
        st.success("Model berhasil dimuat.")
        return loaded_model
    except Exception as e:
        st.error(f"Error saat membaca model: {e}")
        return None

# Function to make diabetes prediction
def predict_diabetes(model, input_data):
    if model:
        prediction = model.predict(input_data)
        return prediction[0]
    return None

# Function to display the prediction result
def display_diagnosis(prediction):
    if prediction is not None:
        return 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
    return ''

# Streamlit app
st.title('Prediksi Diabetes Menggunakan Neural Network')

# Load models
model_options = ['Model_ann', 'Model_cnn']  # Add model names accordingly
selected_model = st.radio('Pilih Model:', model_options)

if selected_model == 'Model_ann':
    model_path = 'E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\diabetes_ANN.h5'
# Update with the correct file path
elif selected_model == 'Model_cnn':
    model_path = 'E:\\syntax code\\python\\jupytr\\neural network\\diabetes\\model\\diabetes_CNN.h5'

diabetes_model = load_model_file(model_path)

# Input form
input_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
input_data = [st.text_input(f'{column}:') for column in input_columns]

# Handling empty string inputs
input_data = [float(val) if val != '' else 0.0 for val in input_data]

# Convert to NumPy array
input_data = np.array(input_data).reshape(1, -1)

# Prediction and result
diagnosis = ''
if st.button('Diabetes Test Result'):
    prediction = predict_diabetes(diabetes_model, input_data)
    diagnosis = display_diagnosis(prediction)

st.success(diagnosis)
