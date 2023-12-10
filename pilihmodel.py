import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Fungsi untuk memuat model yang dipilih
def load_model_file(model_path):
    try:
        loaded_model = load_model(model_path)
        st.success("Model berhasil dimuat.")
        return loaded_model
    except Exception as e:
        st.error(f"Error saat membaca model: {e}")
        return None

# Fungsi untuk melakukan prediksi diabetes
def predict_diabetes(model, input_data):
    if model:
        prediction = model.predict(input_data)
        return prediction[0]
    return None

# Fungsi untuk menampilkan hasil prediksi
def display_diagnosis(prediction):
    if prediction is not None:
        return 'Orang tersebut menderita diabetes' if prediction > 0.5 else 'Orang tersebut tidak menderita diabetes'
    return ''

# Aplikasi Streamlit
st.title('Prediksi Diabetes Menggunakan Neural Network')

# Pilihan model
model_options = ['Model_ann', 'Model_cnn']  # Tambahkan nama model sesuai kebutuhan
selected_model = st.radio('Pilih Model:', model_options)

model_folder = 'luckyseven-ux/diabetes/'  # Sesuaikan path folder ini

if selected_model == 'Model_ann':
    model_path = model_folder + 'diabetes_ANN.h5'
elif selected_model == 'Model_cnn':
    model_path = model_folder + 'diabetes_CNN.h5'

diabetes_model = load_model_file(model_path)

# Form input
input_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
input_data = [st.text_input(f'{column}:') for column in input_columns]

# Mengatasi input string kosong
input_data = [float(val) if val != '' else 0.0 for val in input_data]

# Konversi menjadi array NumPy
input_data = np.array(input_data).reshape(1, -1)

# Prediksi dan hasil
diagnosis = ''
if st.button('Hasil Test Diabetes'):
    prediction = predict_diabetes(diabetes_model, input_data)
    diagnosis = display_diagnosis(prediction)

st.success(diagnosis)
