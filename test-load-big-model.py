import streamlit as st
from tensorflow.keras.models import load_model
import gdown
import os

url = "https://drive.google.com/uc?id=1AyPDoibUsYhx1CnFkFouPh_fIy0pXpB5"
model_path = "model.h5"

# Загрузить файл, если его нет
if not os.path.exists(model_path):
    st.write("Скачивание модели...")
    gdown.download(url, model_path, quiet=False)

# Проверить размер файла
file_size = os.path.getsize(model_path) / 1024 / 1024  # в МБ
if file_size < 1:
    st.error("Файл модели поврежден. Удалите его и попробуйте снова.")
    st.stop()

# Загрузить модель
try:
    model = load_model(model_path)
    st.success("Модель успешно загружена!")
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")