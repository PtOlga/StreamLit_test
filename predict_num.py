import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import joblib
import gdown
import os
import matplotlib.pyplot as plt

# Заголовок приложения
st.title("Рисуйте цифру, а модель её распознает!")

# URL модели на Google Drive (ЗАМЕНИТЕ НА ВАШ ID!)
url = 'https://drive.google.com/uc?id=1AyPDoibUsYhx1CnFkFouPh_fIy0pXpB5'
model_path = 'best_model_rf.joblib'

# Загрузка модели с Google Drive
if not os.path.exists(model_path):
    try:
        st.write("Скачивание модели...")
        gdown.download(url, model_path, quiet=False)
        st.success("Модель скачана!")
    except Exception as e:
        st.error(f"Ошибка скачивания: {e}")
        st.stop()

# Загрузка модели в память
try:
    model = joblib.load(model_path)
    st.success("Модель загружена!")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# Настройки для рисования
st.sidebar.header("Настройки")
stroke_width = st.sidebar.slider("Толщина линии:", 1, 25, 10)
stroke_color = st.sidebar.color_picker("Цвет линии:", "#FFFFFF")
bg_color = st.sidebar.color_picker("Цвет фона:", "#000000")

# Область для рисования
canvas_result = st_canvas(
    fill_color=bg_color,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Обработка изображения
if canvas_result.image_data is not None:
    try:
        # Подготовка изображения
        image = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype('float32') / 255.0
        image = 1 - image  # Инверсия цветов
        image = image.reshape(1, -1)  # 1D-массив

        # Предсказание
        predictions = model.predict_proba(image)[0]
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        # Отображение результатов
        st.header("Результаты")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Нарисованная цифра")
            st.image(image.reshape(28, 28), width=150)

        with col2:
            st.subheader("Предсказание")
            st.markdown(f"**Цифра:** {predicted_digit}")
            st.markdown(f"**Уверенность:** {confidence:.2f}%")

        # Гистограмма вероятностей
        st.subheader("Распределение вероятностей")
        fig, ax = plt.subplots()
        ax.bar(range(10), predictions)
        ax.set_xlabel("Цифра")
        ax.set_ylabel("Вероятность")
        ax.set_xticks(range(10))
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка: {e}")
