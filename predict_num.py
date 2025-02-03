import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# Заголовок приложения
st.title("Рисуйте цифру, а модель её распознает!")

# Загрузка модели
try:
    model = joblib.load("best_model_rf.joblib")
    st.success("Модель успешно загружена!")
except Exception as e:
    st.error(f"Ошибка при загрузке модели: {e}")
    st.stop()

# Настройки для рисования
st.sidebar.header("Настройки")
stroke_width = st.sidebar.slider("Толщина линии:", 1, 25, 10)
stroke_color = st.sidebar.color_picker("Цвет линии:", "#FFFFFF")
bg_color = st.sidebar.color_picker("Цвет фона:", "#000000")

# Область для рисования
st.sidebar.header("Область для рисования")
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

# Обработка нарисованного изображения
if canvas_result.image_data is not None:
    try:
        # Преобразуем изображение в черно-белое и изменяем размер
        image = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype('float32') / 255.0
        image = 1 - image  # Инвертируем цвета
        image = image.reshape(1, -1)  # Преобразуем в 1D-массив для модели

        # Предсказание с использованием модели
        predictions = model.predict_proba(image)[0]
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        # Отображение результатов
        st.header("Результаты распознавания")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Нарисованная цифра")
            st.image(image.reshape(28, 28), width=150)

        with col2:
            st.subheader("Предсказание модели")
            st.markdown(f"**Цифра:** {predicted_digit}")
            st.markdown(f"**Уверенность:** {confidence:.2f}%")

        # Гистограмма распределения вероятностей
        st.subheader("Распределение вероятностей")
        fig, ax = plt.subplots()
        ax.bar(range(10), predictions)
        ax.set_xlabel("Цифра")
        ax.set_ylabel("Вероятность")
        ax.set_xticks(range(10))
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка при обработке изображения: {e}")
