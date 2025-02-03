import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pickle  # Используем pickle
import gdown
import os
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass  # Для центрирования цифры

# Заголовок приложения
st.title("Рисуйте цифру, а модель её распознает!")

# URL модели на Google Drive
url = 'https://drive.google.com/uc?id=1BAJH_MFg6hcs_x_IB6rb-aO_Zfb3kiyy'
model_path = 'best_model.pkl'

# Загрузка модели
if not os.path.exists(model_path):
    try:
        st.write("Скачивание модели...")
        gdown.download(url, model_path, quiet=False)
        st.success("Модель скачана!")
    except Exception as e:
        st.error(f"Ошибка скачивания: {e}")
        st.stop()

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    st.success("Модель загружена!")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# Настройки для рисования
st.sidebar.header("Настройки")
stroke_width = st.sidebar.slider("Толщина линии:", 1, 25, 10)
bg_color = st.sidebar.color_picker("Цвет фона:", "#FFFFFF")
stroke_color = st.sidebar.color_picker("Цвет линии:", "#000000")

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

# Функция центрирования изображения
def center_image(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    shift_x = cols // 2 - int(cx)
    shift_y = rows // 2 - int(cy)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (cols, rows))

# Обработка изображения
if canvas_result.image_data is not None:
    try:
        # Преобразование в ч/б и изменение размера
        image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_BGR2GRAY)

        # Инверсия цветов (MNIST использует белую цифру на чёрном фоне)
        image = cv2.bitwise_not(image)

        # Нормализация значений (от 0 до 1)
        image = image.astype('float32') / 255.0

        # Центрирование цифры
        image = center_image(image)

        # Бинаризация (повышает чёткость границ)
        _, image = cv2.threshold(image, 0.5, 1.0, cv2.THRESH_BINARY)

        # Изменение размера до 28x28 пикселей
        image = cv2.resize(image, (28, 28))

        # Подготовка для модели
        image_flat = image.reshape(1, -1)

        # Предсказание модели
        predictions = model.predict_proba(image_flat)[0]
        predicted_digit = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        # Отображение результатов
        st.header("Результаты")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Нарисованная цифра")
            st.image(image, clamp=True, width=150)

        with col2:
            st.subheader("Предсказание")
            st.markdown(f"**Цифра:** {predicted_digit}")
            st.markdown(f"**Уверенность:** {confidence:.2f}%")

        # Гистограмма предсказаний
        st.subheader("Распределение вероятностей")
        fig, ax = plt.subplots()
        ax.bar(range(10), predictions)
        ax.set_xlabel("Цифра")
        ax.set_ylabel("Вероятность")
        ax.set_xticks(range(10))
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка обработки изображения: {e}")

# Кнопка очистки холста
if st.button("Очистить холст"):
    st.session_state.canvas = None
