import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# Заголовок приложения
st.title("Рисуйте цифру, а модель её распознает!")

# Загрузка модели
model_path = 'best_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
st.success("Модель загружена!")

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

# Функция для центрирования изображения
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
        # 1. Перевод в ч/б
        image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_BGR2GRAY)

        # 2. Инверсия цветов (если фон белый, а цифра чёрная)
        image = cv2.bitwise_not(image)

        # 3. Центрирование цифры
        image = center_image(image)

        # 4. Изменение размера
        image = cv2.resize(image, (28, 28))

        # 5. Нормализация
        image = image.astype('float32') / 255.0

        # 6. Бинаризация (0 и 1 вместо оттенков серого)
        _, image = cv2.threshold(image, 0.5, 1.0, cv2.THRESH_BINARY)

        # 7. Разворачивание вектор для подачи в модель
        image_flat = image.reshape(1, -1)

        # Предсказание
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

        # Гистограмма
        st.subheader("Распределение вероятностей")
        fig, ax = plt.subplots()
        ax.bar(range(10), predictions)
        ax.set_xlabel("Цифра")
        ax.set_ylabel("Вероятность")
        ax.set_xticks(range(10))
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка: {e}")

# Кнопка очистки холста
if st.button("Очистить холст"):
    st.session_state.canvas = None
