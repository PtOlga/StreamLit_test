import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# App title
st.title("Draw a digit, and the model will recognize it!")

# Load the model
model_path = 'best_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
st.success("Model loaded successfully!")

# Drawing settings
st.sidebar.header("Settings")
stroke_width = st.sidebar.slider("Line thickness:", 10, 25, 25)
bg_color = st.sidebar.color_picker("Background color:", "#FFFFFF")
stroke_color = st.sidebar.color_picker("Line color:", "#000000")

# Drawing canvas
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

# Function to center the image
def center_image(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    shift_x = cols // 2 - int(cx)
    shift_y = rows // 2 - int(cy)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (cols, rows))

# Process the image if the canvas is not empty
if canvas_result.image_data is not None:
    try:
        # Convert canvas to grayscale image
        image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_BGR2GRAY)

        # Check if the image is empty (all white pixels)
        if np.all(image == 255):  
            st.warning("Draw a digit first!")
        else:
            # Invert colors if needed
            image = cv2.bitwise_not(image)

            # Center the digit
            image = center_image(image)

            # Resize to 28x28
            image = cv2.resize(image, (28, 28))

            # Normalize pixel values
            image = image.astype('float32') / 255.0

            # Binarization
            _, image = cv2.threshold(image, 0.5, 1.0, cv2.THRESH_BINARY)

            # Flatten the image for the model
            image_flat = image.reshape(1, -1)

            # Model prediction
            predictions = model.predict_proba(image_flat)[0]
            predicted_digit = np.argmax(predictions)
            confidence = np.max(predictions) * 100

            # Display results
            st.header("Results")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Drawn Digit")
                st.image(image, clamp=True, width=150)

            with col2:
                st.subheader("Prediction")
                st.markdown(f"**Digit:** {predicted_digit}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")

            # Probability distribution chart
            st.subheader("Probability Distribution")
            fig, ax = plt.subplots()
            ax.bar(range(10), predictions)
            ax.set_xlabel("Digit")
            ax.set_ylabel("Probability")
            ax.set_xticks(range(10))
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

# Fix: Properly reset the canvas when clicking "Clear Canvas"
if st.button("Clear Canvas"):
    st.experimental_rerun()
