
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cats vs Dogs CNN",
    page_icon="🐾"
)

st.title("🐱 Cats vs Dogs 🐶")
st.write("CNN Image Classification Model")

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_dogs_cnn_small.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    # Resize and normalize image
    img = image.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Image"):
        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction >= 0.5:
            st.success("🐶 Prediction: Dog")
            confidence = prediction * 100
        else:
            st.success("🐱 Prediction: Cat")
            confidence = (1 - prediction) * 100

        st.write(f"Model Confidence: {confidence:.2f}%")
