
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_cat_model.keras")

model = load_model()

# UI
st.title("🐶 Dog vs Cat Classifier")
st.write("Upload an image and let AI predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Your uploaded image", width="stretch")

    if st.button("Predict Image", type="primary"):
        with st.spinner("Analyzing image..."):
            # Preprocess image
            image = image.resize((160, 160))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            score = float(
                model.predict(image_array, verbose=0)[0][0]
            )

            if score >= 0.5:
                label = "Dog 🐶"
                confidence = score
            else:
                label = "Cat 🐱"
                confidence = 1 - score

        st.success(f"Prediction: {label}")
        st.metric("Model score", f"{confidence:.2%}")

        st.progress(confidence)

        st.caption(
            "The score is the model's output, not a guarantee "
            "that the prediction is correct."
        )
