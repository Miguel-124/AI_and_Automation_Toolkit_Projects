import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image
# import os, certifi
# os.environ['SSL_CERT_FILE'] = certifi.where()

def load_model():
    """Load the pre-trained MobileNetV2 model."""
    model = MobileNetV2(weights='imagenet')
    return model

def pre_process_image(image):
    """Preprocess the image for MobileNetV2."""
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = preprocess_input(img)  # Preprocess the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image(model, image):
    """Classify the image using the pre-trained model."""
    try:
        processed_image = pre_process_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        return decoded_predictions
    except Exception as e:
        st.error(f"Error during image classification: {e}")
        return None
    
def main():
    st.set_page_config(page_title="Image Classifier", page_icon="🤖", layout="centered")
    st.title("Image Classifier 🤖")
    st.markdown("Upload an image and get predictions on what it contains.")

    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()
    uploaded_file = st.file_uploader("Upload an image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1) Otwórz plik jako PIL Image
        pil_img = Image.open(uploaded_file).convert("RGB")
        # 2) Pokaż w Streamlit
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        btn = st.button("Classify Image")
        if btn:
            with st.spinner("Classifying..."):
                # 3) Przekaż oba argumenty: model i PIL-owy obraz
                predictions = classify_image(model, pil_img)

                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"{label}: {score:.2%}")

if __name__ == "__main__":
    main()