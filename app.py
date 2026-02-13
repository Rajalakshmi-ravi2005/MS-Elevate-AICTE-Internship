import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Melanoma Detection System",
    layout="wide",
    page_icon="🩺"
)

st.title("🩺 Melanoma Detection & Risk Assessment")
st.markdown("### AI-Powered Early Detection System")
st.markdown("---")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_cnn():
    model = load_model("melanoma_model.h5", compile=False)
    return model

cnn_model = load_cnn()

# ------------------ GRAD-CAM FUNCTION (KERAS 3 SAFE) ------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer):

    # Create feature extractor model (only conv output)
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    # Create classifier model (remaining layers after conv)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_output = feature_extractor(img_array)
        tape.watch(conv_output)
        predictions = classifier_model(conv_output)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


# ------------------ LAYOUT ------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📸 Upload Skin Image")
    uploaded_file = st.file_uploader(
        "Choose JPG/PNG image",
        type=["jpg", "jpeg", "png"]
    )

with col_right:
    st.subheader("⚠️ Patient Risk Profile")

    age = st.slider("Age", 0, 100, 30)
    skin_type = st.selectbox("Skin Type", ("Light", "Medium", "Dark"))
    family_history = st.selectbox("Family History?", ("No", "Yes"))
    sun_exposure = st.selectbox("Sun Exposure?", ("Low", "High"))
    mole_count = st.slider("Total Moles", 0, 100, 5)

# ------------------ PREDICTION ------------------
if uploaded_file:

    target_size = tuple(cnn_model.input_shape[1:3])

    img = image.load_img(uploaded_file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Run model once (IMPORTANT for Keras 3)
    prediction = cnn_model(img_array, training=False)
    image_prob = float(prediction.numpy()[0][0])

    st.image(uploaded_file, caption="Uploaded Image", width=300)

    # ------------------ RISK CALCULATION ------------------
    risk_score = np.array([
        age / 100,
        1 if skin_type == "Light" else 0.5 if skin_type == "Medium" else 0,
        1 if family_history == "Yes" else 0,
        1 if sun_exposure == "High" else 0,
        min(mole_count / 50, 1)
    ]).mean()

    combined_score = (0.7 * image_prob) + (0.3 * risk_score)

    st.markdown("---")
    st.subheader("📊 Detection Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Image Score", f"{image_prob:.1%}")
    col2.metric("Risk Score", f"{risk_score:.1%}")
    col3.metric("Combined Risk", f"{combined_score:.1%}")

    # ------------------ GRAD-CAM ------------------
    st.markdown("---")
    st.subheader("🔥 Model Attention (Grad-CAM)")

    # Automatically find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(cnn_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        st.error("No Conv2D layer found.")
    else:
        heatmap = make_gradcam_heatmap(img_array, cnn_model, last_conv_layer)

        heatmap = cv2.resize(heatmap, target_size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cm.jet(heatmap)[:, :, :3]
        heatmap = np.uint8(heatmap * 255)

        original_img = image.img_to_array(
            image.load_img(uploaded_file, target_size=target_size)
        )

        superimposed_img = heatmap * 0.4 + original_img
        superimposed_img = np.uint8(superimposed_img)

        st.image(superimposed_img, width=300)

    # ------------------ FINAL RISK ------------------
    st.markdown("---")

    if combined_score < 0.4:
        st.success("LOW RISK - Continue monitoring.")
    elif combined_score < 0.7:
        st.warning("MODERATE RISK - Dermatologist visit recommended.")
    else:
        st.error("HIGH RISK - Immediate medical consultation required.")

st.markdown("---")
st.caption("AI Detection Tool • Not a Medical Diagnosis")
