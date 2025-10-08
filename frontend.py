import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
import warnings
import time
from datetime import datetime
import boto3  # ADDED FOR S3 CONNECTION

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------------
# S3-based model download helper
# -------------------------------
def fetch_model_from_s3(model_path, s3_key):
    if not os.path.exists(model_path):
        try:
            s3 = boto3.client('s3')
            s3.download_file('dfd-models', s3_key, model_path)
            print(f"Downloaded {s3_key} from S3 bucket dfd-models.")
        except Exception as e:
            print(f"Could not download {s3_key} from S3: {e}")
            raise

# -------------------------------
# 🔹 Load models based on selection (S3 ONLY)
# -------------------------------
@st.cache_resource
def load_model_by_name(model_name):
    if model_name == "SE+CNN":
        model_path = "deepfake_cnn+se_model.h5"
        s3_key = "deepfake_cnn+se_model.h5"
        fetch_model_from_s3(model_path, s3_key)
        return load_model(model_path)
    elif model_name == "CNN":
        model_path = "deepfake_cnn_model.h5"
        s3_key = "deepfake_cnn_model.h5"
        fetch_model_from_s3(model_path, s3_key)
        return load_model(model_path)
    else:
        raise ValueError("Model not recognized")

# -------------------------------
# 🔹 Page Navigation Function
# -------------------------------
def navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About", "How It Works", "FAQ", "Contact"], key="navigation_radio")
    return page

# -------------------------------
# 🔹 Set up Streamlit page
# -------------------------------
st.set_page_config(
    page_title="EchoDetect", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 🔹 Custom CSS Styling
# -------------------------------
st.markdown("""
<style>
/* ... (Your unchanged CSS remains here) ... */
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🔹 Helper Functions
# -------------------------------
def preprocess_frame(frame):
    frame = cv2.resize(frame, (96, 96))
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)

def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos % interval == 0:
            frames.append(frame)
            count += 1
    cap.release()
    return frames

def predict_video(video_path, model):
    frames = extract_frames(video_path)
    real, fake = 0, 0
    predictions = []

    for frame in frames:
        pred = model.predict(preprocess_frame(frame), verbose=0)[0][0]
        predictions.append(pred)
        if pred > 0.5:
            real += 1
        else:
            fake += 1

    total = real + fake
    real_pct = (real / total) * 100 if total else 0
    fake_pct = (fake / total) * 100 if total else 0
    verdict = "✅ REAL" if real_pct > fake_pct else "❌ FAKE"

    return verdict, real_pct, fake_pct, frames, predictions

# -------------------------------
# 🔹 Page Content Functions
# -------------------------------
def home_page():
    col1, col2 = st.columns([1, 2])
    with col2:
        st.markdown('<div class="main-title">🎬 EchoDetect</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Advanced Deepfake Detection Powered by AI</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        logo_path = "logo.png"
        # Logo download logic (unchanged for S3-only model loading)
        # If you wish, move logo to S3 and adapt fetch_model_from_s3 as above.

        if not os.path.exists(logo_path):
            pass # Leave as-is, or remove logo download logic; not required for model.

        st.image(logo_path, width=150, use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 🔍 Analyze Your Video
        Upload a video and let our AI determine whether it's REAL or DEEPFAKE.
    """)

    st.markdown("#### Select a Detection Model")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        se_cnn = st.checkbox("SE+CNN", value=True)
        st.markdown('<div class="model-name">SE+CNN</div>', unsafe_allow_html=True)
        st.write("Squeeze-and-Excitation with CNN for feature extraction")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        cnn = st.checkbox("CNN", value=False)
        st.markdown('<div class="model-name">CNN</div>', unsafe_allow_html=True)
        st.write("Convolutional Neural Network for spatial features")
        st.markdown("</div>", unsafe_allow_html=True)

    # Checkbox logic for model selection (as before)
    if se_cnn:
        model_option = "SE+CNN"
        if cnn: cnn = False
    elif cnn:
        model_option = "CNN"
        if se_cnn: se_cnn = False
    else:
        model_option = "SE+CNN"
        se_cnn = True

    # Load selected model (uses S3 connection now)
    model = load_model_by_name(model_option)

    st.markdown(f"*Selected Model:* {model_option}")
    st.markdown("#### Upload Video")
    video_file = st.file_uploader("📁 Select a video file to analyze", type=["mp4", "avi", "mov"])
    st.markdown("</div>", unsafe_allow_html=True)

    if video_file is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎥 Preview")
        st.video(video_file)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Analysis")

        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.markdown('<p class="analyzing">🔍 Extracting frames...</p>', unsafe_allow_html=True)
            elif i < 70:
                status_text.markdown('<p class="analyzing">⚙ Processing with AI model...</p>', unsafe_allow_html=True)
            else:
                status_text.markdown('<p class="analyzing">📊 Finalizing results...</p>', unsafe_allow_html=True)
            time.sleep(0.03)

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        temp_path = temp_file.name

        verdict, real_score, fake_score, sample_frames, predictions = predict_video(temp_path, model)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Result")
            if verdict.startswith("✅"):
                st.markdown(f'<p class="result-real">{verdict}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="result-fake">{verdict}</p>', unsafe_allow_html=True)

            st.markdown("#### Confidence Scores")
            st.markdown(f"🔍 *Real Confidence:* {real_score:.2f}%")
            st.markdown(f"🧪 *Fake Confidence:* {fake_score:.2f}%")

            if verdict.startswith("✅"):
                st.markdown(f"""
                    <div style="background-color: #e2f6e9; border-radius: 10px; padding: 10px; margin-top: 10px;">
                        <div style="background-color: #10B981; width: {real_score}%; height: 20px; border-radius: 5px;"></div>
                        <p style="text-align: center; margin-top: 5px; font-size: 0.8rem; color: #10B981;">Real: {real_score:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color: #fee2e2; border-radius: 10px; padding: 10px; margin-top: 10px;">
                        <div style="background-color: #EF4444; width: {fake_score}%; height: 20px; border-radius: 5px;"></div>
                        <p style="text-align: center; margin-top: 5px; font-size: 0.8rem; color: #EF4444;">Fake: {fake_score:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        with col2:
            st.markdown("#### Analysis Details")
            st.markdown(f"*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"*Model Used:* {model_option}")
            st.markdown(f"*Frames Analyzed:* {len(sample_frames)}")

            confidence_variance = np.var(predictions) * 100
            confidence_mean = np.mean(predictions) * 100

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{confidence_mean:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Average Confidence</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{confidence_variance:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Confidence Variance</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### 🔎 Sample Frames Analysis")
            cols = st.columns(5)
            for i, frame in enumerate(sample_frames[:5]):
                with cols[i]:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    confidence = predictions[i] * 100 if i < len(predictions) else 0
                    if confidence > 50:
                        st.markdown(f'<p style="text-align: center; color: #10B981; font-size: 0.8rem;">Frame {i+1}: {confidence:.1f}% Real</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p style="text-align: center; color: #EF4444; font-size: 0.8rem;">Frame {i+1}: {100-confidence:.1f}% Fake</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Analysis Summary")

        if verdict.startswith("✅"):
            st.markdown("""
                #### This video appears to be *REAL*
                Our AI model has analyzed multiple frames from your video and determined it's likely authentic content.
                The high real confidence score indicates minimal signs of manipulation or deepfake characteristics.
                *What does this mean?*
                - The video likely shows genuine footage without AI manipulation
                - Facial features, movements, and lighting patterns appear natural
                - No significant artifacts that would suggest synthetic generation
            """)
        else:
            st.markdown("""
                #### This video appears to be a *DEEPFAKE*
                Our AI model has analyzed multiple frames from your video and detected characteristics consistent with manipulated content.
                The high fake confidence score indicates signs of artificial generation or manipulation.
                *What does this mean?*
                - The video shows signs of AI manipulation or synthetic generation
                - Facial features, movements, or lighting patterns may contain inconsistencies
                - Potential artifacts that suggest the content was artificially created or altered
            """)
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            os.remove(temp_path)
        except Exception as e:
            st.warning(f"Could not delete temp file: {e}")

# ... About, How It Works, FAQ, Contact pages (remain unchanged!) ...

def main():
    page = navigation()
    if page == "Home":
        home_page()
    elif page == "About":
        about_page()
    elif page == "How It Works":
        how_it_works_page()
    elif page == "FAQ":
        faq_page()
    elif page == "Contact":
        contact_page()
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("© 2025 EchoDetect | Powered by Advanced AI | Privacy Policy | Terms of Service", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
