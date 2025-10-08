import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
import warnings
import time
from datetime import datetime
import gdown
import boto3   # Only import needed libraries here

aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
region_name = st.secrets["AWS_DEFAULT_REGION"]
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name,
)

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
    /* Main theme colors */
    :root {
        --primary: #4F46E5;
        --secondary: #10B981;
        --background: #F3F4F6;
        --text: #1F2937;
        --danger: #EF4444;
        --warning: #F59E0B;
        --success: #10B981;
    }
    
    /* General styling */
    .stApp {
        background-color: var(--background);
        color: var(--text);
    }
    
    h1, h2, h3 {
        color: var(--primary);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3730A3;
        transform: translateY(-2px);
    }
    
    /* Results styling */
    .result-real {
        font-size: 1.5rem;
        color: var(--success);
        font-weight: bold;
    }
    
    .result-fake {
        font-size: 1.5rem;
        color: var(--danger);
        font-weight: bold;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: var(--primary);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E293B;
    }
    
    .css-1d391kg .sidebar .sidebar-content {
        background-color: #1E293B;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
        margin-top: 2rem;
    }
    
    /* Timeline for How It Works */
    .timeline {
        margin: 20px 0;
        position: relative;
    }
    
    .timeline-item {
        padding: 10px 40px;
        position: relative;
        background-color: inherit;
        width: 100%;
    }
    
    .timeline-item::after {
        content: '';
        position: absolute;
        width: 20px;
        height: 20px;
        background-color: var(--primary);
        border: 4px solid #fff;
        border-radius: 50%;
        top: 15px;
        left: 5px;
        z-index: 1;
    }
    
    .timeline-content {
        padding: 15px;
        background-color: white;
        border-radius: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* FAQ Styling */
    .faq-question {
        font-weight: bold;
        font-size: 1.1rem;
        color: var(--primary);
        margin-top: 1rem;
    }
    
    .faq-answer {
        margin-bottom: 1.5rem;
        padding-left: 1rem;
        border-left: 3px solid var(--secondary);
    }
    
    /* Model info cards */
    .model-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border-left: 4px solid var(--primary);
    }
    
    .model-name {
        font-weight: bold;
        color: var(--primary);
    }
    
    /* Analysis dashboard */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary);
    }
    
    .metric-label {
        color: #6B7280;
        font-size: 0.9rem;
    }
    
    /* Radio buttons */
    div.row-widget.stRadio > div {
        display: flex;
        flex-direction: row;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin-right: 10px;
        border: 1px solid #E5E7EB;
        transition: all 0.3s;
    }
    
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        background-color: #F3F4F6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* File uploader */
    .css-1offfwp {
        border-color: var(--primary) !important;
    }
    
    /* Analysis animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .analyzing {
        animation: pulse 1.5s infinite;
        color: var(--primary);
        font-weight: bold;
    }
    
    /* Contact form styling */
    .contact-input {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        width: 100%;
    }
    
    .contact-textarea {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        width: 100%;
        height: 150px;
    }
    
    .contact-submit {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .contact-submit:hover {
        background-color: #3730A3;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🔹 Load models based on selection
# -------------------------------
@st.cache_resource
def load_model_by_name(model_name):
    s3_bucket = "dfd-models"  # Your bucket name

    # Map model names to exact S3 keys
    model_map = {
        "SE+CNN": "deepfake_cnn+se_model.h5",  # removed trailing space
        "CNN": "deepfake_cnn_model.h5"
    }

    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' not recognized.")

    s3_key = model_map[model_name]
    local_path = s3_key  # Save locally with same name

    # Setup S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )

    # Check if file exists locally
    if not os.path.exists(local_path):
        try:
            # Confirm object exists in S3 first
            s3.head_object(Bucket=s3_bucket, Key=s3_key)
            # Download file
            s3.download_file(s3_bucket, s3_key, local_path)
            print(f"✅ Downloaded '{s3_key}' from S3.")
        except Exception as e:
            raise RuntimeError(f"Failed to download '{s3_key}' from S3: {e}")

    # Load and return Keras model
    return load_model(local_path)


# -------------------------------
# 🔹 Helper Functions
# -------------------------------
def preprocess_frame(frame):
    frame = cv2.resize(frame, (96, 96))
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)

def extract_frames(video_path, max_frames=20):
    """Extract frames from video with robust error handling"""
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            st.error(f"❌ Error: Could not open video file")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.info(f"📹 Video Info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Handle edge cases
        if total_frames <= 0:
            st.warning("⚠️ Could not determine frame count, reading sequentially...")
            # Fallback: read frames sequentially
            count = 0
            while count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                count += 1
        else:
            # Calculate which frames to extract
            if total_frames <= max_frames:
                # Extract all frames if video is short
                frame_indices = list(range(total_frames))
            else:
                # Extract evenly spaced frames
                interval = total_frames / max_frames
                frame_indices = [int(i * interval) for i in range(max_frames)]
            
            st.info(f"🎯 Extracting frames at indices: {frame_indices[:5]}... (showing first 5)")
            
            # Extract frames
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frames.append(frame)
                else:
                    # If seeking fails, try sequential read
                    st.warning(f"⚠️ Could not seek to frame {idx}, trying sequential read...")
        
        cap.release()
        
        # Final status
        if len(frames) == 0:
            st.error("❌ Error: No frames were extracted from the video")
        else:
            st.success(f"✅ Successfully extracted {len(frames)} frames")
        
        return frames
        
    except Exception as e:
        st.error(f"❌ Exception during frame extraction: {str(e)}")
        if 'cap' in locals():
            cap.release()
        return []

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
    real_pct = (real / total) * 100
    fake_pct = (fake / total) * 100
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
    
    # Logo
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        logo_path = "logo.png"
        if not os.path.exists(logo_path):
            gdown.download(
                "https://drive.google.com/uc?id=1nGBRn6Y_nr96Di1pjPGaA2ow1CXIMLq9",
                logo_path,
                quiet=False
            )

        st.image(logo_path, width=150, use_container_width=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 🔍 Analyze Your Video
        Upload a video and let our AI determine whether it's REAL or DEEPFAKE.
    """)
    
    # Model Selection with custom formatting
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
    
   
    
    # Handle checkbox logic for model selection
    if se_cnn:
        model_option = "SE+CNN"
        if cnn: cnn = False
       
    elif cnn:
        model_option = "CNN"
        if se_cnn: se_cnn = False
   
    else:
        model_option = "SE+CNN"  # Default
        se_cnn = True
    
    # Load selected model
    model = load_model_by_name(model_option)
    
    st.markdown(f"*Selected Model:* {model_option}")
    
    # File uploader for video with better styling
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
        
        # Simulate analysis progress
        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.markdown('<p class="analyzing">🔍 Extracting frames...</p>', unsafe_allow_html=True)
            elif i < 70:
                status_text.markdown('<p class="analyzing">⚙ Processing with AI model...</p>', unsafe_allow_html=True)
            else:
                status_text.markdown('<p class="analyzing">📊 Finalizing results...</p>', unsafe_allow_html=True)
            time.sleep(0.03)
        
        # Save video temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        temp_path = temp_file.name
        
        # Predict
        verdict, real_score, fake_score, sample_frames, predictions = predict_video(temp_path, model)
        
        # Display Results in a dashboard layout
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
            
            # Use different colored progress bars for real vs fake
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
            
            # Additional metrics
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
        
        # Show sample frames with hover effect
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
        
        # Analysis summary
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
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            st.warning(f"Could not delete temp file: {e}")

def about_page():
    st.markdown('<div class="main-title">About EchoDetect</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 🔍 Our Mission
        
        *EchoDetect* is dedicated to combating misinformation in the digital age through advanced AI technology. 
        As deepfake technology becomes increasingly sophisticated, the need for reliable detection tools has never been more critical.
        
        Our platform leverages multiple state-of-the-art deep learning models to analyze videos and determine 
        whether they contain artificially generated or manipulated content.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 🧠 Our Technology
        
        EchoDetect employs a multi-model approach to deepfake detection:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown('<div class="model-name">SE+CNN Model</div>', unsafe_allow_html=True)
        st.markdown("""
            Combines Squeeze-and-Excitation networks with Convolutional Neural Networks to:
            - Capture spatial features
            - Recalibrate channel-wise feature responses
            - Focus on the most informative features
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown('<div class="model-name">CNN Model</div>', unsafe_allow_html=True)
        st.markdown("""
            Standard Convolutional Neural Network that excels at:
            - Extracting spatial features
            - Identifying visual patterns
            - Detecting inconsistencies in images
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 📊 Performance & Limitations
        
        Our models have been trained on diverse datasets of real and synthetic videos, achieving:
        
        - *77.24* accuracy on unseen test data
        - *76.2%* precision in detecting deepfakes
        - *79.5%* recall rate
        
        *Limitations:*
        
        - Performance may vary with extremely high-quality deepfakes
        - Low resolution videos might yield less reliable results
        - Heavily compressed videos can affect detection accuracy
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 👥 The Team
        
        EchoDetect was developed by a team of AI researchers and engineers committed to creating 
        tools that help maintain the integrity of digital media in an era of advanced manipulation technologies.
        
        *Our Members:*
        -Bontha Mallikarjun Reddy 23070126026
        -Jhotika Raja 23070126050
        -Eesha Barad 23070126161
        -Karmadeept Sarkar (23070126114)​
        
        - Privacy-first approach
    """)
    st.markdown("</div>", unsafe_allow_html=True)

def how_it_works_page():
    st.markdown('<div class="main-title">How EchoDetect Works</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 🔍 The Detection Process
        
        EchoDetect uses a sophisticated pipeline to analyze videos for signs of manipulation:
    """)
    
    st.markdown('<div class="timeline">', unsafe_allow_html=True)
    
    st.markdown('<div class="timeline-item">', unsafe_allow_html=True)
    st.markdown('<div class="timeline-content">', unsafe_allow_html=True)
    st.markdown("""
        #### 1. Video Upload & Preprocessing
        
        When you upload a video, our system:
        - Extracts key frames at regular intervals
        - Resizes frames to a standard dimension (96x96 pixels)
        - Normalizes pixel values for optimal model performance
    """)
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="timeline-item">', unsafe_allow_html=True)
    st.markdown('<div class="timeline-content">', unsafe_allow_html=True)
    st.markdown("""
        #### 2. Feature Extraction
        
        Our models analyze each frame for:
        - Facial inconsistencies
        - Unnatural lighting and shadows
        - Blending artifacts at boundaries
        - Unusual texture patterns
        - Temporal inconsistencies between frames
    """)
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="timeline-item">', unsafe_allow_html=True)
    st.markdown('<div class="timeline-content">', unsafe_allow_html=True)
    st.markdown("""
        #### 3. Deep Learning Analysis
        
        Depending on the selected model:
        - *SE+CNN*: Focuses on recalibrating feature importance
        - *CNN*: Identifies spatial inconsistencies
        - *CNN+LSTM*: Analyzes both spatial and temporal patterns
    """)
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="timeline-item">', unsafe_allow_html=True)
    st.markdown('<div class="timeline-content">', unsafe_allow_html=True)
    st.markdown("""
        #### 4. Frame-by-Frame Prediction
        
        Each frame receives a confidence score:
        - Values > 0.5 indicate the frame appears genuine
        - Values < 0.5 suggest the frame may be manipulated
    """)
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="timeline-item">', unsafe_allow_html=True)
    st.markdown('<div class="timeline-content">', unsafe_allow_html=True)
    st.markdown("""
        #### 5. Result Aggregation
        
        Individual frame predictions are combined to determine the final verdict:
        - The percentage of frames classified as real vs. fake is calculated
        - A confidence score is generated for the entire video
        - The final verdict (REAL or FAKE) is presented with supporting metrics
    """)
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 🧪 Technical Details
        
        #### Model Architectures
        
        *SE+CNN Architecture:*
        - Input layer: 96x96x3 (RGB image)
        - 4 convolutional blocks with SE modules
        - Global average pooling
        - Dense layers with dropout for regularization
        - Binary classification output
        
        *CNN Architecture:*
        - Input layer: 96x96x3 (RGB image)
        - 5 convolutional blocks with max pooling
        - Batch normalization
        - Dense layers with dropout
        - Binary classification output
        
      
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 📈 Interpreting Results
        
        #### Understanding Confidence Scores
        
        - *High real confidence (>80%)*: Strong indication the video is authentic
        - *High fake confidence (>80%)*: Strong indication the video is manipulated
        - *Mixed confidence (40-60%)*: Uncertain classification, requiring further investigation
        
        #### False Positives & Negatives
        
        Certain factors can affect detection accuracy:
        - Heavy video compression
        - Poor lighting conditions
        - Unusual camera movements
        - Face occlusions
        
        For best results, use videos with:
        - Clear visibility of subjects
        - Minimal compression artifacts
        - Standard frame rates
        - Good lighting conditions
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def faq_page():
    st.markdown('<div class="main-title">Frequently Asked Questions</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown('<div class="faq-question">What is a deepfake?</div>', unsafe_allow_html=True)
    #st.markdown('<div class="faq-answer">A deepfake is synthetic media where a person\'s likeness is replaced with someone else\'s using artificial intelligence. This technology can create convincing but fabricated videos or images that show
    st.markdown('<div class="faq-question">Is my data private when I use EchoDetect?</div>', unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">Yes. All video processing happens on our secure servers, and we do not store your uploaded videos after analysis. Your data privacy is our priority, and no personal information is collected during the analysis process.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="faq-question">Which model should I choose for analysis?</div>', unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">For most general use cases, we recommend the SE+CNN model as it provides the best balance of accuracy and speed. The CNN+LSTM model is better for videos with complex motion patterns, while the basic CNN model works well for simple footage.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="faq-question">Can EchoDetect detect all types of deepfakes?</div>', unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">While our technology is advanced, no detection system can guarantee 100% accuracy for all deepfakes. Particularly sophisticated deepfakes created with cutting-edge technology might sometimes evade detection. We continuously update our models to improve detection capabilities.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="faq-question">How long does the analysis take?</div>', unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">Analysis time depends on video length and the selected model. Most videos under 1 minute are processed in less than 30 seconds. Longer videos may take up to a few minutes to complete.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="faq-question">What should I do if I encounter a deepfake?</div>', unsafe_allow_html=True)
    st.markdown('<div class="faq-answer">If you confirm a video is a deepfake, consider: 1) Not sharing it further, 2) Reporting it to the platform where you found it, 3) Informing the person being impersonated if possible, and 4) Sharing information about deepfake detection tools like EchoDetect to raise awareness.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def contact_page():
    st.markdown('<div class="main-title">Contact Us</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 📬 Get in Touch
        
        Have questions, feedback, or need support with EchoDetect? We'd love to hear from you!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            #### Contact Information
            
            📧 *Email*: mallikarjun57561@gmail.com
            
            📞 *Phone*: +91 9032257561
            
            🏢 *Address*:  
            
        """)
    
    with col2:
        st.markdown("#### Send Us a Message")
        name = st.text_input("Name")
        email = st.text_input("Email")
        subject = st.selectbox("Subject", ["General Inquiry", "Technical Support", "Feedback", "Business Inquiry", "Report an Issue"])
        message = st.text_area("Message", height=150)
        
        if st.button("Send Message"):
            st.success("Thank you for your message! We'll get back to you shortly.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 🤝 Partnerships & Research Collaboration
        
        We're open to collaborations with:
        
        - Academic institutions researching media forensics
        - Organizations working on digital literacy and misinformation
        - Media companies interested in content verification
        - Technology partners developing complementary solutions
        
        For partnership inquiries, please contact: mallikarjun57561@gmail.com
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        ### 💡 Suggestions & Feature Requests
        
        We're constantly improving EchoDetect based on user feedback. Have an idea for making our tool better?
        
        Let us know what features you'd like to see in future updates!
    """)
    
    feature_request = st.text_area("Your suggestion:", height=100)
    if st.button("Submit Suggestion"):
        st.success("Thank you for your suggestion! We appreciate your input.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 🔹 Main App Logic
# -------------------------------
def main():
    # Display navigation
    page = navigation()
    
    # Display selected page
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
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("© 2025 EchoDetect | Powered by Advanced AI | Privacy Policy | Terms of Service", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()


