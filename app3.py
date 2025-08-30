import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import time
import pandas as pd
import base64
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO

# Define class labels and image parameters
CLASS_LABELS = ['pituitary', 'glioma', 'notumor', 'meningioma']
IMAGE_SIZE = 128

# Create directory for uploaded files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
# @st.cache_resource
# def load_tumor_model():
#     try:
#         model = load_model('models/model.h5')
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

def build_model():
    """Builds the VGG16-based model architecture."""
    base_model = tf.keras.applications.VGG16(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    for layer in base_model.layers:
        layer.trainable = False
    
    base_model.layers[-2].trainable = True
    base_model.layers[-3].trainable = True
    base_model.layers[-4].trainable = True
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASS_LABELS), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    return model

@st.cache_resource
def load_tumor_model():
    """Loads the trained model weights into the defined architecture."""
    try:
        model = build_model()
        model.load_weights('models/model.weights.h5') 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the 'models/model.weights.h5' file exists.")
        return None

def predict_tumor(image_path, model):
    try:
        # Open and process the image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        # Get prediction scores for visualization
        all_scores = {CLASS_LABELS[i]: float(predictions[0][i]) for i in range(len(CLASS_LABELS))}

        if CLASS_LABELS[predicted_class_index] == 'notumor':
            return "No Tumor Detected", confidence_score, CLASS_LABELS[predicted_class_index], all_scores
        else:
            return f"{CLASS_LABELS[predicted_class_index].capitalize()} Tumor Detected", confidence_score, CLASS_LABELS[predicted_class_index], all_scores
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return "Error", 0.0, "error", {}
    
#     return fig
def create_radial_progress(confidence, class_name):
    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # Create a custom colormap based on confidence level and class
    if class_name == 'notumor':
        cmap = LinearSegmentedColormap.from_list("", ["#81C784", "#2E7D32"])
    else:
        if confidence > 0.8:
            cmap = LinearSegmentedColormap.from_list("", ["#EF5350", "#B71C1C"])
        elif confidence > 0.6:
            cmap = LinearSegmentedColormap.from_list("", ["#FF8A65", "#E64A19"])
        else:
            cmap = LinearSegmentedColormap.from_list("", ["#FFB74D", "#F57C00"])
    
    fig, ax = plt.subplots(figsize=(1.8, 1.8), subplot_kw={'projection': 'polar'})
    
    # Background ring
    theta = np.linspace(0, 2*np.pi, 100)
    radii = np.ones_like(theta) * 0.65
    ax.plot(theta, radii, color='#e0e0e0', linewidth=6, alpha=0.5)
    
    # Progress ring
    end_angle = 2*np.pi * confidence
    theta_progress = np.linspace(0, end_angle, 100)
    
    for i in range(len(theta_progress)-1):
        color = cmap(i/len(theta_progress))
        ax.plot(theta_progress[i:i+2], np.ones(2) * 0.65, color=color, linewidth=6)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    
    # Color for text based on class
    if class_name == 'notumor':
        color = '#2E7D32'
    else:
        if confidence > 0.8:
            color = '#C62828'
        elif confidence > 0.6:
            color = '#E64A19'
        else:
            color = '#F57C00'
    
    # Shifted upward text position
    ax.text(0, 0, f"{confidence*100:.2f}%", fontsize=12, fontweight='bold',
            ha='center', va='center', color=color)

    plt.tight_layout()
    ax.spines['polar'].set_visible(False)

    return fig

def create_confidence_bars(scores):
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(8, 3))
    classes = list(sorted_scores.keys())
    values = list(sorted_scores.values())
    
    # Define colors based on class and confidence
    colors = []
    for i, cls in enumerate(classes):
        if cls == 'notumor':
            colors.append('#2E7D32')  # Green for no tumor
        else:
            # Reds for tumors, intensity based on confidence
            if values[i] > 0.8:
                colors.append('#C62828')  # Deep red for high confidence
            elif values[i] > 0.5:
                colors.append('#E64A19')  # Orange-red for medium confidence
            else:
                colors.append('#FFB74D')  # Light orange for low confidence
    
    y_pos = np.arange(len(classes))
    
    # Create horizontal bar chart
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, height=0.6)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{width*100:.2f}%", va='center', fontsize=10, fontweight='bold')
    
    # Add drop shadow effect to bars
    for bar in bars:
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.add_patch(plt.Rectangle((x, y-0.02), w, h, color='gray', alpha=0.2))
    
    # Customize appearance
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.capitalize() for c in classes], fontsize=10, fontweight='semibold')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score', fontsize=10, fontweight='semibold')
    ax.set_title('Prediction Confidence by Class', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_sample_images():
    samples = {
        'pituitary': 'samples/pituitary_sample.jpg',
        'glioma': 'samples/glioma_sample.jpg',
        'meningioma': 'samples/meningioma_sample.jpg',
        'notumor': 'samples/notumor_sample.jpg'
    }
    return samples

def get_image_as_base64(path):
    """Convert an image to base64 string for backgrounds"""
    try:
        img = Image.open(path)
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
    except:
        return None

def main():
    st.set_page_config(
        page_title="NeuroScan AI | Brain MRI Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Sidebar from app1.py: ------------------------------------
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.image(
            "logo3.png",
            use_container_width=True
        )
        
        st.markdown("## 🔍 Project Information")
        st.info("""
        **Brain MRI Tumor Detection System**
        
        A deep learning application for automated detection and classification of brain tumors from MRI scans.
        
        Version: 1.0.0
        """)
        
        st.markdown("## ⚠️ Disclaimer")
        st.warning("""
        This application is for research and educational purposes only. It is not FDA/CE approved and should not be used for clinical diagnosis.
        
        Always consult with a qualified healthcare professional for proper medical diagnosis and treatment.
        """)
        
        st.markdown("## 🧠 About the Model")
        st.markdown("""
        This system uses a deep learning model based on the VGG16 architecture, fine-tuned on a dataset of brain MRI scans. The model can detect and classify:
        
        - Pituitary tumor
        - Glioma
        - Meningioma
        - No tumor
        
        **Technical Specifications:**
        - Input size: 128×128 pixels
        - Base: Pre-trained VGG16
        - Fine-tuned convolutional layers
        - Customized fully connected layers
        """)
        
        st.markdown("## 📊 Dataset")
        st.markdown("""
        The model was trained on a curated dataset of brain MRI scans from multiple medical institutions, with expert radiologist annotations.
        
        **Dataset Distribution:**
        - Total images: ~7000
        - Balanced class representation
        - Multiple MRI machines/protocols
        """)
        
        st.markdown("## 📚 References")
        st.markdown("""
        [Measurement: Sensors](https://www.sciencedirect.com/science/article/pii/S2665917424000023)
        """)
                # 2. [Brain Tumor: Classification, Detection and Segmentation](https://arxiv.org)
        # 3. [Deep Learning for Medical Image Analysis](https://www.sciencedirect.com)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Custom CSS styling with more advanced effects
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');
    
    :root {
        --primary-color: #1E88E5;
        --primary-dark: #1565C0;
        --primary-light: #64B5F6;
        --secondary-color: #7E57C2;
        --success-color: #2E7D32;
        --warning-color: #FF8F00;
        --danger-color: #C62828;
        --light-color: #f8f9fa;
        --dark-color: #263238;
        --gradient-blue: linear-gradient(135deg, #42A5F5, #1976D2);
        --gradient-purple: linear-gradient(135deg, #9575CD, #5E35B1);
    }
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated gradient background for header */
    .main-header {
        # position: fixed;
        top: 50px;
        left: 0;
        width: 100%;
        z-index: 9999;
        padding: 1.5rem 1rem;
        margin: 0;
        border-radius: 0;
        background: linear-gradient(-45deg, #1E88E5, #5E35B1, #1565C0, #7E57C2);
        background-size: 400% 400%;
        color: white;
        animation: gradientBG 15s ease infinite;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: var(--primary-dark);
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        background-color: rgba(30, 136, 229, 0.1);
        border-left: 5px solid var(--primary-color);
        display: flex;
        align-items: center;
    }
    
    .sub-header svg {
        margin-right: 10px;
    }
    
    .card {
        background-color: #ffffff;
        border-radius: 0.8rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 5px solid var(--primary-color);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.12);
    }
    
    .card-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid #eee;
        color: var(--primary-dark);
    }
    
    .info-box {
        background-color: var(--light-color);
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        background-color: #e9f3fd;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    
    .dashboard-card {
        background-color: #ffffff;
        border-radius: 0.8rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-top: 4px solid var(--primary-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-3px);
    }
    
    .dashboard-card h3 {
        color: var(--primary-dark);
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.4em 0.8em;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 2rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.12);
    }
    
    .badge-info {
        background-color: #E3F2FD;
        color: #1976D2;
        border: 1px solid #BBDEFB;
    }
    
    .badge-warning {
        background-color: #FFF8E1;
        color: #FF8F00;
        border: 1px solid #FFE082;
    }
    
    .badge-danger {
        background-color: #FFEBEE;
        color: #C62828;
        border: 1px solid #FFCDD2;
    }
    
    .badge-success {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #C8E6C9;
    }
    
    /* Specialized result headers */
    .result-header-normal {
        color: var(--success-color);
        font-size: 1.8rem;
        font-weight: 700;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        background-color: rgba(46, 125, 50, 0.1);
        border-radius: 0.5rem;
        border-left: 5px solid var(--success-color);
        text-align: center;
    }
    
    .result-header-tumor {
        color: var(--danger-color);
        font-size: 1.8rem;
        font-weight: 700;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        background-color: rgba(198, 40, 40, 0.1);
        border-radius: 0.5rem;
        border-left: 5px solid var(--danger-color);
        text-align: center;
    }
    
    .highlight-text {
        font-weight: 600;
        color: var(--primary-dark);
    }
    
    /* Modern button styling */
    .stButton>button {
        font-weight: 600 !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.15) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-top: 3px solid var(--primary-color) !important;
        border-right: 1px solid #e0e0e0 !important;
        border-left: 1px solid #e0e0e0 !important;
    }
    
    /* Footer with visual enhancements */
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        background-color: var(--light-color);
        border-top: 1px solid #e0e0e0;
        text-align: center;
        border-radius: 0.5rem;
    }
    
    .footer-logo {
        max-width: 80px;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    
    .footer-text {
        font-size: 0.9rem;
        color: #555;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .footer-copyright {
        font-size: 0.8rem;
        color: #777;
        margin-top: 1rem;
    }
    
    /* Empty state container */
    .empty-state {
        height: 500px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        background-color: white;
        border-radius: 0.8rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .empty-state img {
        margin-bottom: 1.5rem;
        max-width: 120px;
        opacity: 0.7;
    }
    
    .empty-state h3 {
        color: #555;
        margin-bottom: 1rem;
    }
    
    .empty-state p {
        color: #888;
        max-width: 80%;
    }
    
    /* Table styling */
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        min-width: 400px;
        border-radius: 5px 5px 0 0;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    
    .styled-table thead tr {
        background-color: var(--primary-color);
        color: #ffffff;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Sidebar styling with depth */
    .css-1544g2n {
        padding: 2rem 1rem;
        background-color: #f8f9fa;
        border-right: 1px solid #eaeaea;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .sidebar-heading {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary-dark);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-light);
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Enhanced dataframe styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        border: none !important;
    }
    
    .dataframe th {
        background-color: var(--primary-color);
        color: white;
        padding: 10px;
        text-align: left;
        font-weight: 500;
    }
    
    .dataframe td {
        padding: 8px 10px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .dataframe tr:hover {
        background-color: #f5f5f5;
    }
                
     /* Subtabs styling for nested tabs */
    .stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: #f0f2f6;
        padding: 0.5rem 0.5rem 0 0.5rem;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-size: 0.9rem;
        font-weight: 500;
        background-color: #e9ecef;
        margin-right: 2px;
    }

    .stTabs [data-baseweb="tab-panel"] .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-top: 3px solid var(--primary-color) !important;
        font-weight: 600;
    }

    /* Add depth to nested tab panels */
    .stTabs [data-baseweb="tab-panel"] .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0 0 8px 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }           
    </style>
    """, unsafe_allow_html=True)
    
    # App header with logo and title
    st.markdown('<div class="main-header">NeuroScan: Brain MRI Tumor Detection</div>', unsafe_allow_html=True)
    # Add vertical space below the fixed header
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    # Create three main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 MRI Analysis", "🧠 About Predicted Tumors", "📊 About Model"])
    
    # --------------------- Tab 1: MRI Analysis ---------------------
    with tab1:
        col1, separator, col2 = st.columns([1, 0.05, 1])
        
        with col1:
            st.markdown('<div class="sub-header">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <p>Upload a brain MRI image to detect and classify potential tumors. 
            The system analyzes the scan and outputs predictions with confidence scores.</p>
            <p><strong>For best results</strong>, please use T1-weighted axial MRI scans.</p>
            </div>
            """, unsafe_allow_html=True)
            
            model = load_tumor_model()
            if model is None:
                st.error("Failed to load model. Check if the model file exists in the 'models' directory.")
                return
            
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Choose a Brain MRI image</div>', unsafe_allow_html=True)
            upload_col, sample_col = st.columns([3, 1])
            with upload_col:
                uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png", "webp"])

            if uploaded_file is not None:
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image = Image.open(file_path)
                st.image(image, caption='Uploaded MRI Image', width=550)
            else:
                file_path = None
            
            if file_path:
                analyze_button = st.button("🔍 Analyze MRI Scan", use_container_width=True, type="primary")
            else:
                analyze_button = False
            st.markdown('</div>', unsafe_allow_html=True)


        with separator:
            st.markdown(
                """
                <div style="height: 100%; border-left: 1px solid #bbb;"></div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown('<div class="sub-header">📋 Analysis Results</div>', unsafe_allow_html=True)
            if file_path is not None and analyze_button:
                with st.spinner("Processing MRI scan..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    result, confidence, class_name, all_scores = predict_tumor(file_path, model)
                    # st.markdown('<div class="card">', unsafe_allow_html=True)
                    if class_name == 'notumor':
                        st.markdown(f'<div class="result-header-normal">{result}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-header-tumor">{result}</div>', unsafe_allow_html=True)
                    
                    col_gauge, col_chart = st.columns([1, 2])
                    with col_gauge:
                        st.markdown("""<h3 style="white-space: nowrap; margin: 0;">Confidence Level</h3>""", unsafe_allow_html=True)
                        st.markdown("<div style='margin-top: -35px'></div>", unsafe_allow_html=True)
                        radial_fig = create_radial_progress(confidence, class_name)
                        st.pyplot(radial_fig)

                    with col_chart:
                        # st.markdown("### Prediction Breakdown")
                        st.markdown("""<h3 style="margin-left: 50px;"> Prediction Breakdown</h3>""", unsafe_allow_html=True)
                        st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)
                        confidence_fig = create_confidence_bars(all_scores)
                        st.pyplot(confidence_fig)
                    
                    if class_name != 'notumor' and class_name != 'error':
                        st.markdown("### 🔬 Tumor Analysis")
                        if class_name == 'pituitary':
                            st.markdown("""
                            <div class="info-box">
                            <h4>Pituitary Tumor Detected</h4>
                            <div style="margin-bottom:10px;">
                            <span class="badge badge-danger">High Confidence</span>
                            <span class="badge badge-info">Common</span>
                            <span class="badge badge-warning">Requires Attention</span>
                            </div>
                            <p><span class="highlight-text">Description:</span> Tumors forming in the pituitary gland affect hormone regulation.</p>
                            <p><span class="highlight-text">Key MRI Characteristics:</span></p>
                            <ul>
                                <li>Well-defined sellar/suprasellar mass</li>
                                <li>Homogeneous enhancement with contrast</li>
                                <li>Possible cystic components</li>
                                <li>Potential compression of optic chiasm</li>
                            </ul>
                            <p><span class="highlight-text">Common Symptoms:</span></p>
                            <ul>
                                <li>Vision problems (due to optic chiasm compression)</li>
                                <li>Hormonal imbalances (e.g., fatigue, weight changes)</li>
                                <li>Headaches</li>
                                <li>Menstrual irregularities or sexual dysfunction</li>
                            </ul>
                            <p><span class="highlight-text">Follow-up:</span> Consultation with endocrinologist and neurosurgeon is advised.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif class_name == 'glioma':
                            st.markdown("""
                            <div class="info-box">
                            <h4>Glioma Tumor Detected</h4>
                            <div style="margin-bottom:10px;">
                            <span class="badge badge-danger">Urgent</span>
                            <span class="badge badge-info">Primary Brain Tumor</span>
                            <span class="badge badge-warning">Requires Immediate Attention</span>
                            </div>
                            <p><span class="highlight-text">Description:</span> Tumors from glial cells, ranging from low- to high-grade.</p>
                            <p><span class="highlight-text">Key MRI Characteristics:</span></p>
                            <ul>
                                <li>Infiltrative growth with ill-defined borders</li>
                                <li>Variable signal intensity</li>
                                <li>Associated with edema</li>
                                <li>Potential mass effect and midline shift</li>
                            </ul>
                            <p><span class="highlight-text">Common Symptoms:</span></p>
                            <ul>
                                <li>Seizures</li>
                                <li>Headaches</li>
                                <li>Cognitive or personality changes</li>
                                <li>Weakness or numbness on one side</li>
                                <li>Speech difficulties</li>
                            </ul>
                            <p><span class="highlight-text">Follow-up:</span> Immediate neurosurgical evaluation is recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif class_name == 'meningioma':
                            st.markdown("""
                            <div class="info-box">
                            <h4>Meningioma Tumor Detected</h4>
                            <div style="margin-bottom:10px;">
                            <span class="badge badge-warning">Attention Required</span>
                            <span class="badge badge-info">Generally Benign</span>
                            <span class="badge badge-success">Often Slow-Growing</span>
                            </div>
                            <p><span class="highlight-text">Description:</span> Tumors arising from the meninges with extra-axial origin.</p>
                            <p><span class="highlight-text">Key MRI Characteristics:</span></p>
                            <ul>
                                <li>Extra-axial location</li>
                                <li>Well-circumscribed with broad dural attachment</li>
                                <li>"Dural tail" sign</li>
                                <li>Homogeneous enhancement with contrast</li>
                            </ul>
                            <p><span class="highlight-text">Common Symptoms:</span></p>
                            <ul>
                                <li>Headaches</li>
                                <li>Vision or hearing changes</li>
                                <li>Memory issues</li>
                                <li>Seizures</li>
                                <li>Weakness in limbs</li>
                            </ul>
                            <p><span class="highlight-text">Follow-up:</span> Neurosurgical consultation is recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    elif class_name == 'notumor':
                        st.markdown("""
                        <div class="info-box">
                        <h4>Normal Brain Scan Assessment</h4>
                        <div style="margin-bottom:10px;">
                        <span class="badge badge-success">No Tumor Detected</span>
                        <span class="badge badge-info">Normal Findings</span>
                        </div>
                        <p>The analysis indicates a normal brain MRI with no evidence of tumor.</p>
                        <p><span class="highlight-text">Recommendation:</span> Further clinical correlation may be advised if symptoms persist.</p>                                  
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="empty-state">
                    <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f50d.svg" alt="Analysis" width="100">
                    <h3>Upload and analyze an MRI scan to see results</h3>
                    <p>Select a file using the uploader on the left, then click 'Analyze MRI Scan'</p>
                </div>
                """, unsafe_allow_html=True)
    
    # --------------------- Tab 2: About Predicted Tumors ---------------------
    with tab2:
        st.markdown('<div class="sub-header">🧠 About Brain Tumors</div>', unsafe_allow_html=True)
        # st.markdown("""
        # <div class="info-box">
        # <p>The NeuroScan AI system identifies three primary types of brain tumors along with normal brain scans. 
        # Each type has distinct characteristics, origins, and treatment approaches. Select an option below to learn more.</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        # Create subtabs within Tab 2
        tumor_subtab1, tumor_subtab2 = st.tabs(["🧠 Brain Tumor Types", "🔬 Comparative Analysis"])
        
        # Content for first subtab - Brain Tumor Types
        with tumor_subtab1:
            # Create cards for each tumor type
            tumor_col1, tumor_col2 = st.columns(2)
            
            with tumor_col1:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>🔴 Glioma</h3>
                    <div style="margin-bottom:0.1px;">
                        <span class="badge badge-danger">Common</span>
                        <span class="badge badge-warning">Often Aggressive</span>
                        <span class="badge badge-info">Primary Brain Tumor</span>
                    </div>
                    <p><strong>Origin:</strong> Develops from glial cells that surround and support neurons in the brain.</p>
                    <p><strong>Characteristics:</strong></p>
                    <ul>
                        <li>Typically shows infiltrative growth patterns</li>
                        <li>Can range from low-grade (slow-growing) to high-grade (aggressive)</li>
                        <li>Often presents with significant surrounding edema</li>
                        <li>May cross the midline through the corpus callosum</li>
                    </ul>
                    <p><strong>MRI Appearance:</strong> Variable signal intensity, irregular margins, surrounding edema, potential enhancement with contrast.</p>
                    <p><strong>Treatment Approach:</strong> Usually requires a multidisciplinary approach including surgery, radiation therapy, and chemotherapy depending on the grade.</p>
                    <a href="https://en.wikipedia.org/wiki/Glioma" target="_blank">
                        <button style="margin-top:10px; padding:6px 16px; background-color:#E53935; color:white; border:none; border-radius:5px; font-weight:600;">
                            🔗 Read More on Wikipedia
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)

                
                st.markdown("""
                <div class="dashboard-card">
                    <h3>🟢 No Tumor (Normal)</h3>
                    <div style="margin-bottom:46px;">
                        <span class="badge badge-success">Healthy</span>
                        <span class="badge badge-info">Reference</span>
                    </div>
                    <p><strong>Characteristics:</strong></p>
                    <ul>
                        <li>Normal brain anatomy without mass effect</li>
                        <li>Clear differentiation between gray and white matter</li>
                        <li>Symmetrical brain structures</li>
                        <li>No abnormal enhancement with contrast</li>
                        <li>Normal ventricular system size and shape</li>
                    </ul>
                    <p><strong>MRI Appearance:</strong> Regular tissue boundaries, consistent signal intensity patterns, and normal anatomical structures.</p>
                    <p><strong>Clinical Importance:</strong> Identifying normal brain scans is crucial to avoid unnecessary treatments and provide patient reassurance.</p>
                    <a href="https://en.wikipedia.org/wiki/Human_brain" target="_blank">
                        <button style="margin-top:10px; padding:6px 16px; background-color:#43A047; color:white; border:none; border-radius:5px; font-weight:600;">
                            🔗 Read More about Brain Anatomy
                        </button>
                    </a>
                    </div>
                """, unsafe_allow_html=True)
            
            with tumor_col2:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>🟣 Meningioma</h3>
                    <div style="margin-bottom:22px;">
                        <span class="badge badge-info">Common</span>
                        <span class="badge badge-success">Usually Benign</span>
                        <span class="badge badge-warning">Extra-axial</span>
                    </div>
                    <p><strong>Origin:</strong> Arises from the meninges, the protective coverings of the brain and spinal cord.</p>
                    <p><strong>Characteristics:</strong></p>
                    <ul>
                        <li>Well-circumscribed masses attached to the dura mater</li>
                        <li>May cause compression of adjacent brain tissue</li>
                        <li>Typically slow-growing</li>
                        <li>Often benign (WHO Grade I), but atypical and malignant variants exist</li>
                    </ul>
                    <p><strong>MRI Appearance:</strong> Extra-axial location, well-defined borders, homogeneous enhancement with contrast, characteristic "dural tail" sign.</p>
                    <p><strong>Treatment Approach:</strong> Surgical resection for symptomatic or growing tumors; stereotactic radiosurgery may be an alternative.</p>
                    <a href="https://en.wikipedia.org/wiki/Meningioma" target="_blank">
                        <button style="margin-top:10px; padding:6px 16px; background-color:#6A1B9A; color:white; border:none; border-radius:5px; font-weight:600;">
                            🔗 Read More on Wikipedia
                        </button>
                    </a>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="dashboard-card">
                    <h3>🔵 Pituitary Tumor</h3>
                    <div style="margin-bottom:10px;">
                        <span class="badge badge-info">Endocrine</span>
                        <span class="badge badge-warning">Sellar Location</span>
                        <span class="badge badge-success">Often Treatable</span>
                    </div>
                    <p><strong>Origin:</strong> Develops from cells in the pituitary gland, which controls many hormonal functions.</p>
                    <p><strong>Characteristics:</strong></p>
                    <ul>
                        <li>Located in the pituitary fossa/sella turcica</li>
                        <li>May be functional (hormone-producing) or non-functional</li>
                        <li>Can compress adjacent structures like optic chiasm</li>
                        <li>Microadenomas (<10mm) or macroadenomas (>10mm)</li>
                    </ul>
                    <p><strong>MRI Appearance:</strong> Well-defined sellar mass, often with suprasellar extension, homogeneous enhancement with contrast.</p>
                    <p><strong>Treatment Approach:</strong> Treatment may include surgical removal, medication for hormone control, or radiation therapy.</p>
                    <a href="https://en.wikipedia.org/wiki/Pituitary_adenoma" target="_blank">
                        <button style="margin-top:10px; padding:6px 16px; background-color:#1E88E5; color:white; border:none; border-radius:5px; font-weight:600;">
                            🔗 Read More on Wikipedia
                        </button>
                    </a>
                    </div>
                """, unsafe_allow_html=True)
        
        # Content for second subtab - Comparative Analysis
        with tumor_subtab2:
            st.markdown("""
            <div class="card">
                <div class="card-header">Differentiating Features of Brain Tumors</div>
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Glioma</th>
                            <th>Meningioma</th>
                            <th>Pituitary</th>
                            <th>No Tumor</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Location</strong></td>
                            <td>Intra-axial (within brain parenchyma)</td>
                            <td>Extra-axial (outside brain tissue)</td>
                            <td>Sellar/suprasellar region</td>
                            <td>N/A</td>
                        </tr>
                        <tr>
                            <td><strong>Borders</strong></td>
                            <td>Infiltrative, poorly defined</td>
                            <td>Well-defined, distinct</td>
                            <td>Well-circumscribed</td>
                            <td>Normal tissue boundaries</td>
                        </tr>
                        <tr>
                            <td><strong>Enhancement</strong></td>
                            <td>Variable, often heterogeneous</td>
                            <td>Strong, homogeneous</td>
                            <td>Homogeneous</td>
                            <td>No abnormal enhancement</td>
                        </tr>
                        <tr>
                            <td><strong>Edema</strong></td>
                            <td>Often extensive</td>
                            <td>Variable, may be minimal</td>
                            <td>Minimal or none</td>
                            <td>None</td>
                        </tr>
                        <tr>
                            <td><strong>Growth Rate</strong></td>
                            <td>Variable (grade-dependent)</td>
                            <td>Typically slow</td>
                            <td>Usually slow</td>
                            <td>N/A</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # --------------------- Tab 3: About Model ---------------------
    with tab3:
        st.markdown('<div class="sub-header">📊 Model Architecture & Performance</div>', unsafe_allow_html=True)
        # st.markdown("""
        # <div class="info-box">
        # <p>NeuroScan AI is powered by a deep learning model based on the VGG16 architecture with transfer learning. 
        # The model is trained on a dataset of brain MRI images and optimized for tumor classification.</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        # Create subtabs within Tab 3
        model_subtab1, model_subtab2, model_subtab3, model_subtab4 = st.tabs([
            "📐 Model Summary", 
            "📈 Model Performance Metrics", 
            "📉 Performance Visualizations", 
            "⚙️ Technical Implementation Details"
        ])
        
        # Content for first subtab - Model Summary
        with model_subtab1:
            # ROW 1: Dataset Distribution + Chart Side by Side
            dataset_col, chart_col = st.columns([1, 1])

            with dataset_col:
                st.markdown("""
                <div class="dashboard-card" style="margin-top: 10px;">
                    <h3>🧾 Dataset Distribution</h3>
                    <div style="font-size: 14px; line-height: 1.6;">
                        <strong>Training Set:</strong><br/>
                        • Pituitary: 1,457<br/>
                        • Glioma: 1,321<br/>
                        • No Tumor: 1,463<br/>
                        • Meningioma: 1,339<br/><br/>
                        <strong>Test Set:</strong><br/>
                        • Pituitary: 300<br/>
                        • Glioma: 300<br/>
                        • No Tumor: 405<br/>
                        • Meningioma: 306<br/><br/>
                        <strong>Total Images:</strong> ~7,000
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with chart_col:
                st.image("Training and Test DataSet Distribution.png", caption="Training & Test Dataset Distribution", use_container_width=True)

            # ROW 2: Model Summary below both
            st.markdown("""
            <div class="dashboard-card" style="margin-top: 20px;">
                <h3><code>sequential</code></h3>
                <div style="overflow-x: auto; width: 100%;">        
                    <table class="styled-table" style="width: 100%; min-width: 800px; border-collapse: collapse;">
                        <thead>
                            <tr>
                                <th>Layer (type)</th>
                                <th>Output Shape</th>
                                <th>Param #</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td>vgg16 (Functional)</td><td>(None, 4, 4, 512)</td><td>14,714,688</td></tr>
                            <tr><td>flatten (Flatten)</td><td>(None, 8192)</td><td>0</td></tr>
                            <tr><td>dropout (Dropout)</td><td>(None, 8192)</td><td>0</td></tr>
                            <tr><td>dense (Dense)</td><td>(None, 128)</td><td>1,048,704</td></tr>
                            <tr><td>dropout_1 (Dropout)</td><td>(None, 128)</td><td>0</td></tr>
                            <tr><td>dense_1 (Dense)</td><td>(None, 4)</td><td>516</td></tr>
                        </tbody>
                    </table>
                    <br/>
                    <div style="font-size: 14px; line-height: 1.6;">
                        <strong>Total params:</strong> 32,021,198 (122.15 MB)<br/>
                        <strong>Trainable params:</strong> 8,128,644 (31.01 MB)<br/>
                        <strong>Non-trainable params:</strong> 7,635,264 (29.13 MB)<br/>
                        <strong>Optimizer params:</strong> 16,257,290 (62.02 MB)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Content for second subtab - Model Performance Metrics
        with model_subtab2:
            metric_col1, metric_col2 = st.columns([1.5, 1.5])
            
            with metric_col1:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>Classification Report</h3>
                    <div style="overflow-x: auto;">
                        <table class="styled-table">
                            <thead>
                                <tr>
                                    <th>Class</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-score</th>
                                    <th>Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Pituitary (0)</td>
                                    <td>1.00</td>
                                    <td>0.85</td>
                                    <td>0.92</td>
                                    <td>300</td>
                                </tr>
                                <tr>
                                    <td>Glioma (1)</td>
                                    <td>0.98</td>
                                    <td>0.89</td>
                                    <td>0.93</td>
                                    <td>300</td>
                                </tr>
                                <tr>
                                    <td>No Tumor (2)</td>
                                    <td>0.93</td>
                                    <td>1.00</td>
                                    <td>0.96</td>
                                    <td>405</td>
                                </tr>
                                <tr>
                                    <td>Meningioma (3)</td>
                                    <td>0.83</td>
                                    <td>0.95</td>
                                    <td>0.88</td>
                                    <td>306</td>
                                </tr>
                            </tbody>
                            <tfoot>
                                <tr>
                                    <td>Accuracy</td>
                                    <td colspan="2"></td>
                                    <td>0.93</td>
                                    <td>1311</td>
                                </tr>
                                <tr>
                                    <td>Macro avg</td>
                                    <td>0.93</td>
                                    <td>0.92</td>
                                    <td>0.92</td>
                                    <td>1311</td>
                                </tr>
                                <tr>
                                    <td>Weighted avg</td>
                                    <td>0.93</td>
                                    <td>0.93</td>
                                    <td>0.93</td>
                                    <td>1311</td>
                                </tr>
                            </tfoot>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown("""
                <div class="dashboard-card" style="text-align: center; display: flex; flex-direction: column; align-items: center;">
                    <h3>Key Performance Stats</h3>
                    <div style="margin-bottom: 115px;">
                        <div style="display: flex; justify-content: center; align-items: center; background: var(--gradient-blue); border-radius: 100%; width: 150px; height: 150px; color: white; font-size: 54px; font-weight: bold; margin: 0.5rem; line-height: 200px;">
                            93%
                        </div>
                        <p style="font-weight: 600; color: var(--primary-dark);">Overall Accuracy</p>
                    </div>
                    <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
                        <div style="text-align: center; margin: 0.5rem;">
                            <div style="background-color: #E8F5E9; color: #2E7D32; padding: 0.5rem; border-radius: 0.5rem; font-weight: bold;">
                                0.96
                            </div>
                            <p style="font-size: 0.8rem; margin-top: 0.2rem;">Best F1 (No Tumor)</p>
                        </div>
                        <div style="text-align: center; margin: 0.5rem;">
                            <div style="background-color: #E3F2FD; color: #1976D2; padding: 0.5rem; border-radius: 0.5rem; font-weight: bold;">
                                1.00
                            </div>
                            <p style="font-size: 0.8rem; margin-top: 0.2rem;">Best Precision</p>
                        </div>
                        <div style="text-align: center; margin: 0.5rem;">
                            <div style="background-color: #FFF8E1; color: #FF8F00; padding: 0.5rem; border-radius: 0.5rem; font-weight: bold;">
                                1.00
                            </div>
                            <p style="font-size: 0.8rem; margin-top: 0.2rem;">Best Recall</p>
                        </div>
                        <div style="text-align: center; margin: 0.5rem;">
                            <div style="background-color: #FFEBEE; color: #C62828; padding: 0.5rem; border-radius: 0.5rem; font-weight: bold;">
                                0.88
                            </div>
                            <p style="font-size: 0.8rem; margin-top: 0.2rem;">Lowest F1</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Content for third subtab - Performance Visualizations
        with model_subtab3:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>Confusion Matrix</h3>
                    <p>Visualizes model predictions versus actual classes.</p>
                """, unsafe_allow_html=True)
                
                # Use placeholder image until the real one is available
                st.image("confusion_matrix.png", use_container_width=True)
                
                st.markdown("""
                    <div style="font-size: 0.9rem; color: #555; margin-top: 0.5rem;">
                        <p>The confusion matrix shows strong diagonal values, indicating good model performance across all classes. Notice how "No Tumor" (Class 2) has perfect recall with all cases correctly identified.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with viz_col2:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>ROC Curve</h3>
                    <p>Shows the diagnostic ability of the classifier system.</p>
                """, unsafe_allow_html=True)
                
                # Use placeholder image until the real one is available
                st.image("roc_curve.png", use_container_width=True)
                
                st.markdown("""
                    <div style="font-size: 0.9rem; color: #555; margin-top: 0.5rem;">
                        <p>The ROC curves demonstrate excellent performance across all tumor classes with AUC values above 0.9, indicating strong discrimination ability of the model.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="dashboard-card">
                <h3>Training History</h3>
                <p>Model training and validation performance over epochs.</p>
            """, unsafe_allow_html=True)
            
            # Use placeholder image until the real one is available
            st.image("training_vs_validation_curves.png", use_container_width=True)
            
            st.markdown("""
                <div style="font-size: 0.9rem; color: #555; margin-top: 0.5rem;">
                    <p>The training curves show progressive improvement in both accuracy and loss metrics. The close alignment between training and validation curves indicates good generalization without overfitting.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Content for fourth subtab - Technical Implementation Details
        with model_subtab4:
            tech_col1, tech_col2 = st.columns([1, 1])
            
            with tech_col1:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>Training Configuration</h3>
                    <ul>
                        <li><strong>Framework:</strong> TensorFlow/Keras</li>
                        <li><strong>Base Model:</strong> VGG16 (pre-trained on ImageNet)</li>
                        <li><strong>Transfer Learning:</strong> Feature extraction with fine-tuning of last few layers</li>
                        <li><strong>Input Size:</strong> 128×128×3 pixels</li>
                        <li><strong>Batch Size:</strong> 20</li>
                        <li><strong>Optimizer:</strong> Adam (Learning rate: 0.0001)</li>
                        <li><strong>Loss Function:</strong> Sparse Categorical Crossentropy</li>
                        <li><strong>Epochs:</strong> 5</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tech_col2:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>Data Processing</h3>
                    <ul style="margin-bottom:75px;">
                        <li><strong>Preprocessing:</strong> Resizing, normalization (0-1 range)</li>
                        <li><strong>Augmentation:</strong> Random brightness and contrast adjustments</li>
                        <li><strong>Class Balance:</strong> Relatively balanced dataset across 4 classes</li>
                        <li><strong>Training Samples:</strong> 5580</li>
                        <li><strong>Testing Samples:</strong> 1311</li>
                        <li><strong>Class Labels:</strong> Pituitary, Glioma, No tumor, Meningioma</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Footer section
        st.markdown("""
        <div class="footer" style="text-align: center; margin-top: 20px;">
            <p class="footer-text" style="margin: 0; padding: 0; display: inline;">NeuroScan AI is a demonstration project showcasing the application of deep learning for medical image analysis. This tool is for educational purposes only and should not replace professional medical diagnosis.</p>
            <br>
            <p class="footer-copyright" style="margin: 0; padding: 0;">© 2025 NeuroScan - Brain MRI Analysis System</p>
        </div>
        """, unsafe_allow_html=True)
        # <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f9e0.svg" alt="NeuroScan AI" class="footer-logo">
if __name__ == "__main__":
    main()