import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
from datetime import datetime

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Plant Doctor AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# Custom CSS
# ----------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d5016 0%, #1a3409 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    h1 {
        color: #2d5016;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-align: center;
        padding: 20px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(86, 171, 47, 0.6);
    }
    
    .result-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        color: #2d2d2d;
    }
    
    .result-card h2, .result-card h3 {
        color: #2d5016 !important;
    }
    
    .result-card p, .result-card strong {
        color: #333333 !important;
    }
    
    .healthy-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
    }
    
    .disease-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
    }
    
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2d2d2d;
    }
    
    .chat-message strong {
        color: #1a1a1a !important;
    }
    
    .user-message {
        background: #e3f2fd;
        margin-left: 20%;
        color: #1565c0;
    }
    
    .user-message strong {
        color: #0d47a1 !important;
    }
    
    .bot-message {
        background: #f5f5f5;
        margin-right: 20%;
        color: #2d2d2d;
    }
    
    .bot-message strong {
        color: #1a1a1a !important;
    }
    
    .feature-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s;
        color: #333333;
    }
    
    .feature-card h3 {
        color: #2d5016 !important;
    }
    
    .feature-card p {
        color: #555555 !important;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        color: #333333;
    }
    
    .stat-box div:not(.stat-number) {
        color: #555555 !important;
    }
    
    .stat-number {
        font-size: 36px;
        font-weight: bold;
        color: #56ab2f;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Initialize Session State
# ----------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'plant_context' not in st.session_state:
    st.session_state.plant_context = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# ----------------------------------
# API Configuration
# ----------------------------------
PLANT_ID_API_KEY = st.secrets.get("PLANT_ID_API_KEY", "")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")

# ----------------------------------
# Plant.id API Functions
# ----------------------------------
def identify_plant(image_data):
    """
    Identify plant using Plant.id API
    """
    if not PLANT_ID_API_KEY:
        return {
            "error": True,
            "message": "Plant.id API key not configured. Please add PLANT_ID_API_KEY to .streamlit/secrets.toml"
        }
    
    url = "https://api.plant.id/v2/identify"
    
    # Convert image to base64
    buffered = io.BytesIO()
    image_data.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    payload = {
        "images": [f"data:image/jpeg;base64,{img_str}"],
        "modifiers": ["similar_images"],
        "plant_details": ["common_names", "taxonomy", "wiki_description", "wiki_image"]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PLANT_ID_API_KEY
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return {"error": False, "data": response.json()}
        else:
            return {
                "error": True,
                "message": f"API Error: {response.status_code} - {response.text}"
            }
    except Exception as e:
        return {"error": True, "message": f"Request failed: {str(e)}"}


def check_health(image_data):
    """
    Check plant health using Plant.id health assessment API
    """
    if not PLANT_ID_API_KEY:
        return {
            "error": True,
            "message": "Plant.id API key not configured"
        }
    
    url = "https://api.plant.id/v2/health_assessment"
    
    # Convert image to base64
    buffered = io.BytesIO()
    image_data.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    payload = {
        "images": [f"data:image/jpeg;base64,{img_str}"],
        "modifiers": ["similar_images"],
        "disease_details": ["cause", "common_names", "classification", "description", "treatment"]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PLANT_ID_API_KEY
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return {"error": False, "data": response.json()}
        else:
            return {
                "error": True,
                "message": f"API Error: {response.status_code}"
            }
    except Exception as e:
        return {"error": True, "message": f"Request failed: {str(e)}"}

# ----------------------------------
# Hugging Face Chat Functions
# ----------------------------------
def chat_with_huggingface(user_message, context=None):
    """
    Chat with Hugging Face AI about plants
    """
    if not HUGGINGFACE_API_KEY:
        return "Hugging Face API key not configured. Please add HUGGINGFACE_API_KEY to .streamlit/secrets.toml"
    
    try:
        # Use a powerful open-source model
        # Options: "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf", "HuggingFaceH4/zephyr-7b-beta"
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Build context-aware prompt
        if context:
            system_prompt = f"""You are a helpful plant expert assistant. 

Current plant context:
- Plant: {context.get('plant_name', 'Unknown')}
- Scientific Name: {context.get('scientific_name', 'N/A')}
- Health Status: {context.get('health_status', 'N/A')}
- Disease: {context.get('disease', 'None detected')}

Provide helpful, accurate advice about plant care, diseases, and gardening. Be friendly and conversational.

User question: {user_message}

Answer:"""
        else:
            system_prompt = f"""You are a helpful plant expert assistant. Provide accurate advice about plant care, identification, diseases, and gardening. Be friendly and conversational.

User question: {user_message}

Answer:"""
        
        payload = {
            "inputs": system_prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', '') or result.get('output', '')
            else:
                generated_text = str(result)
            
            # Clean up the response
            generated_text = generated_text.strip()
            
            if not generated_text:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            return generated_text
            
        elif response.status_code == 503:
            return "⏳ The AI model is loading. This usually takes 20-30 seconds. Please try again in a moment."
        elif response.status_code == 401:
            return "⚠️ Invalid Hugging Face API key. Please check your configuration."
        else:
            return f"⚠️ API Error ({response.status_code}): {response.text[:200]}"
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error: {str(e)}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.markdown("# 🌿 Navigation")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox(
    "Select Page", 
    ["🏠 Home", "🔍 Plant Detection", "💬 AI Chat", "📚 My Plants"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Plants Identified", len(st.session_state.detection_history))
st.sidebar.metric("Supported Species", "10,000+")
st.sidebar.metric("Disease Database", "600+")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
- Use clear, well-lit photos
- Focus on leaves
- Avoid blurry images
- Multiple angles help
- Include disease symptoms
""")

# ----------------------------------
# Home Page
# ----------------------------------
if app_mode == "🏠 Home":
    st.markdown("<h1>🌿 Plant Doctor AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px; color: #666;'>AI-Powered Plant Identification & Health Diagnosis</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero Image Placeholder
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1466781783364-36c955e42a7f?w=800", 
                 use_column_width=True, caption="Identify plants instantly with AI")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.markdown("### ✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 48px;">🔍</div>
            <h3>Plant Identification</h3>
            <p>Identify 10,000+ plant species from a single photo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 48px;">🏥</div>
            <h3>Disease Detection</h3>
            <p>Detect 600+ plant diseases and get treatment advice</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 48px;">💬</div>
            <h3>AI Chat Expert</h3>
            <p>Ask questions and get personalized plant care advice</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Stats
    st.markdown("### 📊 Powered by Advanced AI")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">10K+</div>
            <div>Plant Species</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">600+</div>
            <div>Diseases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">95%</div>
            <div>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">⚡</div>
            <div>Instant</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Getting Started
    st.markdown("### 🚀 Getting Started")
    st.info("""
    1. Go to **Plant Detection** page
    2. Upload a clear photo of your plant
    3. Get instant identification and health assessment
    4. Chat with AI for personalized advice
    """)
    
    # API Setup Warning
    if not PLANT_ID_API_KEY or not HUGGINGFACE_API_KEY:
        st.warning("""
        ⚠️ **API Keys Required**
        
        To use this app, you need to configure API keys:
        1. Get Plant.id API key from https://web.plant.id/
        2. Get Hugging Face API key from https://huggingface.co/settings/tokens (FREE!)
        3. Create `.streamlit/secrets.toml` file with:
        ```
        PLANT_ID_API_KEY = "your_plant_id_key"
        HUGGINGFACE_API_KEY = "your_huggingface_key"
        ```
        
        **Both services have free tiers available!**
        """)

# ----------------------------------
# Plant Detection Page
# ----------------------------------
elif app_mode == "🔍 Plant Detection":
    st.markdown("<h1>🔍 Plant Detection & Health Assessment</h1>", unsafe_allow_html=True)
    
    if not PLANT_ID_API_KEY:
        st.error("⚠️ Plant.id API key not configured. Please add it to .streamlit/secrets.toml")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Plant Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="plant_upload")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                identify_btn = st.button("🔍 Identify Plant", use_container_width=True)
            with col_b:
                health_btn = st.button("🏥 Check Health", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Analysis Results")
        
        if uploaded_file:
            if identify_btn:
                with st.spinner("🔍 Identifying plant..."):
                    result = identify_plant(image)
                    
                    if result.get("error"):
                        st.error(result.get("message"))
                    else:
                        data = result.get("data", {})
                        suggestions = data.get("suggestions", [])
                        
                        if suggestions:
                            top_ma
