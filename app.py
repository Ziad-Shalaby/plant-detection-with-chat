import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import google.generativeai as genai
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
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Gemini will be configured in the chat function

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
# Gemini Chat Functions
# ----------------------------------
def chat_with_gemini(user_message, context=None):
    """
    Chat with Gemini AI about plants
    """
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Please add GEMINI_API_KEY to .streamlit/secrets.toml"
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try different model names that are currently supported
        model_names = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest'
        ]
        
        # Build context-aware prompt
        if context:
            system_prompt = f"""You are a helpful plant expert assistant. 
            
Current plant context:
- Plant: {context.get('plant_name', 'Unknown')}
- Scientific Name: {context.get('scientific_name', 'N/A')}
- Health Status: {context.get('health_status', 'N/A')}
- Disease: {context.get('disease', 'None detected')}

Provide helpful, accurate advice about plant care, diseases, and gardening. Be friendly and conversational."""
            
            full_prompt = f"{system_prompt}\n\nUser question: {user_message}"
        else:
            full_prompt = f"""You are a helpful plant expert assistant. Provide accurate advice about plant care, identification, diseases, and gardening. Be friendly and conversational.

User question: {user_message}"""
        
        # Try each model until one works
        last_error = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                last_error = str(e)
                continue
        
        # If all models fail
        return f"⚠️ Could not connect to Gemini API. All models failed.\n\nPlease ensure:\n1. Your API key is correct\n2. The Generative Language API is enabled\n3. Wait a few minutes if you just enabled it\n\nLast error: {last_error}"
        
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease verify your Gemini API key at: https://aistudio.google.com/app/apikey"

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
    if not PLANT_ID_API_KEY or not GEMINI_API_KEY:
        st.warning("""
        ⚠️ **API Keys Required**
        
        To use this app, you need to configure API keys:
        1. Get Plant.id API key from https://web.plant.id/
        2. Get Gemini API key from https://makersuite.google.com/app/apikey
        3. Create `.streamlit/secrets.toml` file with:
        ```
        PLANT_ID_API_KEY = "your_plant_id_key"
        GEMINI_API_KEY = "your_gemini_key"
        ```
        
        **Free tiers available for both services!**
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
                            top_match = suggestions[0]
                            plant_name = top_match.get("plant_name", "Unknown")
                            probability = top_match.get("probability", 0) * 100
                            
                            plant_details = top_match.get("plant_details", {})
                            common_names = plant_details.get("common_names", [])
                            taxonomy = plant_details.get("taxonomy", {})
                            scientific_name = taxonomy.get("genus", "") + " " + taxonomy.get("species", "")
                            
                            # Display results
                            st.markdown(f"""
                            <div class="result-card">
                                <h2>🌿 {plant_name}</h2>
                                <p><strong>Scientific Name:</strong> {scientific_name}</p>
                                <p><strong>Common Names:</strong> {', '.join(common_names[:3]) if common_names else 'N/A'}</p>
                                <p><strong>Confidence:</strong> {probability:.1f}%</p>
                                <p><strong>Family:</strong> {taxonomy.get('family', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Save to context
                            st.session_state.plant_context = {
                                'plant_name': plant_name,
                                'scientific_name': scientific_name,
                                'common_names': common_names,
                                'confidence': probability
                            }
                            
                            # Save to history
                            st.session_state.detection_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'plant_name': plant_name,
                                'type': 'identification'
                            })
                            
                            st.success("✅ Plant identified! You can now chat with AI about this plant.")
                        else:
                            st.warning("No plant identified. Please try a clearer image.")
            
            if health_btn:
                with st.spinner("🏥 Analyzing plant health..."):
                    result = check_health(image)
                    
                    if result.get("error"):
                        st.error(result.get("message"))
                    else:
                        data = result.get("data", {})
                        
                        # Safely get health status
                        is_healthy_data = data.get("is_healthy", {})
                        if isinstance(is_healthy_data, dict):
                            is_healthy = is_healthy_data.get("binary", False)
                        else:
                            is_healthy = bool(is_healthy_data)
                        
                        # Safely get plant status
                        is_plant_data = data.get("is_plant", {})
                        if isinstance(is_plant_data, dict):
                            is_plant = is_plant_data.get("binary", True)
                        else:
                            is_plant = bool(is_plant_data) if is_plant_data is not None else True
                        
                        if not is_plant:
                            st.warning("⚠️ This doesn't appear to be a plant image.")
                        elif is_healthy:
                            st.markdown("""
                            <div class="healthy-card">
                                <h2>✅ Healthy Plant!</h2>
                                <p>Your plant appears to be in good health with no diseases detected.</p>
                                <p><strong>Health Score:</strong> Excellent</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                            
                            # Update context
                            if st.session_state.plant_context:
                                st.session_state.plant_context['health_status'] = 'Healthy'
                                st.session_state.plant_context['disease'] = 'None'
                        else:
                            suggestions = data.get("suggestions", [])
                            if suggestions:
                                disease = suggestions[0]
                                disease_name = disease.get("name", "Unknown Disease")
                                probability = disease.get("probability", 0) * 100
                                
                                disease_details = disease.get("disease_details", {})
                                description = disease_details.get("description", "No description available")
                                treatment = disease_details.get("treatment", {})
                                
                                st.markdown(f"""
                                <div class="disease-card">
                                    <h2>⚠️ Disease Detected</h2>
                                    <h3>{disease_name}</h3>
                                    <p><strong>Confidence:</strong> {probability:.1f}%</p>
                                    <p><strong>Description:</strong> {description[:200]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if treatment:
                                    st.markdown("### 💊 Treatment Recommendations")
                                    biological = treatment.get("biological", [])
                                    chemical = treatment.get("chemical", [])
                                    prevention = treatment.get("prevention", [])
                                    
                                    if biological:
                                        st.write("**Biological Treatment:**")
                                        for treat in biological[:3]:
                                            st.write(f"- {treat}")
                                    
                                    if chemical:
                                        st.write("**Chemical Treatment:**")
                                        for treat in chemical[:3]:
                                            st.write(f"- {treat}")
                                    
                                    if prevention:
                                        st.write("**Prevention:**")
                                        for prev in prevention[:3]:
                                            st.write(f"- {prev}")
                                
                                # Update context
                                if st.session_state.plant_context:
                                    st.session_state.plant_context['health_status'] = 'Diseased'
                                    st.session_state.plant_context['disease'] = disease_name
                                else:
                                    st.session_state.plant_context = {
                                        'plant_name': 'Unknown',
                                        'health_status': 'Diseased',
                                        'disease': disease_name
                                    }
                                
                                st.session_state.detection_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    'plant_name': st.session_state.plant_context.get('plant_name', 'Unknown'),
                                    'type': 'disease',
                                    'disease': disease_name
                                })
        else:
            st.info("👆 Upload an image to get started")

# ----------------------------------
# AI Chat Page
# ----------------------------------
elif app_mode == "💬 AI Chat":
    st.markdown("<h1>💬 Chat with Plant Expert AI</h1>", unsafe_allow_html=True)
    
    if not GEMINI_API_KEY:
        st.error("⚠️ Gemini API key not configured. Please add it to .streamlit/secrets.toml")
        st.stop()
    
    # Display context if available
    if st.session_state.plant_context:
        with st.expander("📌 Current Plant Context", expanded=True):
            context = st.session_state.plant_context
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Plant:** {context.get('plant_name', 'Unknown')}")
                st.write(f"**Scientific Name:** {context.get('scientific_name', 'N/A')}")
            with col2:
                st.write(f"**Health Status:** {context.get('health_status', 'N/A')}")
                st.write(f"**Disease:** {context.get('disease', 'None')}")
    else:
        st.info("💡 Identify a plant first to get context-aware advice, or ask general questions!")
    
    st.markdown("---")
    
    # Chat History
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Plant Expert:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat Input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Ask me anything about plants...", key="chat_input", placeholder="e.g., How do I treat fungal diseases?")
    
    with col2:
        send_btn = st.button("Send 📨", use_container_width=True)
    
    if send_btn and user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = chat_with_gemini(user_input, st.session_state.plant_context)
        
        # Add bot response
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        st.rerun()
    
    # Quick Questions
    if len(st.session_state.chat_history) == 0:
        st.markdown("### 💡 Quick Questions")
        quick_questions = [
            "What are common tomato diseases?",
            "How often should I water succulents?",
            "What causes yellow leaves?",
            "Best fertilizer for roses?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })
                    response = chat_with_gemini(question, st.session_state.plant_context)
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    st.rerun()
    
    # Clear Chat
    if len(st.session_state.chat_history) > 0:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# ----------------------------------
# My Plants Page
# ----------------------------------
elif app_mode == "📚 My Plants":
    st.markdown("<h1>📚 My Plant Collection</h1>", unsafe_allow_html=True)
    
    if not st.session_state.detection_history:
        st.info("🌱 You haven't identified any plants yet. Go to Plant Detection to get started!")
    else:
        st.markdown(f"### Total Identifications: {len(st.session_state.detection_history)}")
        
        for i, record in enumerate(reversed(st.session_state.detection_history)):
            with st.expander(f"🌿 {record['plant_name']} - {record['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Plant:** {record['plant_name']}")
                    st.write(f"**Type:** {record['type'].title()}")
                with col2:
                    st.write(f"**Date:** {record['timestamp']}")
                    if record.get('disease'):
                        st.write(f"**Disease:** {record['disease']}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()

