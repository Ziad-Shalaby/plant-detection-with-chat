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
# Plant Detection Functions
# ----------------------------------
def detect_plant_with_huggingface(image_data):
    """
    Detect plant using Hugging Face Plant Classification Models
    """
    if not HUGGINGFACE_API_KEY:
        return {
            "error": True,
            "message": "Hugging Face API key not configured"
        }
    
    try:
        # Convert image to bytes
        buffered = io.BytesIO()
        image_data.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Try specialized plant classification models
        models = [
            "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",  # Plant disease + identification
            "nateraw/plant-image-classifier",                                   # General plant classifier
            "google/vit-base-patch16-224",                                      # Vision transformer (backup)
        ]
        
        for model_name in models:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
                headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
                
                response = requests.post(
                    API_URL,
                    headers=headers,
                    data=image_bytes,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse classification results
                    if isinstance(result, list) and len(result) > 0:
                        # Get top predictions
                        predictions = sorted(result, key=lambda x: x.get('score', 0), reverse=True)[:3]
                        
                        return {
                            "error": False,
                            "data": {
                                "predictions": predictions,
                                "model": model_name,
                                "top_prediction": predictions[0] if predictions else None
                            }
                        }
                elif response.status_code == 503:
                    # Model loading, try next
                    continue
                    
            except Exception as e:
                continue
        
        return {
            "error": True,
            "message": "Could not detect plant. Models may be loading (can take 20-60 seconds). Please try again."
        }
        
    except Exception as e:
        return {"error": True, "message": f"Detection failed: {str(e)}"}


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


# ----------------------------------
# Hugging Face Chat Functions
# ----------------------------------
def chat_with_huggingface(user_message, context=None):
    """
    Chat with Hugging Face models about plants
    """
    if not HUGGINGFACE_API_KEY:
        return "Hugging Face API key not configured. Please add HUGGINGFACE_API_KEY to .streamlit/secrets.toml"
    
    try:
        # Try multiple models in order of preference (using router API)
        models = [
            "Qwen/Qwen2.5-72B-Instruct",              # Very powerful and fast
            "meta-llama/Llama-3.2-3B-Instruct",       # Efficient
            "mistralai/Mistral-7B-Instruct-v0.3",     # Reliable
            "microsoft/Phi-3.5-mini-instruct"         # Fast backup
        ]
        
        # Build context-aware prompt
        if context:
            system_prompt = f"""You are a helpful plant expert assistant. 

Current plant context:
- Plant: {context.get('plant_name', 'Unknown')}
- Scientific Name: {context.get('scientific_name', 'N/A')}

Provide helpful, accurate advice about plant care, diseases, and gardening. Be friendly and conversational."""
            
            full_prompt = f"{system_prompt}\n\nUser question: {user_message}\n\nAnswer:"
        else:
            full_prompt = f"""You are a helpful plant expert assistant. Provide accurate advice about plant care, identification, diseases, and gardening. Be friendly and conversational.

User question: {user_message}

Answer:"""
        
        # Try each model until one works
        last_error = None
        for model in models:
            try:
                # Using Hugging Face Router API (new endpoint)
                API_URL = f"https://router.huggingface.co/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "stream": False
                }
                
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle chat completion format
                    if "choices" in result and len(result["choices"]) > 0:
                        generated_text = result["choices"][0]["message"]["content"]
                    else:
                        generated_text = str(result)
                    
                    if generated_text and len(generated_text.strip()) > 10:
                        return generated_text.strip()
                    else:
                        last_error = f"Empty response from {model}"
                        continue
                        
                elif response.status_code == 503:
                    # Model is loading, try next one
                    last_error = f"Model {model} is loading"
                    continue
                else:
                    last_error = f"Error {response.status_code}: {response.text}"
                    continue
                    
            except requests.exceptions.Timeout:
                last_error = f"Timeout with {model}"
                continue
            except Exception as e:
                last_error = f"Error with {model}: {str(e)}"
                continue
        
        # If all models fail
        return f"""⚠️ Could not get a response from Hugging Face models. This might be because:

1. The models are currently loading (this can take 20-60 seconds)
2. High API traffic - please try again in a moment
3. API key issue

Please wait a moment and try again. 

Last error: {last_error}

**To get your free Hugging Face API key:**
1. Visit https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Add it to .streamlit/secrets.toml as HUGGINGFACE_API_KEY"""
        
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease verify your Hugging Face API key at: https://huggingface.co/settings/tokens"

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

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
- Use clear, well-lit photos
- Focus on leaves
- Avoid blurry images
- Multiple angles help
""")

# ----------------------------------
# Home Page
# ----------------------------------
if app_mode == "🏠 Home":
    st.markdown("<h1>🌿 Plant Doctor AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px; color: #666;'>AI-Powered Plant Identification & Chat Assistant</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero Image Placeholder
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1466781783364-36c955e42a7f?w=800", 
                 use_column_width=True, caption="Identify plants instantly with AI")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.markdown("### ✨ Key Features")
    col1, col2 = st.columns(2)
    
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
            <div style="font-size: 48px;">💬</div>
            <h3>AI Chat Expert</h3>
            <p>Ask questions and get personalized plant care advice</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Stats
    st.markdown("### 📊 Powered by Advanced AI")
    col1, col2, col3 = st.columns(3)
    
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
            <div class="stat-number">95%</div>
            <div>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
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
    3. Get instant identification
    4. Chat with AI for personalized advice
    """)
    
    # API Setup Warning
    if not PLANT_ID_API_KEY or not HUGGINGFACE_API_KEY:
        st.warning("""
        ⚠️ **API Keys Configuration**
        
        This app supports two detection methods:
        
        **Option 1: Plant.id (Most Accurate)**
        - Get API key from https://web.plant.id/
        - 10,000+ species database
        - High accuracy identification
        
        **Option 2: Hugging Face (Completely Free)**
        - Get API key from https://huggingface.co/settings/tokens
        - Uses AI vision models
        - Works great in Egypt! 🇪🇬
        
        Create `.streamlit/secrets.toml` file with:
        ```
        PLANT_ID_API_KEY = "your_plant_id_key"  # Optional
        HUGGINGFACE_API_KEY = "your_huggingface_key"  # Required for chat
        ```
        
        **You can use just Hugging Face for everything!**
        """)
    else:
        st.success("✅ API keys configured! You're ready to identify plants.")

# ----------------------------------
# Plant Detection Page
# ----------------------------------
elif app_mode == "🔍 Plant Detection":
    st.markdown("<h1>🔍 Plant Identification</h1>", unsafe_allow_html=True)
    
    # Detection method selector
    col_method1, col_method2 = st.columns(2)
    with col_method1:
        use_plantid = st.checkbox("Use Plant.id (Accurate)", value=bool(PLANT_ID_API_KEY), disabled=not PLANT_ID_API_KEY)
    with col_method2:
        use_hf = st.checkbox("Use Hugging Face (Free)", value=True)
    
    if not PLANT_ID_API_KEY and not HUGGINGFACE_API_KEY:
        st.error("⚠️ At least one API key is required. Please add PLANT_ID_API_KEY or HUGGINGFACE_API_KEY to .streamlit/secrets.toml")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Plant Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="plant_upload")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            identify_btn = st.button("🔍 Identify Plant", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Identification Results")
        
        if uploaded_file:
            if identify_btn:
                results_found = False
                
                # Try Hugging Face first if selected
                if use_hf and HUGGINGFACE_API_KEY:
                    with st.spinner("🤖 Analyzing with Hugging Face Plant Classifier..."):
                        hf_result = detect_plant_with_huggingface(image)
                        
                        if not hf_result.get("error"):
                            data = hf_result.get("data", {})
                            predictions = data.get("predictions", [])
                            top_pred = data.get("top_prediction", {})
                            
                            if top_pred:
                                plant_label = top_pred.get('label', 'Unknown')
                                confidence = top_pred.get('score', 0) * 100
                                
                                # Clean up label (remove underscores, capitalize)
                                plant_name = plant_label.replace('_', ' ').title()
                                
                                st.markdown(f"""
                                <div class="result-card">
                                    <h2>🤖 AI Plant Classification</h2>
                                    <h3>🌿 {plant_name}</h3>
                                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                    <p><strong>Model:</strong> {data.get('model', 'Unknown').split('/')[-1]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show top 3 predictions
                                if len(predictions) > 1:
                                    st.markdown("### 📊 Top Predictions")
                                    for i, pred in enumerate(predictions[:3], 1):
                                        label = pred.get('label', 'Unknown').replace('_', ' ').title()
                                        score = pred.get('score', 0) * 100
                                        st.write(f"{i}. **{label}** - {score:.1f}%")
                                
                                # Get more info using AI chat
                                if HUGGINGFACE_API_KEY:
                                    with st.spinner("🤔 Getting detailed plant information..."):
                                        plant_query = f"Tell me about {plant_name}. Include: scientific name, family, common care tips, and interesting facts. Keep it concise (3-4 sentences)."
                                        plant_info = chat_with_huggingface(plant_query, None)
                                        
                                        st.markdown(f"""
                                        <div class="result-card">
                                            <h3>📚 Plant Information</h3>
                                            <p>{plant_info}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Save to context
                                st.session_state.plant_context = {
                                    'plant_name': plant_name,
                                    'scientific_name': 'Detected by AI Model',
                                    'common_names': [plant_name],
                                    'confidence': confidence
                                }
                                
                                results_found = True
                        else:
                            st.warning(f"Hugging Face: {hf_result.get('message')}")
                
                # Try Plant.id if selected and available
                if use_plantid and PLANT_ID_API_KEY:
                    with st.spinner("🔍 Identifying with Plant.id..."):
                        result = identify_plant(image)
                        
                        if not result.get("error"):
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
                                
                                results_found = True
                        else:
                            st.warning(f"Plant.id error: {result.get('message')}")
                
                if results_found:
                    # Save to history
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'plant_name': st.session_state.plant_context.get('plant_name', 'Unknown'),
                        'type': 'identification'
                    })
                    
                    st.success("✅ Plant identified! You can now chat with AI about this plant.")
                else:
                    st.warning("No plant identified. Please try a clearer image or enable detection methods.")
        else:
            st.info("👆 Upload an image to get started")

# ----------------------------------
# AI Chat Page
# ----------------------------------
elif app_mode == "💬 AI Chat":
    st.markdown("<h1>💬 Chat with Plant Expert AI</h1>", unsafe_allow_html=True)
    
    if not HUGGINGFACE_API_KEY:
        st.error("⚠️ Hugging Face API key not configured. Please add it to .streamlit/secrets.toml")
        st.info("Get your free API key at: https://huggingface.co/settings/tokens")
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
                st.write(f"**Common Names:** {', '.join(context.get('common_names', [])[:2])}")
                st.write(f"**Confidence:** {context.get('confidence', 0):.1f}%")
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
        with st.spinner("🤔 Thinking... (first request may take 30-60 seconds)"):
            response = chat_with_huggingface(user_input, st.session_state.plant_context)
        
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
                    with st.spinner("🤔 Thinking..."):
                        response = chat_with_huggingface(question, st.session_state.plant_context)
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
        
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()
