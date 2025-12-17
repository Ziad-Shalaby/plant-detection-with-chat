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
# API Configuration - Multiple FREE Options!
# ----------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")  # FREE and FAST!
TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", "")  # FREE credits
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", "")  # FREE tier available

# ----------------------------------
# Plant Detection with Groq (FREE Vision AI!)
# ----------------------------------
def detect_plant_with_groq_llama_vision(image_data):
    """
    Detect plant using Groq's Llama 3.2 Vision (100% FREE!)
    Note: Groq vision is still in preview and may have limitations
    """
    if not GROQ_API_KEY:
        return {
            "error": True,
            "message": "Groq API key not configured"
        }
    
    try:
        # Resize image to reduce size (Groq has size limits)
        max_size = (800, 800)
        image_copy = image_data.copy()
        image_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = io.BytesIO()
        image_copy.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Groq API endpoint
        API_URL = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        # Simplified prompt that works better with vision models
        payload = {
            "model": "llama-3.2-11b-vision-preview",  # Using 11B model (more stable)
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Look at this plant image. Identify it and provide:
- Common name
- Scientific name (Genus species)
- Plant family
- Brief description (2-3 sentences)
- 3 care tips
- Is it edible?
- Native region

Format as JSON."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                text_content = result["choices"][0]["message"]["content"]
                
                try:
                    import re
                    # Clean up response
                    text_content = re.sub(r'```json\s*|\s*```', '', text_content).strip()
                    json_match = re.search(r'\{[\s\S]*\}', text_content)
                    
                    if json_match:
                        plant_data = json.loads(json_match.group())
                        # Ensure we have required fields
                        if "plant_name" not in plant_data:
                            plant_data["plant_name"] = "Detected Plant"
                        if "confidence" not in plant_data:
                            plant_data["confidence"] = 85
                    else:
                        # Extract info from text response
                        plant_data = {
                            "plant_name": "Detected Plant",
                            "description": text_content[:500],
                            "confidence": 80
                        }
                    
                    return {
                        "error": False,
                        "data": plant_data,
                        "raw_response": text_content,
                        "source": "Groq Llama 11B Vision"
                    }
                except json.JSONDecodeError:
                    return {
                        "error": False,
                        "data": {
                            "plant_name": "Detected Plant",
                            "description": text_content[:500],
                            "confidence": 75
                        },
                        "raw_response": text_content,
                        "source": "Groq Llama Vision"
                    }
            else:
                return {"error": True, "message": "No content in response"}
        else:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = f" - {error_data.get('error', {}).get('message', '')}"
            except:
                pass
            return {"error": True, "message": f"API Error: {response.status_code}{error_detail}"}
            
    except Exception as e:
        return {"error": True, "message": f"Error: {str(e)}"}


# ----------------------------------
# Plant Detection with Pixtral (Mistral Vision - FREE!)
# ----------------------------------
def detect_plant_with_mistral_vision(image_data):
    """
    Detect plant using Mistral's Pixtral vision model (FREE!)
    Very reliable and works great in Egypt
    """
    if not MISTRAL_API_KEY:
        return {"error": True, "message": "Mistral API key not configured"}
    
    try:
        # Resize image
        max_size = (1024, 1024)
        image_copy = image_data.copy()
        image_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        image_copy.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        API_URL = "https://api.mistral.ai/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
        
        payload = {
            "model": "pixtral-12b-2409",  # FREE vision model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Identify this plant. Provide in JSON format:
{
    "plant_name": "common name",
    "scientific_name": "Genus species",
    "family": "family name",
    "confidence": 90,
    "description": "brief description",
    "care_tips": ["tip1", "tip2", "tip3"],
    "interesting_facts": "fact",
    "common_issues": ["issue1", "issue2"],
    "is_edible": true/false,
    "native_region": "region"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result:
                text_content = result["choices"][0]["message"]["content"]
                
                try:
                    import re
                    text_content = re.sub(r'```json\s*|\s*```', '', text_content).strip()
                    json_match = re.search(r'\{[\s\S]*\}', text_content)
                    
                    if json_match:
                        plant_data = json.loads(json_match.group())
                        if "plant_name" not in plant_data:
                            plant_data["plant_name"] = "Detected Plant"
                        if "confidence" not in plant_data:
                            plant_data["confidence"] = 85
                        
                        return {
                            "error": False,
                            "data": plant_data,
                            "source": "Mistral Pixtral Vision"
                        }
                except:
                    pass
                
                return {
                    "error": False,
                    "data": {
                        "plant_name": "Detected Plant",
                        "description": text_content[:500],
                        "confidence": 80
                    },
                    "source": "Mistral Pixtral"
                }
        
        return {"error": True, "message": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": True, "message": f"Error: {str(e)}"}


# ----------------------------------
# Plant Detection with Together AI (FREE credits!)
# ----------------------------------
def detect_plant_with_together(image_data):
    """
    Detect plant using Together AI's vision models (FREE $25 credits!)
    """
    if not TOGETHER_API_KEY:
        return {"error": True, "message": "Together API key not configured"}
    
    try:
        buffered = io.BytesIO()
        image_data.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        API_URL = "https://api.together.xyz/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOGETHER_API_KEY}"
        }
        
        payload = {
            "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Identify this plant and provide details in JSON format:
{
    "plant_name": "Common name",
    "scientific_name": "Genus species",
    "family": "Family",
    "confidence": 90,
    "description": "Brief description",
    "care_tips": ["tip1", "tip2", "tip3"],
    "interesting_facts": "Fact",
    "common_issues": ["issue1", "issue2"],
    "is_edible": true,
    "native_region": "Region"
}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result:
                text_content = result["choices"][0]["message"]["content"]
                
                try:
                    import re
                    text_content = re.sub(r'```json\s*|\s*```', '', text_content).strip()
                    json_match = re.search(r'\{[\s\S]*\}', text_content)
                    
                    if json_match:
                        plant_data = json.loads(json_match.group())
                        return {
                            "error": False,
                            "data": plant_data,
                            "source": "Together AI"
                        }
                except:
                    pass
                
                return {
                    "error": False,
                    "data": {
                        "plant_name": "Detected Plant",
                        "description": text_content[:500],
                        "confidence": 80
                    },
                    "source": "Together AI"
                }
        
        return {"error": True, "message": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": True, "message": f"Error: {str(e)}"}


# ----------------------------------
# Smart Plant Detection (tries multiple free APIs)
# ----------------------------------
def smart_plant_detection(image_data):
    """
    Try multiple FREE APIs in order until one works
    """
    st.info("🔄 Trying FREE AI models...")
    
    # Try Mistral Pixtral first (very reliable!)
    if MISTRAL_API_KEY:
        with st.spinner("🎯 Trying Mistral Pixtral Vision (FREE)..."):
            result = detect_plant_with_mistral_vision(image_data)
            if not result.get("error"):
                st.success(f"✅ Success with {result.get('source', 'Mistral')}!")
                return result
            else:
                st.warning(f"⚠️ Mistral failed: {result.get('message')}")
    
    # Try Groq (FREE and FAST!)
    if GROQ_API_KEY:
        with st.spinner("🚀 Trying Groq Llama Vision (FREE)..."):
            result = detect_plant_with_groq_llama_vision(image_data)
            if not result.get("error"):
                st.success(f"✅ Success with {result.get('source', 'Groq')}!")
                return result
            else:
                st.warning(f"⚠️ Groq failed: {result.get('message')}")
    
    # Try Together AI (FREE $25 credits)
    if TOGETHER_API_KEY:
        with st.spinner("🔄 Trying Together AI (FREE)..."):
            result = detect_plant_with_together(image_data)
            if not result.get("error"):
                st.success(f"✅ Success with {result.get('source', 'Together')}!")
                return result
            else:
                st.warning(f"⚠️ Together AI failed: {result.get('message')}")
    
    return {
        "error": True,
        "message": "All FREE APIs failed. Please check your API keys or try again later. Make sure you have at least one valid API key configured."
    }


# ----------------------------------
# Chat Functions (Multiple FREE Options)
# ----------------------------------
def chat_with_ai(user_message, context=None):
    """
    Chat using multiple FREE AI APIs with fallback
    """
    # Build system prompt
    if context:
        system_prompt = f"""You are a helpful plant expert specializing in Egypt and Middle East.

Current plant context:
- Plant: {context.get('plant_name', 'Unknown')}
- Scientific Name: {context.get('scientific_name', 'N/A')}
- Family: {context.get('family', 'N/A')}

Provide practical, actionable advice suitable for Egyptian climate. Be concise, friendly, and focus on real-world tips."""
    else:
        system_prompt = "You are a knowledgeable plant expert with expertise in Egyptian and Middle Eastern plants, climate, and gardening. Provide helpful, practical advice. Be concise and friendly."
    
    # Try Groq first (fastest)
    if GROQ_API_KEY:
        try:
            API_URL = "https://api.groq.com/openai/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            }
            
            payload = {
                "model": "llama-3.1-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            pass  # Fall through to next API
    
    # Try Mistral as backup
    if MISTRAL_API_KEY:
        try:
            API_URL = "https://api.mistral.ai/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MISTRAL_API_KEY}"
            }
            
            payload = {
                "model": "mistral-small-latest",  # FREE tier
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            pass  # Fall through to next API
    
    # Try Together AI as last resort
    if TOGETHER_API_KEY:
        try:
            API_URL = "https://api.together.xyz/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOGETHER_API_KEY}"
            }
            
            payload = {
                "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            pass
    
    return "❌ All chat APIs failed. Please check your API keys are valid. Make sure you have at least one API key (Groq, Mistral, or Together AI) configured in your secrets.toml file."


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

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Active APIs")
if MISTRAL_API_KEY:
    st.sidebar.success("✅ Mistral Pixtral")
if GROQ_API_KEY:
    st.sidebar.success("✅ Groq")
if TOGETHER_API_KEY:
    st.sidebar.success("✅ Together AI")
if not MISTRAL_API_KEY and not GROQ_API_KEY and not TOGETHER_API_KEY:
    st.sidebar.error("❌ No API keys")

# ----------------------------------
# Home Page
# ----------------------------------
if app_mode == "🏠 Home":
    st.markdown("<h1>🌿 Plant Doctor AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px; color: #666;'>AI Plant Identification </p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1466781783364-36c955e42a7f?w=800", 
                 caption="Identify plants instantly with AI")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.markdown("### ✨ 100% FREE Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 48px;">🎯</div>
            <h3>Mistral Pixtral Vision</h3>
            <p>Excellent vision AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 48px;">💬</div>
            <h3>AI Chat</h3>
            <p>Unlimited questions about plant care</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # API Status
    if not MISTRAL_API_KEY and not GROQ_API_KEY and not TOGETHER_API_KEY:
        st.error("""
        ⚠️ **No API Keys Configured**
        
        Please add at least one API key to get started.
        
        **Mistral Pixtral is RECOMMENDED** - excellent vision model! 🎯
        **Groq is fastest** - good backup option! 🚀
        """)
    else:
        api_count = sum([bool(MISTRAL_API_KEY), bool(GROQ_API_KEY), bool(TOGETHER_API_KEY)])
        st.success(f"✅ {api_count} FREE API(s) configured! Ready to identify plants.")

# ----------------------------------
# Plant Detection Page
# ----------------------------------
elif app_mode == "🔍 Plant Detection":
    st.markdown("<h1>🔍 Plant Identification</h1>", unsafe_allow_html=True)
    
    if not MISTRAL_API_KEY and not GROQ_API_KEY and not TOGETHER_API_KEY:
        st.error("⚠️ At least one FREE API key required.")
        st.info("""
        **Get Mistral FREE API (RECOMMENDED):**
        https://console.mistral.ai/api-keys/
        
        **Or Groq FREE API (fastest):**
        https://console.groq.com/keys
        
        **Or Together AI ($25 FREE credits):**
        https://api.together.xyz/settings/api-keys
        """)
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Plant Image")
        uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"], key="plant_upload", label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            identify_btn = st.button("🔍 Identify Plant", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Identification Results")
        
        if uploaded_file and identify_btn:
            result = smart_plant_detection(image)
            
            if not result.get("error"):
                data = result.get("data", {})
                
                plant_name = data.get("plant_name", "Unknown Plant")
                scientific_name = data.get("scientific_name", "N/A")
                family = data.get("family", "N/A")
                confidence = data.get("confidence", 0)
                description = data.get("description", "No description available")
                
                st.markdown(f"""
                <div class="result-card">
                    <h2>🌿 {plant_name}</h2>
                    <p><strong>Scientific Name:</strong> {scientific_name}</p>
                    <p><strong>Family:</strong> {family}</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p><strong>Source:</strong> {result.get('source', 'AI Model')}</p>
                    <hr>
                    <p><strong>Description:</strong><br>{description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Care Tips
                if "care_tips" in data and data["care_tips"]:
                    st.markdown("### 💧 Care Tips")
                    for tip in data["care_tips"]:
                        st.write(f"• {tip}")
                
                # Additional Info
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if "is_edible" in data:
                        edible_status = "✅ Edible" if data["is_edible"] else "⚠️ Not Edible"
                        st.info(edible_status)
                
                with col_b:
                    if "native_region" in data:
                        st.info(f"🌍 {data['native_region']}")
                
                # Interesting Facts
                if "interesting_facts" in data:
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>✨ Did You Know?</h3>
                        <p>{data['interesting_facts']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Common Issues
                if "common_issues" in data and data["common_issues"]:
                    with st.expander("⚠️ Common Issues"):
                        for issue in data["common_issues"]:
                            st.write(f"• {issue}")
                
                # Save to context
                st.session_state.plant_context = {
                    'plant_name': plant_name,
                    'scientific_name': scientific_name,
                    'family': family,
                    'confidence': confidence
                }
                
                # Save to history
                st.session_state.detection_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'plant_name': plant_name,
                    'scientific_name': scientific_name,
                    'type': 'identification'
                })
                
                st.success("✅ Plant identified! Go to AI Chat for more advice.")
            else:
                st.error(f"❌ {result.get('message')}")
        
        elif not uploaded_file:
            st.info("👆 Upload an image to get started")

# ----------------------------------
# AI Chat Page
# ----------------------------------
elif app_mode == "💬 AI Chat":
    st.markdown("<h1>💬 FREE AI Plant Expert</h1>", unsafe_allow_html=True)
    
    if not MISTRAL_API_KEY and not GROQ_API_KEY and not TOGETHER_API_KEY:
        st.error("⚠️ At least one API key needed for chat.")
        st.info("""
        Get a FREE API key from:
        - Mistral: https://console.mistral.ai/api-keys/
        - Groq: https://console.groq.com/keys
        - Together AI: https://api.together.xyz/settings/api-keys
        """)
        st.stop()
    
    # Display context
    if st.session_state.plant_context:
        with st.expander("📌 Current Plant", expanded=True):
            context = st.session_state.plant_context
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Plant:** {context.get('plant_name', 'Unknown')}")
                st.write(f"**Scientific:** {context.get('scientific_name', 'N/A')}")
            with col2:
                st.write(f"**Family:** {context.get('family', 'N/A')}")
                st.write(f"**Confidence:** {context.get('confidence', 0)}%")
    
    st.markdown("---")
    
    # Chat History
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
                <strong>🤖 AI:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat Input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Ask anything...", key="chat_input", placeholder="How to grow tomatoes in Egypt?")
    
    with col2:
        send_btn = st.button("Send 📨", use_container_width=True)
    
    if send_btn and user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        with st.spinner("🤔 AI thinking..."):
            response = chat_with_ai(user_input, st.session_state.plant_context)
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()
    
    # Quick Questions
    if len(st.session_state.chat_history) == 0:
        st.markdown("### 💡 Quick Questions")
        questions = [
            "Best plants for Egyptian summer?",
            "How to deal with aphids?",
            "Indoor plants for apartments?",
            "When to fertilize roses?"
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(questions):
            with cols[i % 2]:
                if st.button(q, key=f"q_{i}"):
                    st.session_state.chat_history.append({'role': 'user', 'content': q})
                    with st.spinner("🤔 Thinking..."):
                        response = chat_with_ai(q, st.session_state.plant_context)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()
    
    # Clear Chat
    if len(st.session_state.chat_history) > 0:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ----------------------------------
# My Plants Page
# ----------------------------------
elif app_mode == "📚 My Plants":
    st.markdown("<h1>📚 My Plant Collection</h1>", unsafe_allow_html=True)
    
    if not st.session_state.detection_history:
        st.info("🌱 No plants identified yet. Go to Plant Detection!")
    else:
        st.markdown(f"### Total: {len(st.session_state.detection_history)}")
        
        for record in reversed(st.session_state.detection_history):
            with st.expander(f"🌿 {record['plant_name']} - {record['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Plant:** {record['plant_name']}")
                    st.write(f"**Scientific:** {record.get('scientific_name', 'N/A')}")
                with col2:
                    st.write(f"**Date:** {record['timestamp']}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()



