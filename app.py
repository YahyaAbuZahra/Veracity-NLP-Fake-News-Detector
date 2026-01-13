import streamlit as st
import joblib
import os
import time
import re
import string
import textwrap

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Veracity.AI | SVM Powered Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Internal Cleaning Function ---
def clean_text(text):
    """
    Same cleaning logic used in training to remove leakage and noise.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove leakage words (agency names)
    text = re.sub(r'\b(?:reuters|wire|image|via|ap|fp|afp)\b', '', text)
    # Remove parentheses
    text = re.sub(r'\(.*?\)', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove punctuation & numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 3. Classic/Paper Design (CSS) ---
st.markdown("""
<style>
/* Import Serif Fonts for the Classic Look */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=Merriweather:wght@300;400;700&display=swap');

:root {
    --paper-bg: #fdf6e3;
    --sidebar-bg: #f4ecd8;
    --gold-border: #c5a059;
    --text-main: #2c2c2c;
    --text-sub: #4a3b2a;
    --gold-gradient: linear-gradient(180deg, #e6c883 0%, #c5a059 100%);
    --card-shadow: 2px 4px 12px rgba(0,0,0,0.08);
}

.stApp {
    background-color: var(--paper-bg);
    background-image: url("https://www.transparenttextures.com/patterns/cream-paper.png");
    font-family: 'Merriweather', serif;
    color: var(--text-main);
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
}

p, div, span, label, li {
    color: var(--text-sub);
}

[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: 2px solid var(--gold-border);
}

.sidebar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #2c1a0b;
    margin-bottom: 0.5rem;
    text-shadow: 1px 1px 0px rgba(255,255,255,0.5);
    border-bottom: 2px double var(--gold-border);
    padding-bottom: 10px;
}

.metric-card-sidebar {
    background: #fffdf5;
    border: 1px solid var(--gold-border);
    border-radius: 4px;
    padding: 15px;
    margin-bottom: 12px;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    position: relative;
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #2c1a0b;
}

.metric-label {
    font-family: 'Merriweather', serif;
    font-size: 0.7rem;
    color: #8c6d36;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.hero-container {
    text-align: center;
    padding: 3rem 1rem;
    background: transparent;
    border-radius: 8px;
    margin-bottom: 3rem;
    border: 3px double var(--gold-border);
    position: relative;
}

.hero-quote {
    background: var(--paper-bg);
    padding: 5px 15px;
    font-size: 0.85rem;
    color: #8c6d36;
    display: inline-block;
    margin-bottom: 1rem;
    font-family: 'Merriweather', serif;
    font-style: italic;
    border-bottom: 1px solid var(--gold-border);
}

.hero-title {
    font-size: 3.8rem;
    font-weight: 900;
    color: #1a1a1a;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}

.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.6) !important;
    border: 1px solid var(--gold-border) !important;
    border-radius: 4px !important;
    color: #1a1a1a !important;
    font-family: 'Merriweather', serif !important;
    font-size: 1rem !important;
}

.stButton > button {
    background: var(--gold-gradient) !important;
    color: #2c1a0b !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 4px !important;
    padding: 0.8rem 2rem !important;
    border: 1px solid #a38241 !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    text-transform: uppercase;
}

.result-card {
    background: #fffbf0;
    border-radius: 4px;
    padding: 2.5rem;
    text-align: center;
    border: 1px solid var(--gold-border);
    margin-top: 2rem;
    box-shadow: var(--card-shadow);
}

.footer-container {
    margin-top: 5rem;
    padding: 2rem;
    text-align: center;
    border-top: 1px solid var(--gold-border);
}
</style>
""", unsafe_allow_html=True)

# --- 4. Load Models (Smart Path Finding) ---
@st.cache_resource
def load_resources():
    # 1. تحديد مكان ملف app.py الحالي
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. البحث الذكي عن مجلد models
    # المحاولة الأولى: هل المجلد بجانبي؟ (داخل src)
    models_dir = os.path.join(current_dir, "models")
    
    # المحاولة الثانية: إذا لم أجده، هل هو في المجلد الرئيسي؟ (نرجع خطوة للوراء)
    if not os.path.exists(models_dir):
        # os.path.dirname(current_dir) يرجعنا من src إلى fake_news_detector
        models_dir = os.path.join(os.path.dirname(current_dir), "models")
    
    # تحديد مسارات الملفات
    model_path = os.path.join(models_dir, "best_fake_news_model.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    
    # التحقق النهائي
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("⚠️ CRITICAL ERROR: Could not find model files.")
        st.info(f"Searched in: {models_dir}")
        st.warning("Please make sure 'models' folder exists in your project root or inside src.")
        return None, None
    
    try:
        loaded_model = joblib.load(model_path)
        loaded_vectorizer = joblib.load(vectorizer_path)
        return loaded_model, loaded_vectorizer
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, vectorizer = load_resources()

# --- 5. Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Veracity.AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a3b2a; margin-bottom:2rem; font-style: italic;">AI-Powered Truth Detection</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card-sidebar">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">98.8%</div>
    </div>
    <div class="metric-card-sidebar">
        <div class="metric-label">F1-Score</div>
        <div class="metric-value">0.988</div>
    </div>
    <div class="metric-card-sidebar">
        <div class="metric-label">Engine</div>
        <div class="metric-value" style="font-size: 1.4rem;">SVM</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<span style='color: #4a3b2a;'>**Model:** Support Vector Machine</span>", unsafe_allow_html=True)

# --- 6. Hero Section ---
st.markdown("""
<div class="hero-container">
    <div class="hero-quote">Trust, but verify. Then verify again with AI.</div>
    <h1 class="hero-title">Fake News Detector</h1>
    <p style="color: #4a3b2a; font-size: 1.2rem; font-family: 'Merriweather', serif;">
        Powered by Linear SVM & TF-IDF for High-Precision Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# --- 7. Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Analyze Article")
    user_input = st.text_area(
        "Input",
        height=250,
        placeholder="Paste the news headline or article content here...",
        label_visibility="collapsed"
    )
    analyze_btn = st.button("ANALYZE VERACITY", use_container_width=True)

with col2:
    st.markdown("### System Guide")
    st.info("""
    **How to get best results:**
    
    1. Paste at least 2-3 full sentences.
    2. Include the title if available.
    3. The system removes agency names (e.g. Reuters) to ensure unbiased analysis.
    """)

# --- 8. Analysis & Results ---
if analyze_btn:
    if not user_input.strip():
        st.error("Please enter some text to analyze.")
    elif not model:
        st.error("Model files not found! Check path settings.")
    else:
        with st.spinner("Processing Linguistic Patterns..."):
            time.sleep(0.8)
            
            try:
                # Cleaning
                cleaned = clean_text(user_input)
                
                if not cleaned:
                    st.warning("Input text contains only symbols or removed words. Please provide more content.")
                else:
                    vector = vectorizer.transform([cleaned])
                    prediction = model.predict(vector)[0]

                    # Assuming 1 = True/Real, 0 = Fake
                    if prediction == 1:
                        color = "#2e7d32"
                        label = "LIKELY REAL NEWS"
                        desc = "This content matches patterns of trusted journalism."
                        icon = "⚖️"
                    else:
                        color = "#c62828" 
                        label = "LIKELY FAKE NEWS"
                        desc = "This content shows strong signs of misinformation."
                        icon = "🚫"

                    html_code = textwrap.dedent(f"""
                        <div class="result-card" style="border-top: 4px solid {color};">
                            <div style="font-size: 3rem; margin-bottom: 10px;">{icon}</div>
                            <h2 style="color: {color}; margin-bottom: 0.5rem; letter-spacing: 1px;">{label}</h2>
                            <p style="font-size: 1.1rem; color: #2c1a0b; font-family: 'Merriweather', serif;">{desc}</p>
                        </div>
                    """)
                    
                    st.markdown(html_code, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
 
# --- 9. Footer ---
st.markdown("""
<div class="footer-container">
    <div class="footer-name" style="font-size: 3.5rem; line-height: 1.2;">Yahya Abu Zahra</div>
    <div style="color: #4a3b2a; margin-top: 10px; font-weight: 600; font-size: 1.8rem;">
        Computer Engineering Undergraduate (AI Track)
    </div>
    <div style="color: #8c6d36; font-size: 1.1rem; margin-top: 25px;">
        © 2025 Veracity.AI Project. Built with SVM & Streamlit.
    </div>
</div>
""", unsafe_allow_html=True)