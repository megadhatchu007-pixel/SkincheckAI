"""
SkinCheck AI — multi-page hospital-theme version
Landing -> Login/Signup -> [Analyze | Skincare Tips | Diet Plan | Shop Products]

HOW TO RUN:
1. Put skin_model.h5 and class_names.json in this same folder
2. pip install streamlit tensorflow pillow bcrypt numpy
3. streamlit run app.py
"""

import streamlit as st
import sqlite3
import bcrypt
import json
import os
import random
import urllib.parse
import numpy as np
from PIL import Image, ImageEnhance

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SkinCheck AI", page_icon="🩺", layout="centered")

# ---------- CUSTOM CSS: HOSPITAL THEME ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(160deg, #e0f2fe 0%, #ffffff 45%, #fef2f2 100%); }
    .hero-title {
        font-size: 44px; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #0ea5e9, #06b6d4, #ef4444);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .hero-sub { text-align: center; color: #475569; font-size: 16px; margin-bottom: 20px; }
    .page-title {
        font-size: 30px; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #0ea5e9, #06b6d4, #ef4444);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .doctor-emoji { font-size: 80px; text-align: center; margin-top: 20px; }
    .feature-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 18px; text-align: center; margin-bottom: 10px;
        box-shadow: 0 4px 14px rgba(14,165,233,0.08);
    }
    .feature-emoji { font-size: 34px; }
    .result-card {
        background: #ffffff; border-radius: 18px; padding: 22px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 18px rgba(14,165,233,0.1);
        margin-bottom: 16px; color: #1e293b;
    }
    .badge-danger { background: #ef4444; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; display: inline-block; }
    .badge-warning { background: #f59e0b; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; display: inline-block; }
    .badge-safe { background: #22c55e; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px; display: inline-block; }
    .tip-card {
        background: #ffffff; border-radius: 14px; padding: 16px 20px;
        border-left: 5px solid #06b6d4; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 12px; color: #1e293b;
    }
    .product-card {
        background: #ffffff; border-radius: 14px; padding: 14px 18px;
        border-left: 5px solid #0ea5e9; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 12px; color: #1e293b;
    }
    .diet-pill {
        display: inline-block; background: rgba(14,165,233,0.1);
        border: 1px solid rgba(14,165,233,0.35); color: #0369a1;
        padding: 6px 14px; border-radius: 20px; margin: 4px; font-size: 14px;
    }
    .diet-card {
        background: #ffffff; border-radius: 14px; padding: 14px 18px;
        border-left: 5px solid #22c55e; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 10px; color: #1e293b; font-size: 15px;
    }
    div.stButton > button {
        border-radius: 12px; font-weight: 600; padding: 10px 0px;
        background: linear-gradient(90deg, #0ea5e9, #06b6d4);
        color: white; border: none;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #0284c7, #0891b2);
        box-shadow: 0 4px 14px rgba(14,165,233,0.35);
    }
    div.stLinkButton > a {
        border-radius: 12px; font-weight: 600;
        background: linear-gradient(90deg, #f472b6, #ef4444) !important;
        color: white !important; border: none !important;
    }
    /* Text inputs */
    .stTextInput input, .stNumberInput input {
        background-color: #ffffff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        padding: 10px 12px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border: 2px solid #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14,165,233,0.2) !important;
    }
    .stTextInput input::placeholder { color: #94a3b8 !important; }
    .stTextInput label, .stSelectbox label, .stRadio label { color: #0f172a !important; font-weight: 600; }
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 10px !important;
        color: #0f172a !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-weight: 700; color: #0369a1; }
    .stTabs [aria-selected="true"] { color: #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- CONSTANTS ----------
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "skin_model.h5")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")
IMG_SIZE = 224

ANALYSIS_MESSAGES = [
    "🔬 Scanning skin cells...", "🤖 Running AI analysis...", "🧪 Processing image...",
    "🩺 Consulting the AI dermatologist...", "✨ Almost there...",
]

CONDITION_INFO = {
    "akiec": {"name": "Actinic Keratosis", "emoji": "🟠", "severity": "warning",
        "severity_label": "Precancerous — see a dermatologist",
        "tips": ["Book an appointment with a dermatologist soon — this can develop into skin cancer if untreated.",
            "Avoid direct sun exposure on the affected area.",
            "Use a broad-spectrum SPF 30+ sunscreen daily.",
            "Do not attempt to treat this with over-the-counter products alone."]},
    "bcc": {"name": "Basal Cell Carcinoma", "emoji": "🔴", "severity": "danger",
        "severity_label": "Cancerous — see a dermatologist immediately",
        "tips": ["This requires prompt medical evaluation — please see a dermatologist as soon as possible.",
            "Avoid picking at or irritating the area.",
            "Protect the area from further sun exposure.",
            "Early treatment has a very high success rate — don't delay."]},
    "bkl": {"name": "Benign Keratosis", "emoji": "🟢", "severity": "safe",
        "severity_label": "Generally harmless",
        "tips": ["Usually non-cancerous, but get any new or changing spot checked to confirm.",
            "Keep the area moisturized with a fragrance-free lotion.",
            "Avoid excessive scratching or friction on the spot.",
            "Use sunscreen to prevent new spots from forming."]},
    "df": {"name": "Dermatofibroma", "emoji": "🟢", "severity": "safe",
        "severity_label": "Benign",
        "tips": ["Typically harmless — a firm, benign skin growth.",
            "No treatment usually needed unless it's bothersome.",
            "Avoid repeated trauma or shaving over the area.",
            "Mention it to your doctor at your next routine check-up."]},
    "mel": {"name": "Melanoma", "emoji": "🔴", "severity": "danger",
        "severity_label": "Serious — see a dermatologist immediately",
        "tips": ["This is a potentially serious finding — seek professional medical evaluation right away.",
            "Do not delay — early detection significantly improves outcomes.",
            "Avoid sun exposure on the area until evaluated.",
            "This tool is not a diagnosis — only a dermatologist can confirm."]},
    "nv": {"name": "Melanocytic Nevus (Common Mole)", "emoji": "🟢", "severity": "safe",
        "severity_label": "Generally harmless",
        "tips": ["Common, benign mole — routine monitoring is enough for most people.",
            "Watch for the ABCDE signs: Asymmetry, Border irregularity, Color change, Diameter growth, Evolving shape.",
            "Use sunscreen to protect moles from UV damage.",
            "See a doctor if it changes in size, shape, or color."]},
    "vasc": {"name": "Vascular Lesion", "emoji": "🟡", "severity": "warning",
        "severity_label": "Generally harmless",
        "tips": ["Usually a harmless blood-vessel related mark (e.g. angioma).",
            "No treatment necessary unless cosmetic concern.",
            "Protect the area from sun exposure.",
            "Consult a doctor if it bleeds, grows rapidly, or changes."]},
}

GENERAL_SKINCARE_TIPS = [
    ("☀️", "Wear SPF 30+ sunscreen every single day, rain or shine."),
    ("💧", "Cleanse twice daily — morning and night — with a gentle cleanser."),
    ("🧴", "Moisturize right after washing your face to lock in hydration."),
    ("🚫", "Avoid touching or picking at your face throughout the day."),
    ("🛏️", "Change your pillowcase weekly to reduce bacteria buildup."),
    ("🧼", "Always remove makeup before sleeping."),
]

SKIN_TYPE_PRODUCTS = {
    "Oily": [("🧴", "Oily Skin Foaming Cleanser", "Cetaphil"),
             ("💧", "2% Salicylic Acid Serum", "Minimalist"),
             ("🍃", "Tea Tree Face Wash", "Mamaearth"),
             ("☀️", "Oil-Free Sunscreen SPF50", "The Derma Co")],
    "Dry": [("🧴", "Moisturizing Cream", "Cetaphil"),
            ("💧", "Ceramide Moisturizer", "Minimalist"),
            ("🍚", "Rice Water Face Wash", "Mamaearth"),
            ("☀️", "Hydrating Sunscreen SPF50", "The Derma Co")],
    "Combination": [("🧴", "Gentle Skin Cleanser", "Cetaphil"),
                     ("💧", "Niacinamide 10% Serum", "Minimalist"),
                     ("🍃", "Vitamin C Face Wash", "Mamaearth"),
                     ("☀️", "Sunscreen Aqua Gel SPF50", "The Derma Co")],
    "Sensitive": [("🧴", "Gentle Skin Cleanser", "Cetaphil"),
                  ("💧", "Sepicalm 3% Serum", "Minimalist"),
                  ("🌿", "Ubtan Face Wash", "Mamaearth"),
                  ("☀️", "Zinc Oxide Mineral Sunscreen", "The Derma Co")],
}

DIET_TIPS_BY_SKIN = {
    "Oily": [("💧", "Plenty of water to flush toxins"),
             ("🥗", "Leafy greens & fiber-rich foods"),
             ("🍋", "Vitamin C fruits to control excess oil"),
             ("🚫", "Avoid deep-fried & greasy foods"),
             ("🍵", "Green tea helps reduce sebum")],
    "Dry": [("🥑", "Healthy fats: avocado, nuts, seeds"),
            ("🐟", "Omega-3 rich fish or flaxseed"),
            ("🍯", "Honey & warm water in the morning"),
            ("💧", "Increase daily water intake"),
            ("🥛", "Milk/yogurt for skin hydration")],
    "Combination": [("⚖️", "Balanced diet with fruits & veggies"),
                     ("🍊", "Vitamin C to balance the T-zone"),
                     ("🥜", "Moderate healthy fats"),
                     ("💧", "Stay well hydrated"),
                     ("🍵", "Green tea for antioxidants")],
    "Sensitive": [("🌾", "Oats & anti-inflammatory foods"),
                  ("🫐", "Berries — antioxidants calm the skin"),
                  ("🚫", "Limit spicy & processed foods"),
                  ("💧", "Warm water, go easy on caffeine"),
                  ("🥥", "Coconut water for gentle hydration")],
}

def purplle_link(query):
    return f"https://www.purplle.com/search?q={urllib.parse.quote(query)}"

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL)""")
    conn.commit()
    conn.close()

def create_user(phone, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        c.execute("INSERT INTO users (phone, password_hash) VALUES (?, ?)", (phone, pw_hash))
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this mobile number already exists."
    finally:
        conn.close()

def verify_user(phone, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE phone = ?", (phone,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return False
    return bcrypt.checkpw(password.encode(), row[0])

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names

def enhance_regular_photo(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    cropped = ImageEnhance.Contrast(cropped).enhance(1.15)
    cropped = ImageEnhance.Sharpness(cropped).enhance(1.3)
    return cropped

def predict_image(image: Image.Image, model, class_names):
    import tensorflow as tf
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argmax(preds)
    return class_names[top_idx], float(preds[top_idx]), preds

# ---------- SESSION STATE ----------
for key, default in [("logged_in", False), ("user_phone", None), ("show_auth", False),
                      ("just_logged_out", False), ("selected_skin_type", "Oily"), ("nav_page", "🔍 Analyze")]:
    if key not in st.session_state:
        st.session_state[key] = default

init_db()

def is_valid_phone(phone):
    digits = phone.strip().replace(" ", "").replace("-", "").replace("+", "")
    return digits.isdigit() and 7 <= len(digits) <= 15

# ---------- LANDING PAGE ----------
def landing_page():
    if st.session_state.just_logged_out:
        st.success("👋 Logged out successfully! Take care of your skin 💖 See you again soon!")
        st.session_state.just_logged_out = False

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.markdown('<div class="doctor-emoji">🧑‍⚕️</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="hero-title">🩺 SkinCheck AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">AI-powered skin analysis with personalized skincare & diet guidance ✨</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="doctor-emoji">👩‍⚕️</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="feature-card"><div class="feature-emoji">📸</div><b>Upload a Photo</b><br><span style="color:#64748b;font-size:13px;">Snap or upload a skin image</span></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-card"><div class="feature-emoji">🤖</div><b>AI Analysis</b><br><span style="color:#64748b;font-size:13px;">Instant condition detection</span></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-card"><div class="feature-emoji">💆</div><b>Skincare & Diet</b><br><span style="color:#64748b;font-size:13px;">Products & food suggestions</span></div>', unsafe_allow_html=True)

    st.write("")
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state.show_auth = True
        st.rerun()

# ---------- LOGIN / SIGNUP ----------
def login_signup_page():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.markdown('<div class="doctor-emoji" style="font-size:60px;">🏥</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="hero-title" style="font-size:32px;">🩺 SkinCheck AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Log in or create an account to continue 🔐</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="doctor-emoji" style="font-size:60px;">💊</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Log In", "📝 Sign Up"])
    with tab1:
        phone = st.text_input("📱 Mobile Number", key="login_phone", placeholder="e.g. 9876543210")
        password = st.text_input("🔒 Password", type="password", key="login_password")
        if st.button("Log In", use_container_width=True):
            if not phone or not password:
                st.warning("⚠️ Please enter both mobile number and password.")
            elif verify_user(phone, password):
                st.session_state.logged_in = True
                st.session_state.user_phone = phone
                st.rerun()
            else:
                st.error("❌ Invalid mobile number or password.")
    with tab2:
        new_phone = st.text_input("📱 Mobile Number", key="signup_phone", placeholder="e.g. 9876543210")
        new_password = st.text_input("🔒 Password", type="password", key="signup_password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password", key="signup_confirm")
        if st.button("Create Account", use_container_width=True):
            if not new_phone or not new_password:
                st.warning("⚠️ Please fill in all fields.")
            elif not is_valid_phone(new_phone):
                st.error("❌ Please enter a valid mobile number.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif len(new_password) < 6:
                st.error("❌ Password must be at least 6 characters.")
            else:
                success, msg = create_user(new_phone, new_password)
                if success:
                    st.success(f"✅ {msg} Please log in from the 'Log In' tab.")
                else:
                    st.error(f"❌ {msg}")

# ---------- PAGE: ANALYZE ----------
def page_analyze():
    st.markdown('<div class="page-title">🔍 Skin Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a clear photo of the skin area you\'re concerned about 📸</div>', unsafe_allow_html=True)
    st.warning("⚠️ AI-generated estimate for educational purposes only — **not a medical diagnosis**. Always consult a dermatologist.", icon="⚠️")

    regular_photo_mode = st.checkbox("📷 This is a regular photo (not a close-up)")
    if regular_photo_mode:
        st.caption("⚠️ This model was trained on close-up dermoscopy images. Regular photos get extra enhancement, but results are still less reliable than a proper close-up.")

    uploaded_file = st.file_uploader("📤 Upload a skin image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("🔍 Analyze Image", use_container_width=True):
            with st.spinner(random.choice(ANALYSIS_MESSAGES)):
                model, class_names = load_model()
                img_to_use = enhance_regular_photo(image) if regular_photo_mode else image
                pred_class, confidence, all_preds = predict_image(img_to_use, model, class_names)
                info = CONDITION_INFO[pred_class]
            st.session_state.last_result = (info, confidence, class_names, all_preds)

    if "last_result" in st.session_state:
        info, confidence, class_names, all_preds = st.session_state.last_result
        badge_class = {"danger": "badge-danger", "warning": "badge-warning", "safe": "badge-safe"}[info["severity"]]
        st.markdown(f"""
        <div class="result-card">
            <h2>{info['emoji']} {info['name']}</h2>
            <span class="{badge_class}">{info['severity_label']}</span>
            <p style="margin-top:12px;color:#475569;">🎯 Confidence: <b>{confidence*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📊 See full confidence breakdown"):
            for i, cname in enumerate(class_names):
                st.write(f"{CONDITION_INFO[cname]['emoji']} {CONDITION_INFO[cname]['name']}: {all_preds[i]*100:.1f}%")
        st.info("👉 Head to the **💆 Skincare Tips** page from the sidebar for suggestions based on this result.")

# ---------- PAGE: SKINCARE TIPS ----------
def page_skincare():
    st.markdown('<div class="page-title">💆 Skincare Tips</div>', unsafe_allow_html=True)

    if "last_result" in st.session_state:
        info = st.session_state.last_result[0]
        st.markdown(f"#### {info['emoji']} Based on your last result: {info['name']}")
        for tip in info["tips"]:
            st.markdown(f'<div class="tip-card">✅ {tip}</div>', unsafe_allow_html=True)
        st.divider()

    st.markdown("#### 🌟 General Skincare Routine")
    for emoji, tip in GENERAL_SKINCARE_TIPS:
        st.markdown(f'<div class="tip-card">{emoji} {tip}</div>', unsafe_allow_html=True)

    if "last_result" not in st.session_state:
        st.info("👉 Run a scan on the **🔍 Analyze** page to get personalized tips here!")

# ---------- PAGE: DIET PLAN ----------
def page_diet():
    st.markdown('<div class="page-title">🥗 Diet Plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Choose your skin type for a tailored food guide 🍽️</div>', unsafe_allow_html=True)

    st.selectbox("🧴 Your Skin Type", ["Oily", "Dry", "Combination", "Sensitive"], key="selected_skin_type")
    skin_type = st.session_state.selected_skin_type

    st.markdown(f"#### 🍽️ Best Foods for {skin_type} Skin")
    for emoji, tip in DIET_TIPS_BY_SKIN[skin_type]:
        st.markdown(f'<div class="diet-card">{emoji} &nbsp; {tip}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 💧 Daily Hydration Goal")
    st.progress(0.7, text="6.5 / 8 glasses today 💧")

# ---------- PAGE: SHOP PRODUCTS ----------
def page_products():
    st.markdown('<div class="page-title">🛒 Shop Skincare Products</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Curated picks based on your skin type ✨</div>', unsafe_allow_html=True)

    st.selectbox("🧴 Your Skin Type", ["Oily", "Dry", "Combination", "Sensitive"], key="selected_skin_type")
    skin_type = st.session_state.selected_skin_type

    st.markdown(f"#### 🧴 Recommended for {skin_type} Skin")
    for emoji, product, brand in SKIN_TYPE_PRODUCTS[skin_type]:
        pc1, pc2 = st.columns([4, 1])
        with pc1:
            st.markdown(f'<div class="product-card">{emoji} <b>{brand} {product}</b></div>', unsafe_allow_html=True)
        with pc2:
            st.link_button("🛒 Buy", purplle_link(f"{brand} {product}"), use_container_width=True)

# ---------- MAIN APP ----------
def main_app():
    st.sidebar.markdown('<div style="font-size:50px;text-align:center;">🧑‍⚕️</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"### 👤 {st.session_state.user_phone}")
    st.sidebar.divider()

    st.sidebar.radio(
        "Navigate",
        ["🔍 Analyze", "💆 Skincare Tips", "🥗 Diet Plan", "🛒 Shop Products"],
        key="nav_page",
        label_visibility="collapsed"
    )

    st.sidebar.divider()
    if st.sidebar.button("🚪 Log Out"):
        st.session_state.logged_in = False
        st.session_state.user_phone = None
        st.session_state.show_auth = False
        st.session_state.just_logged_out = True
        st.session_state.pop("last_result", None)
        st.rerun()

    page = st.session_state.nav_page
    if page == "🔍 Analyze":
        page_analyze()
    elif page == "💆 Skincare Tips":
        page_skincare()
    elif page == "🥗 Diet Plan":
        page_diet()
    elif page == "🛒 Shop Products":
        page_products()

# ---------- ROUTER ----------
if st.session_state.logged_in:
    main_app()
elif st.session_state.show_auth:
    login_signup_page()
else:
    landing_page()
