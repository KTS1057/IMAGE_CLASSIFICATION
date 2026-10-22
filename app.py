import os
import time
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Vision Classifier — Animals, Fruits & Objects",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .animal-badge { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .fruit-badge { background-color: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
    .general-badge { background-color: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DOMAIN KNOWLEDGE BASES (ANIMALS & FRUITS)
# ---------------------------------------------------------
FRUIT_KNOWLEDGE = {
    "banana": {"calories": "89 kcal / 100g", "vitamins": "Vitamin B6, Vitamin C, Potassium", "health_benefit": "Supports sustained energy and healthy blood pressure regulation."},
    "apple": {"calories": "52 kcal / 100g", "vitamins": "Vitamin C, Potassium, Dietary Fiber", "health_benefit": "High in polyphenols and pectin for gut microbiome and cardiovascular wellness."},
    "orange": {"calories": "47 kcal / 100g", "vitamins": "High Vitamin C, Folate, Thiamine", "health_benefit": "Powerful antioxidant booster supporting immune health and collagen synthesis."},
    "lemon": {"calories": "29 kcal / 100g", "vitamins": "Vitamin C, Citric Acid, Flavonoids", "health_benefit": "Enhances iron absorption, kidney health, and digestive function."},
    "strawberry": {"calories": "32 kcal / 100g", "vitamins": "Vitamin C, Manganese, Folate", "health_benefit": "Loaded with anthocyanins that support heart health and glycemic balance."},
    "pineapple": {"calories": "50 kcal / 100g", "vitamins": "Vitamin C, Bromelain, Manganese", "health_benefit": "Contains bromelain, a natural digestive enzyme with anti-inflammatory properties."},
    "pomegranate": {"calories": "83 kcal / 100g", "vitamins": "Vitamin C, Vitamin K, Punicalagins", "health_benefit": "High antioxidant potency supporting cellular protection and blood circulation."},
    "fig": {"calories": "74 kcal / 100g", "vitamins": "Calcium, Potassium, Dietary Fiber", "health_benefit": "Rich in natural prebiotics and essential minerals supporting bone density."}
}

ANIMAL_KNOWLEDGE = {
    "cat": {"taxonomy": "Mammalia (Felidae)", "trait": "Acute night vision and solitary predatory adaptations.", "fun_fact": "Cats can rotate their ears 180 degrees using 32 separate muscles."},
    "dog": {"taxonomy": "Mammalia (Canidae)", "trait": "Highly developed olfactory acuity and complex social hierarchy.", "fun_fact": "A dog's sense of smell is between 10,000 to 100,000 times more acute than humans."},
    "lion": {"taxonomy": "Mammalia (Panthera leo)", "trait": "Apex predator known as the only truly social wild cat species.", "fun_fact": "A lion's roar can be heard from up to 8 kilometers (5 miles) away."},
    "tiger": {"taxonomy": "Mammalia (Panthera tigris)", "trait": "Largest cat species with distinct camouflage coat markings.", "fun_fact": "No two tigers have the exact same stripe pattern; their skin also carries the stripes."},
    "elephant": {"taxonomy": "Mammalia (Elephantidae)", "trait": "Largest terrestrial mammal with advanced cognitive empathy.", "fun_fact": "An elephant trunk contains over 40,000 individual muscles and tendons."},
    "bear": {"taxonomy": "Mammalia (Ursidae)", "trait": "Plantigrade locomotion with omnivorous dietary adaptability.", "fun_fact": "Grizzly bears possess a sense of smell superior to bloodhounds."},
    "zebra": {"taxonomy": "Mammalia (Equidae)", "trait": "Herbivorous ungulate with unique thermal stripe patterns.", "fun_fact": "Zebras run in a zig-zag pattern when pursued by predators to break line of sight."},
    "panda": {"taxonomy": "Mammalia (Ailuropoda melanoleuca)", "trait": "Specialized folivore primarily consuming bamboo stalks.", "fun_fact": "Giant pandas spend up to 12 hours a day feeding to meet their caloric needs."},
    "horse": {"taxonomy": "Mammalia (Equidae)", "trait": "Single-toed ungulate with high endurance athletic physiology.", "fun_fact": "Horses have nearly 360-degree panoramic vision due to lateral eye placement."}
}

# ---------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vision_backbone():
    """Load pre-trained MobileNetV2 backbone (ImageNet weights)"""
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    model = MobileNetV2(weights="imagenet")
    return model

@st.cache_resource(show_spinner=False)
def load_custom_cat_dog_model():
    """Load local custom Cat vs Dog model if present"""
    import tensorflow as tf
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "CAT_DOG_MODEL.h5")
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            return None
    return None

# ---------------------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------------------
def predict_multiclass_vision(image: Image.Image, model, top_k=5):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    
    # Resize to MobileNetV2 input resolution (224, 224)
    target_size = (224, 224)
    img_resized = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    t0 = time.perf_counter()
    raw_preds = model.predict(img_preprocessed, verbose=0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    decoded = decode_predictions(raw_preds, top=top_k)[0]
    # Format: [(class_id, class_name, confidence_float), ...]
    results = []
    for cid, name, score in decoded:
        readable_name = name.replace("_", " ").title()
        results.append({
            "class_id": cid,
            "label": readable_name,
            "confidence": float(score) * 100.0
        })
    return results, latency_ms

def predict_custom_cat_dog(image: Image.Image, model):
    target_size = (160, 160)
    img_fitted = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img_fitted.convert("RGB"), dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0) / 255.0
    
    t0 = time.perf_counter()
    preds = model.predict(img_batch, verbose=0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    score = float(preds[0][0])
    # Dog if > 0.5, Cat if <= 0.5
    if score >= 0.5:
        primary = "Dog 🐶"
        confidence = score * 100.0
    else:
        primary = "Cat 🐱"
        confidence = (1.0 - score) * 100.0
    
    return {
        "primary": primary,
        "confidence": confidence,
        "dog_prob": score * 100.0,
        "cat_prob": (1.0 - score) * 100.0
    }, latency_ms

# ---------------------------------------------------------
# VISUAL FEATURE INSPECTOR (EDGE & SPATIAL GRADIENTS)
# ---------------------------------------------------------
def generate_feature_map(image: Image.Image):
    """Generates visual feature maps highlighting edges and textural gradients"""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    enhanced = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return edges, enhanced

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Model-Transfer_Learning_Vision-0A66C2?style=for-the-badge&logo=tensorflow&logoColor=white", use_container_width=True)
    st.markdown("### ⚙️ Engine Settings")
    
    model_choice = st.radio(
        "Select Vision Engine:",
        options=[
            "🌟 Multi-Category Vision (Animals, Fruits & Objects)",
            "🐾 Custom CNN (Cat vs Dog Dedicated Model)"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Detection Threshold")
    min_confidence = st.slider("Minimum Confidence Filter (%):", min_value=1, max_value=80, value=5)
    
    st.markdown("---")
    st.markdown("### 📚 Model Details")
    if "Multi-Category" in model_choice:
        st.info("**Backbone:** MobileNetV2 (ImageNet Pre-trained)\n\n**Parameters:** 3.5 Million\n\n**Classes:** 1,000 categories including dozens of domestic & wild animals, fruits, and produce.")
    else:
        st.info("**Backbone:** Custom Deep CNN (`CAT_DOG_MODEL.h5`)\n\n**Input Size:** 160 × 160 px\n\n**Target:** Binary classification between Canines and Felines.")

    st.markdown("---")
    st.caption("Developed by [Kratarth Srivastava](https://github.com/KTS1057)")

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧠 AI Vision Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Class Deep Learning Visual Recognition Engine for Animals, Fruits, Vegetables & Objects</div>', unsafe_allow_html=True)

# Input Tabs: Upload vs Live Camera
tab_upload, tab_camera = st.tabs(["📁 Upload Image File", "📸 Live Webcam Capture"])

uploaded_image = None

with tab_upload:
    file = st.file_uploader(
        "Choose an image (JPG, PNG, JPEG, WEBP)...",
        type=["jpg", "png", "jpeg", "webp"],
        help="Upload any clear photograph of an animal, fruit, or object."
    )
    if file:
        uploaded_image = Image.open(file)

with tab_camera:
    camera_pic = st.camera_input("Take a photo directly with your camera")
    if camera_pic:
        uploaded_image = Image.open(camera_pic)

# ---------------------------------------------------------
# INFERENCE PIPELINE
# ---------------------------------------------------------
if uploaded_image is not None:
    # Top Metrics Row
    col_img, col_preds = st.columns([1.1, 1.4], gap="large")
    
    with col_img:
        st.subheader("🖼️ Input Image")
        st.image(uploaded_image, use_container_width=True)
        
        # Image Metadata row
        w, h = uploaded_image.size
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-card"><div class="metric-value">{w}×{h}</div><div class="metric-label">Resolution</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-value">{uploaded_image.mode}</div><div class="metric-label">Channels</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-value">{w/h:.2f}</div><div class="metric-label">Aspect Ratio</div></div>', unsafe_allow_html=True)

    with col_preds:
        st.subheader("📊 Classification Results")
        
        if "Multi-Category" in model_choice:
            with st.spinner("Executing deep neural network inference..."):
                vision_model = load_vision_backbone()
                results, latency = predict_multiclass_vision(uploaded_image, vision_model, top_k=5)
            
            # Top-1 Winner
            top1 = results[0]
            label_lower = top1["label"].lower()
            
            # Category Detection
            is_animal = any(a in label_lower for a in ANIMAL_KNOWLEDGE.keys()) or any(keyword in label_lower for keyword in ["dog", "cat", "terrier", "hound", "retriever", "bird", "owl", "monkey", "fox", "wolf", "fish", "snake", "lizard", "horse", "pig", "cow", "sheep"])
            is_fruit = any(f in label_lower for f in FRUIT_KNOWLEDGE.keys()) or any(keyword in label_lower for keyword in ["apple", "banana", "orange", "lemon", "berry", "melon", "fruit", "pepper", "grape", "mango"])
            
            if is_animal:
                category_badge = '<span class="result-badge animal-badge">🐾 Animal Classification</span>'
            elif is_fruit:
                category_badge = '<span class="result-badge fruit-badge">🍏 Fruit / Produce Classification</span>'
            else:
                category_badge = '<span class="result-badge general-badge">🌐 Visual Recognition</span>'

            st.markdown(f"""
            <div style="background: white; border: 2px solid #2563EB; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    {category_badge}
                    <span style="font-size: 0.85rem; color: #64748B;">⚡ Latency: <b>{latency:.1f} ms</b></span>
                </div>
                <div style="font-size: 1.7rem; font-weight: 800; color: #1E293B;">{top1['label']}</div>
                <div style="font-size: 1.1rem; color: #2563EB; font-weight: 600;">Confidence: {top1['confidence']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top-5 Confidence Breakdown
            st.markdown("##### 🏆 Top Predicted Classes")
            filtered_results = [r for r in results if r["confidence"] >= min_confidence]
            if not filtered_results:
                st.warning(f"No predictions met the minimum confidence threshold ({min_confidence}%).")
            else:
                for idx, item in enumerate(filtered_results):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**#{idx+1} {item['label']}**")
                    c2.markdown(f"**{item['confidence']:.2f}%**")
                    st.progress(min(1.0, item["confidence"] / 100.0))
            
            # Domain Specific Insights
            matched_animal = next((k for k in ANIMAL_KNOWLEDGE.keys() if k in label_lower), None)
            matched_fruit = next((k for k in FRUIT_KNOWLEDGE.keys() if k in label_lower), None)
            
            if matched_animal:
                info = ANIMAL_KNOWLEDGE[matched_animal]
                st.markdown("---")
                st.markdown(f"##### 🐾 Animal Biology Insights (`{matched_animal.capitalize()}`)")
                st.info(f"**Taxonomy:** {info['taxonomy']}\n\n**Key Trait:** {info['trait']}\n\n💡 **Fun Fact:** {info['fun_fact']}")
            elif matched_fruit:
                info = FRUIT_KNOWLEDGE[matched_fruit]
                st.markdown("---")
                st.markdown(f"##### 🍏 Nutritional Profile (`{matched_fruit.capitalize()}`)")
                st.success(f"**Estimated Caloric Content:** {info['calories']}\n\n**Essential Micronutrients:** {info['vitamins']}\n\n🥗 **Health Benefit:** {info['health_benefit']}")

        else:
            # Custom Cat vs Dog Model
            custom_model = load_custom_cat_dog_model()
            if custom_model is None:
                st.error("Custom model file `CAT_DOG_MODEL.h5` was not found. Please ensure it is present in the repository.")
            else:
                with st.spinner("Running custom Cat vs Dog CNN model..."):
                    res, latency = predict_custom_cat_dog(uploaded_image, custom_model)
                
                st.markdown(f"""
                <div style="background: white; border: 2px solid #2563EB; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span class="result-badge animal-badge">Custom Binary CNN</span>
                        <span style="font-size: 0.85rem; color: #64748B;">⚡ Latency: <b>{latency:.1f} ms</b></span>
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #1E293B;">{res['primary']}</div>
                    <div style="font-size: 1.1rem; color: #2563EB; font-weight: 600;">Confidence: {res['confidence']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("##### 📈 Probability Distribution")
                st.markdown(f"🐶 **Dog Probability:** `{res['dog_prob']:.2f}%`")
                st.progress(res['dog_prob'] / 100.0)
                st.markdown(f"🐱 **Cat Probability:** `{res['cat_prob']:.2f}%`")
                st.progress(res['cat_prob'] / 100.0)

    # ---------------------------------------------------------
    # DEEP INSPECTION TABS
    # ---------------------------------------------------------
    st.markdown("---")
    inspect_tab1, inspect_tab2, inspect_tab3 = st.tabs([
        "🔬 Feature & Edge Map Visualization",
        "📐 Model Architecture Comparison",
        "📥 Download Prediction Summary"
    ])
    
    with inspect_tab1:
        st.markdown("##### Convolutional Layer Feature Extraction Representation")
        st.caption("Convolutional Neural Networks extract spatial hierarchies of features — early layers detect edges and gradients, while deeper layers synthesize high-level textures and semantic shapes.")
        edges_img, enhanced_img = generate_feature_map(uploaded_image)
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.image(edges_img, caption="Spatial Edge Boundaries (Sobel/Gradient Filter)", use_container_width=True)
        with f_col2:
            st.image(enhanced_img, caption="Textural Density & Contrast Filter", use_container_width=True)

    with inspect_tab2:
        st.markdown("##### Model Architecture Comparison Matrix")
        st.markdown("""
        | Attribute | MobileNetV2 Vision Backbone | Custom Cat/Dog CNN |
        | :--- | :--- | :--- |
        | **Primary Paradigm** | Pre-trained Transfer Learning (ImageNet) | Custom Shallow CNN Architecture |
        | **Output Classes** | **1,000 Classes** (Dozens of Animals, Fruits, Objects) | **2 Classes** (Cat vs Dog) |
        | **Parameter Count** | ~3.5 Million Parameters | ~2.5 Million Parameters |
        | **Specialized Feature** | Inverted Residual Blocks & Depthwise Separable Convolutions | Standard Sequential Conv2D & MaxPooling |
        | **Inference Efficiency** | Ultra-lightweight, optimized for edge/mobile devices | Lightweight local inference |
        """)

    with inspect_tab3:
        st.markdown("##### Export Inference Diagnostics")
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_resolution": f"{w}x{h}",
            "color_mode": uploaded_image.mode,
            "selected_engine": model_choice,
            "inference_latency_ms": f"{latency:.2f}"
        }
        if "Multi-Category" in model_choice:
            report_data["top_predictions"] = results
        else:
            report_data["prediction"] = res
            
        json_report = json.dumps(report_data, indent=2)
        st.download_button(
            label="📄 Download JSON Diagnostic Report",
            data=json_report,
            file_name="vision_classification_report.json",
            mime="application/json"
        )

else:
    st.info("👆 Please upload an image or capture one with your camera above to begin real-time visual classification.")
    
    # Showcase preview examples
    st.markdown("#### 🌟 What Can This Upgraded Model Classify?")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.markdown("""
        **🐾 Domestic & Wild Animals:**
        - Dogs (over 100 recognized breeds)
        - Cats (Persian, Siamese, Tabby)
        - African & Asian Elephants, Lions, Tigers
        - Bears, Zebras, Giraffes, Pandas, Horses
        """)
    with ex2:
        st.markdown("""
        **🍏 Fruits, Vegetables & Produce:**
        - Apples, Bananas, Oranges, Lemons
        - Strawberries, Pineapples, Pomegranates
        - Bell Peppers, Cucumbers, Broccoli
        - Includes automatic nutritional estimations!
        """)
    with ex3:
        st.markdown("""
        **🌐 General Objects & Everyday Items:**
        - Vehicles (Cars, Aircraft, Boats)
        - Electronics (Laptops, Phones, Cameras)
        - Musical Instruments, Sports Gear, etc.
        - Provides Top-5 probability distributions!
        """)
