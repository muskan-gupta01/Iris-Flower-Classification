import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageOps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# PAGE CONFIGURATION AND THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="Iris AI Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global Settings */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0b0f19 50%, #030712 100%) !important;
        color: #f3f4f6;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Gradient Hero Title */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
    }
    .kpi-card-1 { border: 1px solid rgba(56, 189, 248, 0.3); box-shadow: 0 4px 20px -2px rgba(56, 189, 248, 0.15); }
    .kpi-card-2 { border: 1px solid rgba(168, 85, 247, 0.3); box-shadow: 0 4px 20px -2px rgba(168, 85, 247, 0.15); }
    .kpi-card-3 { border: 1px solid rgba(251, 191, 36, 0.3); box-shadow: 0 4px 20px -2px rgba(251, 191, 36, 0.15); }
    .kpi-card-4 { border: 1px solid rgba(52, 211, 153, 0.3); box-shadow: 0 4px 20px -2px rgba(52, 211, 153, 0.15); }
    
    .kpi-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Modern Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 10px;
    }
    
    /* Consistent metric cards for the dark navy dashboard */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20, 34, 57, 0.96), rgba(13, 23, 40, 0.96));
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 108px;
        box-shadow: 0 6px 20px rgba(2, 8, 23, 0.18);
    }
    div[data-testid="stMetric"] label {
        color: #a9bdd5 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f5f9ff !important;
        font-weight: 750 !important;
    }
    .performance-intro {
        background: linear-gradient(110deg, rgba(30, 64, 112, 0.38), rgba(88, 48, 145, 0.25));
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 15px 18px;
        margin: 8px 0 20px 0;
        color: #dbeafe;
        line-height: 1.6;
    }

    /* Professional About page sections */
    .about-hero { background: linear-gradient(115deg, rgba(37,99,235,.22), rgba(124,58,237,.20)); border: 1px solid rgba(125,160,255,.25); border-radius: 18px; padding: 24px 26px; margin: 8px 0 22px 0; }
    .about-hero h2 { color: #f8fbff; margin: 0 0 9px 0; font-size: 1.65rem; }
    .about-hero p { color: #c5d2e5; line-height: 1.75; margin: 0; }
    .about-section { background: rgba(15,27,46,.78); border: 1px solid rgba(125,160,255,.17); border-radius: 14px; padding: 17px 19px; min-height: 130px; margin-bottom: 12px; }
    .about-section h4 { color: #eaf3ff; margin: 0 0 9px 0; }
    .about-section p, .about-section li { color: #b9c8dc; line-height: 1.65; }
    .team-credit { background: linear-gradient(110deg, rgba(30,64,112,.35), rgba(88,48,145,.24)); border: 1px solid rgba(96,165,250,.25); border-radius: 14px; padding: 18px 20px; color: #dbeafe; line-height: 1.7; }
    .about-stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 12px 0 18px 0; }
    .about-stat { background: linear-gradient(145deg, rgba(20,34,57,.96), rgba(13,23,40,.96)); border: 1px solid rgba(96,165,250,.22); border-radius: 14px; padding: 17px 18px; min-width: 0; }
    .about-stat-label { color: #a9bdd5; font-size: .82rem; font-weight: 650; margin-bottom: 8px; }
    .about-stat-value { color: #f5f9ff; font-size: 1.12rem; line-height: 1.35; font-weight: 750; overflow-wrap: anywhere; }
    .team-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 11px; margin-top: 16px; }
    .team-member { display: flex; align-items: center; gap: 11px; padding: 12px 14px; background: rgba(15,27,46,.78); border: 1px solid rgba(125,160,255,.17); border-radius: 12px; color: #e6efff; }
    .team-number { display: inline-flex; align-items: center; justify-content: center; min-width: 30px; height: 30px; border-radius: 9px; background: rgba(56,189,248,.13); color: #7dd3fc; font-weight: 750; }
    @media (max-width: 850px) { .about-stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .team-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 520px) { .about-stat-grid, .team-grid { grid-template-columns: 1fr; } }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Buttons with Glow */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.6) !important;
        transform: translateY(-2px);
    }
    
    /* Clean Image Cards */
    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    iris_data = load_iris()
    data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    data["species"] = [iris_data.target_names[i] for i in iris_data.target]
    return data, iris_data

df, iris = load_dataset()
FEATURE_COLUMNS = list(iris.feature_names)
SPECIES_NAMES = list(iris.target_names)
SPECIES_COLORS = {
    "setosa": "#38BDF8",      # sky blue
    "versicolor": "#A78BFA",  # soft violet
    "virginica": "#34D399",   # mint green
}

# ---------------------------------------------------------
# FLOWER IMAGES 
# ---------------------------------------------------------
FLOWER_IMAGES = {
    "setosa": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa_2.jpg",
    "versicolor": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor_1.jpg",
    "virginica": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica_-_NRCS.jpg",
}

FLOWER_DESCRIPTIONS = {
    "setosa": "Known for its short, wide petals. It is the most distinct and easily separable species in the dataset.",
    "versicolor": "Features medium-sized petals and sepals. It shares some dimensional similarities with Virginica.",
    "virginica": "The largest of the three species, characterized by long sepals and broad petals. Prefers damp environments."
}

@st.cache_data(show_spinner=False)
def get_flower_image(url):
    try:
        response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return ImageOps.fit(image, (900, 675), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    except Exception:
        return None

def show_flower_image(species):
    image = get_flower_image(FLOWER_IMAGES[species])
    if image is None:
        st.info(f"Image for Iris {species.capitalize()} is unavailable.")
    else:
        st.image(image, width="stretch")

# ---------------------------------------------------------
# LOAD SAVED MODEL
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "iris_random_forest_model.joblib"

@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)

if not MODEL_PATH.exists():
    st.error("iris_random_forest_model.joblib was not found. Keep it in the same folder as app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()

@st.cache_data(show_spinner=False)
def evaluate_model(_estimator, model_version, features, targets):
    """Evaluate a fresh clone of the saved estimator on held-out data and CV folds."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.20, random_state=42, stratify=targets
    )
    holdout_model = clone(_estimator)
    holdout_model.fit(X_train, y_train)
    holdout_predictions = holdout_model.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        clone(_estimator), features, targets, cv=cv, scoring="accuracy"
    )
    return y_test, holdout_predictions, cv_scores, len(X_test)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None
if "bulk_results_file" not in st.session_state:
    st.session_state.bulk_results_file = None

# ---------------------------------------------------------
# COMMON FUNCTIONS
# ---------------------------------------------------------
def make_prediction(values):
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(np.max(probabilities) * 100)
    return prediction, probabilities, confidence

def save_history(values, prediction, confidence):
    st.session_state.history.append({
        "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Sepal Length (cm)": float(values[0]),
        "Sepal Width (cm)": float(values[1]),
        "Petal Length (cm)": float(values[2]),
        "Petal Width (cm)": float(values[3]),
        "Predicted Species": str(prediction),
        "Confidence (%)": round(float(confidence), 2),
    })

def csv_bytes(data):
    return data.to_csv(index=False).encode("utf-8-sig")

def render_chart(fig):
    fig.patch.set_facecolor("#0e1117")
    for ax in fig.axes:
        ax.set_facecolor("#151d2a")
        ax.tick_params(colors="#e5eaf3", labelsize=9)
        ax.xaxis.label.set_color("#e5eaf3")
        ax.yaxis.label.set_color("#e5eaf3")
        ax.title.set_color("#ffffff")
        ax.title.set_fontweight("bold")
        ax.title.set_fontsize(13)
        for spine in ax.spines.values():
            spine.set_color("#718096")
        ax.grid(True, color="#526176", alpha=0.22, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor("#202b3b")
            legend.get_frame().set_edgecolor("#526176")
            for item in legend.get_texts():
                item.set_color("#f4f6fb")
    fig.tight_layout(pad=2.0)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🌸 Iris AI Dashboard")
st.sidebar.caption("Machine Learning Application")
st.sidebar.divider()

def navigate_to(target_page):
    st.session_state["page_navigation"] = target_page


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Flower Prediction",
        "📁 Bulk Prediction",
        "📊 Data Visualization",
        "📈 Model Performance",
        "📜 Prediction History",
        "📘 About Project",
    ],
    key="page_navigation",
)

st.sidebar.divider()
st.sidebar.metric("Total Predictions", len(st.session_state.history))
st.sidebar.caption("Algorithm: Random Forest")
st.sidebar.caption("Dataset: Iris Dataset")
st.sidebar.divider()
st.sidebar.markdown("👥 **BCA AI/ML Project Group**")
st.sidebar.caption("Collaborative Academic Project")
st.sidebar.caption("Invertis University, Bareilly")

# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
        <div>
            <span class="hero-title">Iris MLOps & Classifier</span>
            <span class="status-badge">● Production v1.0.4 Online</span>
        </div>
        <p class="hero-subtitle">High-throughput classification pipeline engineered with Stratified Random Forest ensemble.</p>
    """, unsafe_allow_html=True)
    
    _, _, home_cv_scores, _ = evaluate_model(
        model, MODEL_PATH.stat().st_mtime, df[FEATURE_COLUMNS], df["species"]
    )
    home_cv_accuracy = home_cv_scores.mean() * 100

    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-card-1">
                <div class="kpi-label">Total Records</div>
                <div class="kpi-value">{len(df)}</div>
            </div>
            <div class="kpi-card kpi-card-2">
                <div class="kpi-label">Classified Species</div>
                <div class="kpi-value">3</div>
            </div>
            <div class="kpi-card kpi-card-3">
                <div class="kpi-label">Features Examined</div>
                <div class="kpi-value">4</div>
            </div>
            <div class="kpi-card kpi-card-4">
                <div class="kpi-label">5-Fold CV Accuracy</div>
                <div class="kpi-value">{home_cv_accuracy:.2f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 1rem;'>🌺 Botanical Specimen Gallery</h3>", unsafe_allow_html=True)
    img_cols = st.columns(3)
    
    for idx, species in enumerate(SPECIES_NAMES):
        with img_cols[idx]:
            show_flower_image(species)
            st.markdown(f"<h4 style='margin-top: 10px; margin-bottom: 0px;'>Iris {species.capitalize()}</h4>", unsafe_allow_html=True)
            st.caption(FLOWER_DESCRIPTIONS[species])
            
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<h3>⚡ Quick Pipeline Operations</h3>", unsafe_allow_html=True)
    qc1, qc2 = st.columns(2)
    with qc1:
        st.button(
            "🌸 Start Quick Prediction",
            type="primary",
            width="stretch",
            on_click=navigate_to,
            args=("🔍 Flower Prediction",),
        )
    with qc2:
        st.button(
            "📁 Open Bulk Prediction",
            width="stretch",
            on_click=navigate_to,
            args=("📁 Bulk Prediction",),
        )

    with st.expander("🧠 Model Information"):
        st.write("**Algorithm:** Random Forest Classifier")
        st.write(f"**Model file:** `{MODEL_PATH.name}`")
        st.write(f"**Input features:** {len(FEATURE_COLUMNS)}")
        st.write(f"**Output classes:** {', '.join(name.capitalize() for name in SPECIES_NAMES)}")
        model_parameters = model.get_params(deep=False)
        st.json({key: model_parameters[key] for key in ("n_estimators", "max_depth", "random_state", "criterion") if key in model_parameters})
        st.caption("Evaluation scores are calculated using fresh estimator clones on the Iris dataset; they are not training-history metrics stored inside the model file.")

# ---------------------------------------------------------
# FLOWER PREDICTION
# ---------------------------------------------------------
elif page == "🔍 Flower Prediction":
    st.title("🔍 Flower Prediction")
    st.write("Adjust the four measurements to predict the Iris species.")
    st.divider()
    
    left, right = st.columns(2)
    with left:
        st.subheader("🌿 Sepal Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
    with right:
        st.subheader("🌸 Petal Measurements")
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)
        
    values = [sepal_length, sepal_width, petal_length, petal_width]
    st.divider()
    
    st.subheader("📋 Input Summary")
    summary = st.columns(4)
    summary[0].metric("Sepal Length", f"{sepal_length:.1f} cm")
    summary[1].metric("Sepal Width", f"{sepal_width:.1f} cm")
    summary[2].metric("Petal Length", f"{petal_length:.1f} cm")
    summary[3].metric("Petal Width", f"{petal_width:.1f} cm")
    
    if st.button("🌸 Predict Flower Species", type="primary", width="stretch"):
        with st.spinner("Analysing flower measurements..."):
            prediction, probabilities, confidence = make_prediction(values)
        save_history(values, prediction, confidence)
        st.success("Prediction completed successfully.")
        st.divider()
        
        result, image_col = st.columns(2)
        with result:
            st.subheader("🌺 Predicted Species")
            st.markdown(f"# {prediction.capitalize()}")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
            st.progress(min(int(round(confidence)), 100))
        with image_col:
            st.subheader("🌸 Flower Reference")
            show_flower_image(prediction.lower())
            
        st.divider()
        st.subheader("📊 Class-wise Prediction Probabilities")
        probability_df = pd.DataFrame({
            "Species": [str(name).lower() for name in model.classes_],
            "Probability (%)": probabilities * 100,
        }).sort_values("Probability (%)", ascending=False)
        probability_df["Display name"] = probability_df["Species"].str.capitalize()
        fig, ax = plt.subplots(figsize=(10, 4.2))
        bar_colors = [SPECIES_COLORS.get(name, "#60A5FA") for name in probability_df["Species"]]
        bars = ax.bar(probability_df["Display name"], probability_df["Probability (%)"],
                      color=bar_colors, width=0.55, edgecolor="#dbeafe", linewidth=0.7)
        ax.set_title("Prediction Probability by Iris Species", fontsize=14, pad=15)
        ax.set_xlabel("Predicted species")
        ax.set_ylabel("Estimated probability (%)")
        ax.set_ylim(0, 110)
        for bar, value in zip(bars, probability_df["Probability (%)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, max(value + 2, 2),
                    f"{value:.1f}%", ha="center", va="bottom", color="#F5F9FF", fontweight="bold")
        render_chart(fig)
        st.caption("Each species has its own consistent chart color. Probability is the model's estimated confidence, not a guarantee that the prediction is correct.")

# ---------------------------------------------------------
# DATA VISUALIZATION
# ---------------------------------------------------------
elif page == "📊 Data Visualization":
    st.title("📊 Iris Data Visualization")
    st.write("Explore the Iris dataset through readable charts and statistical summaries. Charts use a consistent size and dark theme.")
    st.divider()
    
    st.subheader("📋 Dataset Overview")
    overview = st.columns(3)
    overview[0].metric("Total Records", len(df))
    overview[1].metric("Input Features", len(FEATURE_COLUMNS))
    overview[2].metric("Flower Species", df["species"].nunique())
    
    with st.expander("🔎 Dataset Preview and Statistical Summary"):
        st.dataframe(df.head(10), width="stretch")
        st.subheader("Statistical Summary")
        st.dataframe(df[FEATURE_COLUMNS].describe().round(2), width="stretch")
        
    st.divider()
    st.subheader("🌺 Species Distribution")
    counts = df["species"].value_counts().reindex(SPECIES_NAMES)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, width=0.58, color=[SPECIES_COLORS[name] for name in counts.index], edgecolor="#e5eaf3", linewidth=0.6)
    ax.set_title("Number of Samples in Each Iris Species", fontsize=14, pad=16)
    ax.set_xlabel("Iris Species")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(counts.values) * 1.2)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(int(bar.get_height())), ha="center", color="#ffffff", fontweight="bold")
    render_chart(fig)
    
    st.divider()
    st.subheader("📏 Feature Distribution")
    selected_feature = st.selectbox("Select a measurement", FEATURE_COLUMNS, key="distribution_feature")
    colors = SPECIES_COLORS
    fig, ax = plt.subplots(figsize=(10, 5))
    for species in SPECIES_NAMES:
        values = df.loc[df["species"] == species, selected_feature]
        ax.hist(values, bins=12, alpha=0.58, label=species.capitalize(), color=colors[species], edgecolor="#172131")
    ax.set_title(f"Distribution of {selected_feature}", fontsize=14, pad=16)
    ax.set_xlabel("Measurement (cm)")
    ax.set_ylabel("Frequency")
    ax.legend(title="Species")
    render_chart(fig)
    
    st.divider()
    st.subheader("📦 Feature Comparison by Species")
    box_feature = st.selectbox("Choose a measurement for comparison", FEATURE_COLUMNS, key="box_feature")
    box_data = [df.loc[df["species"] == species, box_feature] for species in SPECIES_NAMES]
    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(box_data, tick_labels=[species.capitalize() for species in SPECIES_NAMES], patch_artist=True, widths=0.55, medianprops={"color": "#ffbd4a", "linewidth": 2})
    for patch, color in zip(box["boxes"], [SPECIES_COLORS[name] for name in SPECIES_NAMES]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_title(f"{box_feature} by Iris Species", fontsize=14, pad=16)
    ax.set_xlabel("Iris Species")
    ax.set_ylabel("Measurement (cm)")
    render_chart(fig)
    
    st.divider()
    st.subheader("🔥 Feature Correlation Heatmap")
    correlation = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=25, ha="right")
    ax.set_yticklabels(FEATURE_COLUMNS)
    for row in range(len(FEATURE_COLUMNS)):
        for column in range(len(FEATURE_COLUMNS)):
            value = correlation.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="#111827" if abs(value) > 0.55 else "#ffffff", fontweight="bold")
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors="#e5eaf3")
    colorbar.set_label("Correlation", color="#e5eaf3")
    ax.set_title("Correlation Between Input Features", fontsize=14, pad=16)
    render_chart(fig)
    
    st.divider()
    st.subheader("🧬 Principal Component Analysis (PCA)")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    fig, ax = plt.subplots(figsize=(10, 6))
    for species in SPECIES_NAMES:
        selected = pca_df[pca_df["species"] == species]
        ax.scatter(selected["PC1"], selected["PC2"], label=species.capitalize(), color=colors[species], alpha=0.82, s=55, edgecolors="#0e1117", linewidths=0.4)
    variance = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Principal Component 1 ({variance[0]:.1f}% variance)")
    ax.set_ylabel(f"Principal Component 2 ({variance[1]:.1f}% variance)")
    ax.set_title("Iris Dataset: PCA Projection", fontsize=14, pad=16)
    ax.legend(title="Species")
    render_chart(fig)
    st.info(f"The first two principal components explain {variance.sum():.2f}% of the total dataset variance.")

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("""
        <div class="performance-intro">
            <strong>Model Evaluation Dashboard</strong><br>
            Review test-set performance, cross-validation stability, classification quality,
            prediction errors and the contribution of each input feature.
            Metrics are calculated by fitting fresh copies of the saved Random Forest configuration
            on the Iris dataset; they are not historical training results stored in the model file.
        </div>
    """, unsafe_allow_html=True)
    
    X = df[FEATURE_COLUMNS]
    y = df["species"]
    y_test, predictions, cv_scores, test_sample_count = evaluate_model(
        model, MODEL_PATH.stat().st_mtime, X, y
    )
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    
    st.subheader("🎯 Performance Overview")
    metrics = st.columns(4)
    metrics[0].metric("Test Accuracy", f"{accuracy * 100:.2f}%")
    metrics[1].metric("Macro Precision", f"{precision * 100:.2f}%")
    metrics[2].metric("Macro Recall", f"{recall * 100:.2f}%")
    metrics[3].metric("Macro F1-score", f"{f1_score * 100:.2f}%")
    cv_metrics = st.columns(2)
    cv_metrics[0].metric("Mean 5-Fold CV Accuracy", f"{cv_scores.mean() * 100:.2f}%")
    cv_metrics[1].metric("CV Standard Deviation", f"{cv_scores.std() * 100:.2f}%")
    st.caption(f"Holdout test samples: {test_sample_count}. Precision, recall and F1-score use macro averaging across the three species.")
    
    st.divider()
    st.subheader("📊 5-Fold Cross-Validation")
    cv_df = pd.DataFrame({"Fold": [f"Fold {i}" for i in range(1, 6)], "Accuracy (%)": cv_scores * 100})
    fig, ax = plt.subplots(figsize=(10, 5))
    fold_palette = ["#38BDF8", "#60A5FA", "#818CF8", "#A78BFA", "#34D399"]
    bars = ax.bar(cv_df["Fold"], cv_df["Accuracy (%)"], color=fold_palette, width=0.58, edgecolor="#dbeafe", linewidth=0.6)
    ax.set_title("Accuracy Across 5 Validation Folds", fontsize=14, pad=16)
    ax.set_xlabel("Cross-Validation Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    for bar, score in zip(bars, cv_scores * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{score:.1f}%", ha="center", color="#ffffff")
    render_chart(fig)
    st.dataframe(cv_df.round(2), width="stretch")
    
    st.divider()
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")
    
    st.divider()
    st.subheader("🎯 Confusion Matrix")
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[str(label).capitalize() for label in model.classes_])
    from matplotlib.colors import LinearSegmentedColormap
    confusion_cmap = LinearSegmentedColormap.from_list("iris_navy_blue", ["#111827", "#164e78", "#38bdf8"])
    display.plot(cmap=confusion_cmap, values_format="d", ax=ax, colorbar=False)
    ax.set_title("Random Forest Confusion Matrix", fontsize=14, pad=16)
    render_chart(fig)
    
    st.divider()
    st.subheader("🌳 Feature Importance")
    importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    importance_palette = ["#38BDF8", "#60A5FA", "#818CF8", "#A78BFA"]
    ax.barh(importance["Feature"], importance["Importance"], color=importance_palette,
            edgecolor="#dbeafe", linewidth=0.55, height=0.62)
    ax.set_title("Importance of Each Input Feature", fontsize=14, pad=16)
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Feature")
    render_chart(fig)
    st.dataframe(importance.sort_values("Importance", ascending=False).round(4), width="stretch")
    st.success("Model evaluation completed.")

# ---------------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------------
elif page == "📜 Prediction History":
    st.title("📜 Prediction History")
    st.write("Review and download predictions made during the current browser session.")
    st.divider()
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        metrics = st.columns(2)
        metrics[0].metric("Total Predictions", len(history_df))
        metrics[1].metric("Different Species Predicted", history_df["Predicted Species"].nunique())
        
        st.dataframe(history_df, width="stretch")
        st.download_button("⬇️ Download Prediction History", data=csv_bytes(history_df), file_name="iris_prediction_history.csv", mime="text/csv", width="stretch")
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No prediction history is available yet. Make a prediction to see it here.")

# ---------------------------------------------------------
# BULK PREDICTION
# ---------------------------------------------------------
elif page == "📁 Bulk Prediction":
    st.title("📁 Bulk Flower Prediction")
    st.write("Upload a CSV file to predict multiple Iris flowers at once.")
    st.divider()
    
    st.info("The CSV must contain these four columns:\n\n" + ", ".join(FEATURE_COLUMNS))
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 4.2, 1.5]], columns=FEATURE_COLUMNS)
    st.download_button("⬇️ Download Sample CSV", data=csv_bytes(sample), file_name="iris_sample_data.csv", mime="text/csv")
    st.divider()
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="bulk_csv_uploader")
    if uploaded_file is not None:
        if st.session_state.bulk_results_file != uploaded_file.name:
            st.session_state.bulk_results = None
            st.session_state.bulk_results_file = uploaded_file.name
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            if uploaded_data.empty:
                st.error("The uploaded CSV file is empty.")
            else:
                missing = [col for col in FEATURE_COLUMNS if col not in uploaded_data.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                else:
                    numeric_data = uploaded_data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
                    if numeric_data.isnull().any().any():
                        st.error("Some required cells are empty or non-numeric. Please correct the CSV file.")
                    elif not np.isfinite(numeric_data.to_numpy()).all():
                        st.error("The CSV contains invalid numeric values.")
                    else:
                        st.success(f"{len(uploaded_data)} records uploaded successfully.")
                        st.subheader("📋 Uploaded Data")
                        st.dataframe(uploaded_data.head(10), width="stretch")
                        
                        if st.button("🚀 Predict All Flowers", type="primary", width="stretch"):
                            predictions = model.predict(numeric_data)
                            probabilities = model.predict_proba(numeric_data)
                            results = uploaded_data.copy()
                            results["Predicted Species"] = predictions
                            results["Confidence (%)"] = (np.max(probabilities, axis=1) * 100).round(2)
                            st.session_state.bulk_results = results
                            for index, row in enumerate(numeric_data.to_numpy()):
                                save_history(row.tolist(), predictions[index], np.max(probabilities[index]) * 100)
                                
                        if st.session_state.bulk_results is not None:
                            results = st.session_state.bulk_results
                            st.divider()
                            st.subheader("✅ Prediction Results")
                            st.metric("Flowers Processed", len(results))
                            st.dataframe(results, width="stretch")
                            st.download_button("⬇️ Download Prediction Results", data=csv_bytes(results), file_name="iris_bulk_predictions.csv", mime="text/csv", width="stretch")
        except Exception as error:
            st.error(f"Unable to process the CSV file: {error}")

# ---------------------------------------------------------
# ABOUT PROJECT
elif page == "📘 About Project":
    st.title("📘 About the Project")
    st.markdown("""
        <div class="about-hero">
            <h2>🌸 Iris Flower Species Classification</h2>
            <p>An academic machine learning project that classifies an Iris flower into Setosa, Versicolor, or Virginica using four flower measurements. The application combines a Random Forest classifier with an interactive Streamlit dashboard for individual predictions, batch predictions, dataset exploration, and model evaluation.</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Project at a Glance")
    st.markdown(f"""
        <div class="about-stat-grid">
            <div class="about-stat"><div class="about-stat-label">Algorithm</div><div class="about-stat-value">Random Forest Classifier</div></div>
            <div class="about-stat"><div class="about-stat-label">Input Features</div><div class="about-stat-value">{len(FEATURE_COLUMNS)}</div></div>
            <div class="about-stat"><div class="about-stat-label">Flower Classes</div><div class="about-stat-value">{len(SPECIES_NAMES)}</div></div>
            <div class="about-stat"><div class="about-stat-label">Dataset Samples</div><div class="about-stat-value">{len(df)}</div></div>
        </div>
    """, unsafe_allow_html=True)
    st.caption(f"Model artifact: {MODEL_PATH.name}  •  Dataset: Scikit-learn Iris dataset")
    st.divider()

    st.subheader("🎯 Project Purpose")
    purpose_cols = st.columns(2)
    with purpose_cols[0]:
        st.markdown("<div class='about-section'><h4>What the project does</h4><p>Uses sepal and petal measurements to predict the most likely Iris species. Users can enter one flower's measurements or upload a CSV file for batch classification.</p></div>", unsafe_allow_html=True)
    with purpose_cols[1]:
        st.markdown("<div class='about-section'><h4>Learning objectives</h4><p>Understand a classification dataset, explore feature patterns, apply an ensemble learning algorithm, evaluate model performance, and present predictions through a web application.</p></div>", unsafe_allow_html=True)
    st.divider()

    st.subheader("⚙️ How the Application Works")
    workflow = st.columns(4)
    workflow_items = [("01 · Input", "Enter four measurements or upload a CSV file."), ("02 · Prediction", "The saved Random Forest model predicts the species."), ("03 · Results", "View the predicted class and estimated probabilities."), ("04 · Analysis", "Explore dataset charts, evaluation metrics, and history.")]
    for column, (heading, description) in zip(workflow, workflow_items):
        with column:
            st.markdown(f"<div class='about-section'><h4>{heading}</h4><p>{description}</p></div>", unsafe_allow_html=True)
    st.divider()

    st.subheader("🧠 Model Details")
    model_info = st.columns(4)
    model_info[0].metric("Decision Trees", getattr(model, "n_estimators", "N/A"))
    model_info[1].metric("Input Features", len(FEATURE_COLUMNS))
    model_info[2].metric("Output Classes", len(SPECIES_NAMES))
    model_info[3].metric("Model Type", "Ensemble")
    with st.expander("View model configuration"):
        if hasattr(model, "get_params"):
            st.json(model.get_params())
        else:
            st.info("Model configuration is not available for this estimator.")
    st.caption("Evaluation metrics on the Model Performance page are calculated by fitting fresh estimator clones on the Iris dataset. They are not historical scores stored in the model file.")
    st.divider()

    st.subheader("🧰 Technologies Used")
    tech_cols = st.columns(3)
    tech_cards = [("Programming & Data", "Python<br>Pandas<br>NumPy"), ("Machine Learning", "Scikit-learn<br>Random Forest<br>Joblib"), ("Web App & Visualization", "Streamlit<br>Matplotlib<br>Pillow")]
    for column, (heading, details) in zip(tech_cols, tech_cards):
        with column:
            st.markdown(f"<div class='about-section'><h4>{heading}</h4><p>{details}</p></div>", unsafe_allow_html=True)
    st.divider()

    st.subheader("📊 Dataset & Input Features")
    feature_cols = st.columns(2)
    with feature_cols[0]:
        st.markdown("<div class='about-section'><h4>Input measurements</h4><p>• Sepal length (cm)<br>• Sepal width (cm)<br>• Petal length (cm)<br>• Petal width (cm)</p></div>", unsafe_allow_html=True)
    with feature_cols[1]:
        st.markdown("<div class='about-section'><h4>Prediction classes</h4><p>• Setosa<br>• Versicolor<br>• Virginica</p></div>", unsafe_allow_html=True)
    st.divider()

    st.subheader("👥 Project Team")
    st.markdown("""
        <div class='team-credit'>
            <strong style='font-size:1.12rem;'>Collaborative Group Project</strong><br>
            This project is presented as a group submission by students of Invertis University, Bareilly.
            <br><span style='color:#9fb6d2;'>The team collaboratively contributes to project development, interface design, data analysis, documentation, and testing.</span>
        </div>
    """, unsafe_allow_html=True)
    # Names only; GitHub IDs and profile URLs are intentionally not displayed.
    team_members = [
        "Ansh Mishra", "Muskan Gupta", "Ritu Kushwaha", "Ishika Mathur",
        "Pari Pal", "Vibhor Saini", "Unnati Verma", "Hansika Arora",
        "Kashifa Malik", "Palak Saxena", "Abhishek Tiwari", "Somendra",
        "Kashish", "Sonali Mishra"
    ] 
    team_html = "<div class='team-grid'>" + "".join(
        f"<div class='team-member'><span class='team-number'>{index:02d}</span><span>{member}</span></div>"
        for index, member in enumerate(team_members, start=1)
    ) + "</div>"
    st.markdown(team_html, unsafe_allow_html=True)
    st.caption("Academic Institution: Invertis University, Bareilly")
    st.caption("Project presented as a collaborative group submission.")
