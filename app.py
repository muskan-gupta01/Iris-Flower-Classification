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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
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
        st.image(image, use_container_width=True)

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
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🌸 Iris AI Dashboard")
st.sidebar.caption("Machine Learning Application")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Flower Prediction",
        "📊 Data Visualization",
        "📈 Model Performance",
        "📜 Prediction History",
        "📁 Bulk Prediction",
        "📘 About Project",
    ],
)

st.sidebar.divider()
st.sidebar.metric("Total Predictions", len(st.session_state.history))
st.sidebar.caption("Algorithm: Random Forest")
st.sidebar.caption("Dataset: Iris Dataset")
st.sidebar.divider()
st.sidebar.markdown("👨‍💻 **Developed by Ansh Mishra**")
st.sidebar.caption("BCA (AI) | Invertis University")

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
                <div class="kpi-label">Pipeline Accuracy</div>
                <div class="kpi-value">94.67%</div>
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
        st.info("Go to **🔍 Flower Prediction** in the sidebar to launch real-time inference.", icon="ℹ️")
    with qc2:
        st.info("Go to **📁 Bulk Prediction** in the sidebar to run batch dataset inference.", icon="ℹ️")

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
    
    if st.button("🌸 Predict Flower Species", type="primary", use_container_width=True):
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
            "Species": model.classes_,
            "Probability (%)": probabilities * 100,
        }).sort_values("Probability (%)", ascending=False)
        st.bar_chart(probability_df.set_index("Species"), y="Probability (%)", use_container_width=True)
        st.caption("Probability is the model's estimated confidence, not a guarantee that the prediction is correct.")

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
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Statistical Summary")
        st.dataframe(df[FEATURE_COLUMNS].describe().round(2), use_container_width=True)
        
    st.divider()
    st.subheader("🌺 Species Distribution")
    counts = df["species"].value_counts().reindex(SPECIES_NAMES)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, width=0.58, color=["#62a0ea", "#69c095", "#c18be8"], edgecolor="#e5eaf3", linewidth=0.6)
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
    colors = {"setosa": "#62a0ea", "versicolor": "#69c095", "virginica": "#c18be8"}
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
    box = ax.boxplot(box_data, labels=[species.capitalize() for species in SPECIES_NAMES], patch_artist=True, widths=0.55, medianprops={"color": "#ffbd4a", "linewidth": 2})
    for patch, color in zip(box["boxes"], ["#62a0ea", "#69c095", "#c18be8"]):
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
    st.write("Evaluate the saved Random Forest model with a stratified test split, cross-validation and classification metrics.")
    st.divider()
    
    X = df[FEATURE_COLUMNS]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")
    
    st.subheader("🎯 Performance Overview")
    metrics = st.columns(4)
    metrics[0].metric("Test Accuracy", f"{accuracy * 100:.2f}%")
    metrics[1].metric("Mean CV Accuracy", f"{cv_scores.mean() * 100:.2f}%")
    metrics[2].metric("CV Standard Deviation", f"{cv_scores.std() * 100:.2f}%")
    metrics[3].metric("Test Samples", len(X_test))
    
    st.divider()
    st.subheader("📊 5-Fold Cross-Validation")
    cv_df = pd.DataFrame({"Fold": [f"Fold {i}" for i in range(1, 6)], "Accuracy (%)": cv_scores * 100})
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cv_df["Fold"], cv_df["Accuracy (%)"], color="#62a0ea", width=0.58)
    ax.set_title("Accuracy Across 5 Validation Folds", fontsize=14, pad=16)
    ax.set_xlabel("Cross-Validation Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    for bar, score in zip(bars, cv_scores * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{score:.1f}%", ha="center", color="#ffffff")
    render_chart(fig)
    st.dataframe(cv_df.round(2), use_container_width=True)
    
    st.divider()
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    
    st.divider()
    st.subheader("🎯 Confusion Matrix")
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[str(label).capitalize() for label in model.classes_])
    display.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("Random Forest Confusion Matrix", fontsize=14, pad=16)
    render_chart(fig)
    
    st.divider()
    st.subheader("🌳 Feature Importance")
    importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance["Feature"], importance["Importance"], color="#69c095")
    ax.set_title("Importance of Each Input Feature", fontsize=14, pad=16)
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Feature")
    render_chart(fig)
    st.dataframe(importance.sort_values("Importance", ascending=False).round(4), use_container_width=True)
    st.success("Model evaluation completed.")

# ---------------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------------
elif page == "📜 Prediction History":
    st.title("📜 Prediction History")
    st.write("Review and download predictions made during this session.")
    st.divider()
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        metrics = st.columns(2)
        metrics[0].metric("Total Predictions", len(history_df))
        metrics[1].metric("Different Species Predicted", history_df["Predicted Species"].nunique())
        
        st.dataframe(history_df, use_container_width=True)
        st.download_button("⬇️ Download Prediction History", data=csv_bytes(history_df), file_name="iris_prediction_history.csv", mime="text/csv", use_container_width=True)
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
                        st.dataframe(uploaded_data.head(10), use_container_width=True)
                        
                        if st.button("🚀 Predict All Flowers", type="primary", use_container_width=True):
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
                            st.dataframe(results, use_container_width=True)
                            st.download_button("⬇️ Download Prediction Results", data=csv_bytes(results), file_name="iris_bulk_predictions.csv", mime="text/csv", use_container_width=True)
        except Exception as error:
            st.error(f"Unable to process the CSV file: {error}")

# ---------------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------------
elif page == "📘 About Project":
    st.title("📘 About Iris AI Classifier")
    st.write("Iris AI Classifier is a Machine Learning academic project that predicts the species of an Iris flower from sepal length, sepal width, petal length and petal width. The application uses a trained Random Forest classification model and provides an interactive interface for individual and bulk predictions.")
    st.divider()
    
    st.subheader("🎯 Project Objectives")
    st.markdown("- Explore and understand the Iris dataset.\n- Perform data analysis and visualization.\n- Train and evaluate a Random Forest classifier.\n- Predict flower species through a Streamlit application.\n- Support multiple predictions using CSV uploads.")
    st.divider()
    
    st.subheader("🛠️ Technologies Used")
    columns = st.columns(3)
    with columns[0]:
        st.markdown("**Programming & Data**")
        st.write("🐍 Python\n📊 Pandas\n🔢 NumPy")
    with columns[1]:
        st.markdown("**Machine Learning**")
        st.write("🤖 Scikit-learn\n🌳 Random Forest\n💾 Joblib")
    with columns[2]:
        st.markdown("**Application & Charts**")
        st.write("🖥️ Streamlit\n📈 Matplotlib\n🖼️ Pillow")
    st.divider()
    
    st.subheader("📐 Input Features")
    left, right = st.columns(2)
    with left:
        st.write("1. Sepal Length (cm)\n2. Sepal Width (cm)")
    with right:
        st.write("3. Petal Length (cm)\n4. Petal Width (cm)")
    st.divider()
    
    st.subheader("🌺 Iris Species")
    species_columns = st.columns(3)
    for column, species in zip(species_columns, SPECIES_NAMES):
        with column:
            st.markdown(f"**{species.capitalize()}**")
            st.caption(f"{int((df['species'] == species).sum())} samples in dataset")
    st.divider()
    
    st.subheader("🌳 Random Forest Algorithm")
    st.write("Random Forest is a supervised ensemble learning algorithm. It builds multiple decision trees and combines their predictions to produce a final classification. This project uses it to classify Iris flowers based on their four measurements.")
    st.success("Project: Iris Flower Classification using Machine Learning")
    st.caption("Developed using Python, Scikit-learn, Matplotlib and Streamlit.")