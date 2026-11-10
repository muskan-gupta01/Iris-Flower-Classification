# 🌸 Iris Flower Classification and Prediction System

<p align="center">
  <b>An Interactive Machine Learning Web Application for Iris Flower Species Classification</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
  <img src="https://img.shields.io/badge/Project-Academic-orange" alt="Academic Project">
</p>

---

## 📌 Project Overview

The **Iris Flower Classification and Prediction System** is a Machine Learning-based web application developed to classify Iris flower species using four important flower measurements.

The system uses the **Random Forest Classifier** to predict the species based on the following input features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The application provides an interactive interface built with Streamlit, allowing users to perform individual predictions, upload CSV files for bulk predictions, explore dataset visualizations, review prediction history, and understand model performance.

### 🎯 Project Objectives

- Understand the fundamentals of Machine Learning.
- Perform data preprocessing and exploratory data analysis.
- Train and evaluate a Random Forest classification model.
- Develop an interactive prediction application.
- Implement single and bulk prediction functionality.
- Visualize dataset patterns and model performance.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🏠 Home Dashboard | Provides an overview of the project and its capabilities. |
| 🌺 Flower Prediction | Predicts Iris flower species using four input measurements. |
| 📂 Bulk CSV Prediction | Supports multiple predictions through CSV file uploads. |
| 📊 Data Visualization | Displays charts and visual insights from the Iris dataset. |
| 📜 Prediction History | Allows users to review previous predictions. |
| 🤖 Model Performance | Displays information about the trained ML model and its evaluation. |
| ℹ️ About Project | Provides project information and team details. |

---

## 🌼 Iris Flower Species

The system classifies flowers into three categories:

1. **Iris Setosa**
2. **Iris Versicolor**
3. **Iris Virginica**

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning and model evaluation |
| Streamlit | Interactive web application |
| Joblib | Saving and loading the trained model |
| Jupyter Notebook | Model development and experimentation |
| Git & GitHub | Version control and collaboration |

---

## 🧠 Machine Learning Workflow

The project follows a structured Machine Learning workflow:

1. **Data Collection:** Load the Iris dataset.
2. **Data Exploration:** Understand dataset features and distributions.
3. **Data Preprocessing:** Prepare the input features and target variable.
4. **Model Training:** Train the Random Forest Classifier.
5. **Model Evaluation:** Evaluate the model using classification metrics.
6. **Model Serialization:** Save the trained model using Joblib.
7. **Application Development:** Integrate the trained model with Streamlit.
8. **Prediction:** Generate predictions for individual inputs and CSV datasets.

### Algorithm Used

**Random Forest Classifier**

Random Forest is a supervised Machine Learning algorithm that combines multiple decision trees to generate predictions.

It is used in this project to classify Iris flowers into their respective species based on their measurements.

---

## 📁 Project Structure

```text
Iris-MLOps-Project/
│
├── data/
│   └── iris.csv
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── .ipynb_checkpoints/
│
├── app.py
│
├── iris_random_forest_model.joblib
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### File Description

- `data/iris.csv` – Dataset used for training and analysis.
- `notebooks/iris_classification.ipynb` – Jupyter Notebook containing model development and experimentation.
- `app.py` – Main Streamlit application.
- `iris_random_forest_model.joblib` – Saved trained Machine Learning model.
- `requirements.txt` – Required Python dependencies.
- `README.md` – Project documentation.

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/muskan-gupta01/Iris-Flower-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Iris-Flower-Classification
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## 📊 Model Evaluation

The Random Forest model is evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help assess the classification performance of the trained model.

---

## 🚀 Application Modules

### 1. Home Dashboard
Provides an introduction to the Iris Classification System.

### 2. Flower Prediction
Accepts four flower measurements and predicts the corresponding Iris species.

### 3. Bulk CSV Prediction
Allows users to upload a CSV file containing multiple flower records and generate predictions.

### 4. Prediction History
Provides access to previously generated predictions.

### 5. Data Visualization
Displays graphical representations of the dataset to help understand relationships between flower measurements and species.

### 6. Model Performance
Presents information about the Random Forest model and its evaluation.

---

## 👥 Project Team

**Project Title:** Iris Flower Classification and Prediction System Using Machine Learning

**Team Members:**

1. Ansh Mishra
2. Muskan Gupta
3. Ritu Kushwaha
4. Ishika Mathur
5. Pari Pal
6. Vibhor Saini
7. Unnati Verma
8. Hansika Arora
9. Kashifa Malik
10. Palak Saxena
11. Abhishek Tiwari
12. Somendra
13. Kashish
14. Sonali Mishra

**Team Lead:** Muskan Gupta

---

## 🎓 Academic Information

| Details | Information |
|---|---|
| Institution | Invertis University, Bareilly |
| Department | Department of Computer Applications |
| Course | Bachelor of Computer Applications (Artificial Intelligence) |
| Semester | 3rd Semester |
| Section | A |
| Academic Session | 2025–2028 |
| Project Type | Collaborative Academic Group Project |

---

## 🔮 Future Enhancements

- Implement additional Machine Learning algorithms.
- Improve model evaluation and comparison.
- Add more interactive data visualizations.
- Enhance the user interface and user experience.
- Deploy the application on a cloud platform.
- Introduce additional features for model monitoring.

---

## 📄 License

This project is developed for academic and educational purposes.

---

<p align="center">
  <b>Developed as a Collaborative Machine Learning Project</b>
  <br>
  Department of Computer Applications
  <br>
  Invertis University, Bareilly
  <br><br>
  🌸 <i>Learning Machine Learning Through Practical Implementation.</i>
</p>