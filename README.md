# 🌸 Iris Flower Classification System

<p align="center">
  <b>An Interactive Machine Learning Web Application for Iris Flower Species Prediction</b>
  <br><br>
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Library-Scikit--learn-orange">
  <img src="https://img.shields.io/badge/Project-Academic-success">
</p>

---

## 📌 About the Project

The **Iris Flower Classification System** is a Machine Learning-based web application that predicts the species of an Iris flower using its physical measurements.

The project uses a **Random Forest Classifier**, a supervised Machine Learning algorithm, to classify flowers into three different species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

Users can enter flower measurements manually or upload a CSV file to predict multiple records at once.

The application is developed using Python and Streamlit. It provides an interactive dashboard with prediction results, probability scores, data visualizations, model performance metrics, prediction history, and downloadable reports.

**GitHub Repository:**
https://github.com/muskan-gupta01/Iris-Flower-Classification

---

## 🎯 Project Objectives

The main objectives of this project are:

* Understand the basic Machine Learning workflow.
* Explore and analyze the Iris dataset.
* Perform data preparation and feature analysis.
* Train a classification model using Random Forest.
* Evaluate model performance using different metrics.
* Develop an interactive web application using Streamlit.
* Provide single and bulk flower predictions.
* Display results through visualizations and downloadable reports.

---

## 📊 Dataset Information

This project uses the **Iris Dataset** provided by Scikit-learn.

It is a commonly used dataset for understanding supervised Machine Learning and classification problems.

| Dataset Information | Details                   |
| ------------------- | ------------------------- |
| Total Records       | 150                       |
| Input Features      | 4                         |
| Target Classes      | 3                         |
| Problem Type        | Classification            |
| Dataset Source      | Scikit-learn Iris Dataset |

### Input Features

The model uses four measurements:

| Feature           | Description         |
| ----------------- | ------------------- |
| Sepal Length (cm) | Length of the sepal |
| Sepal Width (cm)  | Width of the sepal  |
| Petal Length (cm) | Length of the petal |
| Petal Width (cm)  | Width of the petal  |

### Target Classes

The model predicts one of the following species:

1. Setosa
2. Versicolor
3. Virginica

---

## 🤖 Machine Learning Model

The project uses the **Random Forest Classifier**, a supervised Machine Learning algorithm used for classification tasks.

Random Forest combines multiple Decision Trees to generate a final prediction.

### Model Configuration

| Parameter            | Value                              |
| -------------------- | ---------------------------------- |
| Algorithm            | Random Forest Classifier           |
| Number of Estimators | 100                                |
| Test Size            | 20%                                |
| Random State         | 42                                 |
| Cross-Validation     | 5-Fold Stratified Cross-Validation |

### Model Training Workflow

1. Load the Iris dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Train the Random Forest Classifier.
5. Generate predictions using test data.
6. Evaluate the model performance.
7. Perform 5-fold stratified cross-validation.
8. Save the trained model using Joblib.

**Saved Model:** `iris_random_forest_model.joblib`

---

## 📈 Model Evaluation

The model is evaluated using the following performance metrics:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* 5-Fold Cross-Validation

The application also provides visual representations of model performance, including confusion matrix and feature importance.

---

## ✨ Application Features

The application provides an interactive dashboard with the following sections:

### 1. 🏠 Home

* Introduction to the project.
* Overview of the classification system.
* Information about the three Iris species.
* Navigation to different application sections.

### 2. 🔍 Flower Prediction

* Enter four flower measurements.
* Predict the Iris flower species.
* View prediction probability.
* Display class-wise prediction probabilities.
* View the predicted flower image.

### 3. 📊 Data Visualization

* Dataset preview.
* Species distribution.
* Feature histograms.
* Box plots.
* Correlation heatmap.
* PCA visualization.

### 4. 📈 Model Performance

* Model accuracy.
* Cross-validation results.
* Classification report.
* Confusion matrix.
* Feature importance visualization.

### 5. 📜 Prediction History

* View previous predictions during the current session.
* Check flower measurements.
* View predicted species and probability.
* Download prediction history as a CSV file.
* Clear prediction history.

### 6. 📁 Bulk Prediction

* Upload a CSV file.
* Validate uploaded data.
* Preview input records.
* Generate predictions for multiple flowers.
* View prediction results and probabilities.
* Download prediction results as a CSV file.

### 7. 📚 About Project

* Project introduction.
* Technologies used.
* Input feature information.
* Explanation of the Random Forest algorithm.

---

## 🛠️ Technologies Used

| Technology       | Purpose                               |
| ---------------- | ------------------------------------- |
| Python           | Main Programming Language             |
| Pandas           | Data Manipulation                     |
| NumPy            | Numerical Operations                  |
| Matplotlib       | Data Visualization                    |
| Seaborn          | Statistical Visualization             |
| Scikit-learn     | Machine Learning                      |
| Random Forest    | Classification Algorithm              |
| Streamlit        | Web Application Development           |
| Joblib           | Model Saving and Loading              |
| Jupyter Notebook | Model Development and Experimentation |

---

## 📂 Project Structure

```text
Iris-Flower-Classification/
│
├── notebooks/
│   └── iris_classification.ipynb
│
├── app.py
│
├── iris_random_forest_model.joblib
│
├── requirements.txt
│
└── README.md
```

### File Description

| File                                  | Description                                      |
| ------------------------------------- | ------------------------------------------------ |
| `notebooks/iris_classification.ipynb` | Dataset analysis, model training, and evaluation |
| `app.py`                              | Streamlit web application source code            |
| `iris_random_forest_model.joblib`     | Saved trained Random Forest model                |
| `requirements.txt`                    | Required Python libraries                        |
| `README.md`                           | Project documentation                            |

---

## ⚙️ Installation and Setup

Follow these steps to run the project on your local system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/muskan-gupta01/Iris-Flower-Classification.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Iris-Flower-Classification
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 📁 Bulk Prediction – CSV Format

The Bulk Prediction feature allows users to generate predictions for multiple flower records using a CSV file.

The uploaded CSV file must contain the following exact column names:

```csv
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
6.3,3.3,6.0,2.5
5.8,2.7,5.1,1.9
7.1,3.0,5.9,2.1
```

### How to Use Bulk Prediction

1. Run the Streamlit application.
2. Navigate to the **Bulk Prediction** section.
3. Upload your CSV file.
4. Preview the uploaded dataset.
5. Click on **Predict All Flowers**.
6. View the prediction results.
7. Download the results as a CSV file.

**Note:** All four required columns must be present and contain valid numeric values.

---

## 🚀 Future Improvements

The following features can be considered for future development:

* Deploy the application on a cloud platform.
* Implement automated model retraining.
* Add model monitoring.
* Explore additional Machine Learning algorithms.
* Improve prediction reports.
* Enhance application performance and user experience.

---

## 🎓 Academic Project Information

| Details    | Information                       |
| ---------- | --------------------------------- |
| University | Invertis University, Bareilly     |
| Department | Computer Applications             |
| Course     | BCA (Artificial Intelligence)     |
| Semester   | 3rd Semester                      |
| Section    | A                                 |
| Project    | Iris Flower Classification System |
| Domain     | Machine Learning                  |

---

## 👥 Team Members – GitHub Profiles

The following students are associated with this academic project:

| S. No. | Team Member     | GitHub Profile                                                        |
| ------ | --------------- | --------------------------------------------------------------------- |
| 1      | Muskan Gupta    | [muskan-gupta01](https://github.com/muskan-gupta01)                   |
| 2      | Ansh Mishra     | [AnshAI7](https://github.com/AnshAI7)                                 |
| 3      | Ritu Kushwaha   | [mizuritu](https://github.com/mizuritu)                               |
| 4      | Ishika Mathur   | [ishikamathur-coder](https://github.com/ishikamathur-coder)           |
| 5      | Pari Pal        | [paripal1812](https://github.com/paripal1812)                         |
| 6      | Vibhor Saini    | [sainivibhor235-dotcom](https://github.com/sainivibhor235-dotcom)     |
| 7      | Unnati Verma    | [vunnati163-dotcom](https://github.com/vunnati163-dotcom)             |
| 8      | Hansika Arora   | [hansikaarora726-ship-it](https://github.com/hansikaarora726-ship-it) |
| 9      | Kashifa Malik   | [Kashifamalik](https://github.com/Kashifamalik)                       |
| 10     | Palak Saxena    | [palak7002](https://github.com/palak7002)                             |
| 11     | Abhishek Tiwari | [abhitiwari639273-cell](https://github.com/abhitiwari639273-cell)     |
| 12     | Somendra        | [Somendra47](https://github.com/Somendra47)                           |
| 13     | Kashish         | [Kashish320](https://github.com/Kashish320)                           |
| 14     | Sonali Mishra   | [sonalimishra](https://github.com/sonalimishra)                       |

---

## 📝 Conclusion

The **Iris Flower Classification System** demonstrates the practical application of Machine Learning in solving a classification problem.

By combining dataset analysis, Random Forest classification, model evaluation, and Streamlit application development, this project provides an interactive platform for predicting Iris flower species.

It also provides practical learning experience in Python programming, data analysis, Machine Learning model development, and web application development.

---

<p align="center">
  <b>🌸 Iris Flower Classification System</b>
  <br><br>
  Machine Learning | Data Analysis | Streamlit
  <br><br>
  Developed as an academic project at Invertis University, Bareilly.
</p>
