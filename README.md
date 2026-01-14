#  Student Performance Prediction – End-to-End ML Project

An **end-to-end Machine Learning project** built using **modular coding principles**, covering the complete ML lifecycle — from **data ingestion** to **model prediction**, with a **Flask web interface** for real-time predictions.

This project follows **industry-standard ML engineering practices** and is suitable for **deployment, interviews, and portfolio use**.

---

## 🚀 Project Highlights

- Modular & scalable code structure
- End-to-end ML pipeline (train + predict)
- Custom logging & exception handling
- Flask web application for predictions
- Clean Python packaging using `src/` and `setup.py`
- Production-ready project layout

---

## 🧠 Problem Statement

Predict a student’s academic performance based on:
- Gender
- Race/Ethnicity
- Parental education level
- Lunch type
- Test preparation course
- Reading score
- Writing score

---

##  Project Architecture (Modular Coding)
student_perfo/
├── logs/
├── notebooks/
|--data/
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ ├── model_trainer.py
│ ├── pipeline/
│ │ ├── train_pipeline.py
│ │ ├── predict_pipeline.py
│ ├── exception.py                                 
│ ├── logger.py
│ ├── utils.py
│ └── __init__.py
├── templates/
│ └── home.html
├── .gitignore   
└──  app.py 
├── README.md
├── requirements.txt
├── setup.py



---

## 🔁 ML Pipeline Workflow

### 1️⃣ Data Ingestion
- Reads raw dataset
- Splits data into train and test sets
- Stores outputs in `artifacts/`

### 2️⃣ Data Transformation
- Handles numerical and categorical features
- Applies encoding and scaling
- Saves the preprocessor object

### 3️⃣ Model Training
- Trains machine learning models
- Evaluates performance
- Saves the best-performing model

### 4️⃣ Prediction Pipeline
- Loads trained model and preprocessor
- Accepts user input from Flask UI
- Returns predicted student performance

---

## 🖥️ Flask Web Application

- User-friendly interface
- Accepts student details
- Returns prediction in real time

Run the application locally:

```bash
python app.py


http://127.0.0.1:5000/predictdata

📦 Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

Flask

HTML/CSS

Logging & Exception Handling


### Using the Web Application

1. Open `http://127.0.0.1:5000/predictdata` in your browser
2. Enter student details:
   - Gender, Race/Ethnicity, Parental Education, Lunch, Test Preparation Course, Reading Score, Writing Score
3. Click **Predict**
4. View predicted academic performance




