# 📝 Product Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)

## 📌 Business Use Case

Customer reviews hold valuable insights for businesses. Sentiment analysis helps:
- Understand user satisfaction
- Identify product issues
- Improve services based on real feedback

## 🧠 Features Used

- **Text (Review)** is converted using TF-IDF Vectorization.
- **Model:** Logistic Regression classifier.

## ⚙️ Pipeline Overview

### Training (`model_training.py`)
- Load data
- Preprocess using TF-IDF
- Train Logistic Regression model
- Evaluate and save model

### App (`app.py`)
- Upload new review data
- Run predictions
- Display and download results

## 🚀 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train model
```bash
python model_training.py
```

### 3. Run Streamlit app
```bash
streamlit run app.py
```

## 🗂 Project Structure
```
sentiment_analysis_project/
├── data/
├── model/
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
└── .github/
```

## 📝 License
This project is licensed under the MIT License.
