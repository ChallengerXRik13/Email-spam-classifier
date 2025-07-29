# 📧 Email & SMS Spam Classifier

A complete end-to-end project to build a **spam detection system** for classifying SMS and email messages as spam or not-spam (ham). Built using Python, NLP techniques, and a Multinomial Naive Bayes model.


## 🔍 Overview

This project demonstrates how to:
- Preprocess and clean text data
- Extract features using **TF-IDF vectorization**
- Train a **Multinomial Naive Bayes** classifier for spam detection
- Save and deploy the model with a simple web UI

---

## 📂 Dataset

- **Source**: [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size**: 5,000+ labeled messages

---

## 🔧 Tech Stack

- 🐍 Python
- 📊 Pandas, Numpy
- 🧠 Scikit-learn
- 🗣️ NLTK (text preprocessing)
- 📈 TF-IDF for vectorization
- 💻 Streamlit for UI 
- 📦 Pickle for model persistence

---

## 📁 Project Structure

```
.
├── app.py                   # Main app file (for deployment)
├── sms-spam-detection.ipynb # Exploratory Data Analysis + Model Training
├── spam.csv                 # Dataset
├── model.pkl                # Trained Naive Bayes model
├── vectorizer.pkl           # TF-IDF vectorizer
├── requirements.txt         # Project dependencies
├── README.md                # This file
└── setup.sh, Procfile       # For deployment
```

---

## 🧪 How It Works

1. **Text Preprocessing**  
   Lowercase, punctuation removal, stopwords removal, stemming

2. **Feature Extraction**  
   Convert cleaned text to numerical vectors using **TF-IDF**

3. **Model Training**  
   Train a Naive Bayes classifier to predict spam or ham

4. **Prediction Interface**  
   Use `app.py` for interactive testing (can be deployed)

---

## 🚀 Run Locally

```bash
git clone https://github.com/ChallengerXRik13/Email-spam-classifier.git
cd Email-spam-classifier
pip install -r requirements.txt
python app.py
```

---

## 🛡 Accuracy

- **Training Accuracy**: ~97%
- **Precision (Spam Class)**: 100%

---

## 📄 License

MIT License – *Free to use with attribution.*
