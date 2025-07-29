import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Download required NLTK resources (only the first time)
nltk.download('stopwords')

# Load trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    text = nltk.word_tokenize(text)
    # Remove special characters and stopwords, apply stemming
    y = []
    for word in text:
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            y.append(ps.stem(word))
    return " ".join(y)

# Streamlit UI
st.set_page_config(page_title="Email/SMS Spam Classifier", layout="centered")
st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message below:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a valid message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not a Spam Message")
