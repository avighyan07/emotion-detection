import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from nltk.stem import PorterStemmer
import re
import nltk

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load the models and other necessary files
lg = pickle.load(open("logistic_regression.pkl", "rb"))
lb = pickle.load(open("label_encoder.pkl", "rb"))
tfidfvectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))
model = load_model("model.h5")
vocab_info = pickle.load(open("vocab_info.pkl", "rb"))

# Predefined constants
vocab_size = vocab_info['vocab_size']
max_len = vocab_info['max_len']

# Text preprocessing function
def clean_data(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predictive_system_dl(sentence):
    text = clean_data(sentence)
    one_hot_word = [one_hot(text, n=vocab_size)]
    pad_seq = pad_sequences(sequences=one_hot_word, maxlen=max_len, padding="pre")
    return pad_seq

def predict_emotion(sentence, model_type='ml'):
    cleaned_text = clean_data(sentence)
    if model_type == 'ml':
        # Machine Learning prediction
        input_vectorizer = tfidfvectorizer.transform([cleaned_text])
        predicted_label = lg.predict(input_vectorizer)[0]
        predicted_emotion = lb.inverse_transform([predicted_label])[0]
    elif model_type == 'dl':
        # Deep Learning prediction
        sentence_seq = predictive_system_dl(sentence)
        predicted_label = np.argmax(model.predict(sentence_seq), axis=1)[0]
        predicted_emotion = lb.inverse_transform([predicted_label])[0]
    return predicted_emotion

# Streamlit app interface
st.title("Emotion Prediction App")
st.write("Predict the emotion of the given sentence using Machine Learning or Deep Learning")

# User input
sentence = st.text_input("Enter a sentence to analyze:", "")

# Model selection
model_type = st.selectbox("Choose a model:", ("Machine Learning (Logistic Regression)", "Deep Learning (LSTM)"))

if st.button("Predict Emotion"):
    if sentence.strip():
        if model_type == "Machine Learning (Logistic Regression)":
            emotion = predict_emotion(sentence, model_type='ml')
        elif model_type == "Deep Learning (LSTM)":
            emotion = predict_emotion(sentence, model_type='dl')
        
        st.write(f"The predicted emotion is: **{emotion}**")
    else:
        st.write("Please enter a valid sentence.")

