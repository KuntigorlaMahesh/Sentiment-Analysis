import streamlit as st
import pickle

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("ML-based Sentiment Analysis")

text = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if text:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        if prediction == "positive":
            st.success("Positive 😊")
        elif prediction == "negative":
            st.error("Negative 😠")
        else:
            st.info("Neutral 😐")
    else:
        st.warning("Please enter some text.")
