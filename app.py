import streamlit as st
import pickle

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Function to detect plagiarism
def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

# Streamlit app
st.title("Plagiarism Detection App")
st.write("Enter text below to check for plagiarism:")

# Text input from user
input_text = st.text_area("Enter text here:")

# Button to trigger detection
if st.button("Detect Plagiarism"):
    if input_text.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        detection_result = detect(input_text)
        st.success(f"Result: {detection_result}")
