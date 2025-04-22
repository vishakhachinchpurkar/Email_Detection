import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample model training — you can load pre-trained model too
df = pd.read_csv('C:/Users/vchin/OneDrive/Desktop/DSBDA_PROJECT/data/spam.csv', encoding='latin-1')
df = df.rename(columns={"v1": "Category", "v2": "Message"})
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X = df['Message']
y = df['Spam']

# Train a pipeline (text vectorizer + Naive Bayes)
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Email Spam Detector", layout="centered")
st.title("📧 Email Spam Detection")
st.write("Enter an email message to check if it's spam or not.")

# Text input from user
email_input = st.text_area("Your Email Message", height=150)

if st.button("Detect"):
    if email_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([email_input])[0]
        if prediction == 1:
            st.error("🚨 This is a **Spam Email**!")
        else:
            st.success("✅ This is a **Ham Email**!")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & scikit-learn")
