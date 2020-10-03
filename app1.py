import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('/content/drive/My Drive/Machine learning/csv files/sentiment_review.csv')
x = df.iloc[:,0].values
y = df.iloc[:,1].values
st.title("Sentiment Analysis on Review")
st.subheader('TFIFD Vectorizer')
st.write('This project is based on Naive Bayes Classifier')

text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
text_model.fit(x,y)
message = st.text_area("Enter Text","type here...")
op = text_model.predict([message])
if st.button("Predict"):
  st.title(op)