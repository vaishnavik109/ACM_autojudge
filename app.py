import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack

# Load trained models
clf = joblib.load("classifier.pkl")
reg = joblib.load("regressor.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("AutoJudge – Programming Problem Difficulty Predictor")

desc = st.text_area("Problem Description")
inp = st.text_area("Input Description")
out = st.text_area("Output Description")

def extract_features(text):
    keywords = ["dp", "graph", "tree", "recursion", "greedy", "bitmask"]
    feats = []
    feats.append(len(text))
    feats.append(len(re.findall(r"[+\-*/=<>()]", text)))
    for kw in keywords:
        feats.append(text.lower().count(kw))
    return np.array(feats).reshape(1, -1)

if st.button("Predict"):
    full_text = desc + " " + inp + " " + out
    X_text = tfidf.transform([full_text])
    X_feat = extract_features(full_text)
    X_final = hstack([X_text, X_feat])

    st.success(f"Predicted Difficulty Class: {clf.predict(X_final)[0]}")
    st.success(f"Predicted Difficulty Score: {reg.predict(X_final)[0]:.2f}")
