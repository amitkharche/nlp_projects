"""
Streamlit app for sentiment prediction on product reviews.
"""

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="📝 Sentiment Analysis App", layout="wide")
st.title("📝 Product Review Sentiment Analyzer")

@st.cache_resource
def load_model():
    try:
        with open("model/sentiment_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def make_predictions(df, model):
    df["Predicted Sentiment"] = model.predict(df["Review"])
    return df

def main():
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Review' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "Review" not in df.columns:
            st.error("Uploaded file must contain a 'Review' column.")
            return

        st.write("### Uploaded Reviews", df.head())

        model = load_model()
        if model:
            result_df = make_predictions(df, model)
            st.write("### Prediction Results", result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "sentiment_predictions.csv", "text/csv")

if __name__ == "__main__":
    main()
