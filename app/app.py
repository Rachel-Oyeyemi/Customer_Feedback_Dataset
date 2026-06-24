"""
Streamlit Application for Customer Feedback Sentiment Analysis
============================================================

This Streamlit app provides an interactive interface for exploring the
sentiment analysis project.  Users can view an overview of the project,
inspect exploratory analysis visuals, examine model performance, make
sentiment predictions on new text, and read business insights.

To run the app locally, execute:

```bash
streamlit run app/app.py
```
"""

import json
from pathlib import Path

import streamlit as st

from src.predict import load_model, load_vectorizer, predict_sentiment

# Load models and vectoriser
BASELINE_MODEL_PATH = Path("models/baseline_model.joblib")
ADVANCED_MODEL_PATH = Path("models/advanced_model.joblib")
VECTORIZER_PATH = Path("data/processed/vectorizer.joblib")

@st.cache_resource
def get_models():
    """Load the baseline and advanced models and the vectoriser once and cache them."""
    baseline_model = load_model(str(BASELINE_MODEL_PATH))
    advanced_model = load_model(str(ADVANCED_MODEL_PATH))
    vectorizer = load_vectorizer(str(VECTORIZER_PATH))
    return baseline_model, advanced_model, vectorizer


def load_metrics():
    """Load evaluation metrics for the baseline and advanced models."""
    baseline_report = json.load(open("models/baseline_evaluation.json"))
    advanced_report = json.load(open("models/advanced_evaluation.json"))
    return baseline_report, advanced_report


def show_overview():
    st.header("Project Overview")
    st.write(
        """
        This project analyses customer feedback to automatically classify
        comments into **Positive**, **Neutral**, and **Negative** sentiments.
        The goal is to help businesses monitor customer opinions at scale,
        identify areas for improvement, and highlight strengths.

        The dataset used here is a synthetic sample created for demonstration
        purposes.  It contains a balanced set of comments across the three
        sentiment categories.  The workflow includes data preprocessing,
        feature engineering with TF‑IDF, training baseline and advanced
        models, and evaluating their performance.
        """
    )


def show_prediction_page(baseline_model, advanced_model, vectorizer):
    st.header("Prediction Interface")
    st.write("Enter customer feedback below to predict its sentiment.")
    text_input = st.text_area("Feedback text", "")
    model_choice = st.selectbox("Select model", ("Random Forest (Advanced)", "Logistic Regression (Baseline)"))
    if st.button("Predict Sentiment"):
        if not text_input.strip():
            st.warning("Please enter some text for prediction.")
        else:
            model = advanced_model if model_choice.startswith("Random") else baseline_model
            prediction = predict_sentiment([text_input], model, vectorizer)[0]
            st.success(f"Predicted Sentiment: {prediction}")


def show_model_performance(baseline_report, advanced_report):
    st.header("Model Performance")
    st.write("Comparison of baseline and advanced model metrics:")
    perf_table = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1‑Score"],
        "Logistic Regression": [
            round(baseline_report["accuracy"], 3),
            round(baseline_report["precision"], 3),
            round(baseline_report["recall"], 3),
            round(baseline_report["f1"], 3),
        ],
        "Random Forest": [
            round(advanced_report["accuracy"], 3),
            round(advanced_report["precision"], 3),
            round(advanced_report["recall"], 3),
            round(advanced_report["f1"], 3),
        ],
    }
    st.table(perf_table)
    st.write("Confusion matrices:")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression")
        st.json({"Confusion Matrix": baseline_report["confusion_matrix"]})
    with col2:
        st.subheader("Random Forest")
        st.json({"Confusion Matrix": advanced_report["confusion_matrix"]})


def show_visualizations():
    st.header("Exploratory Analysis Visualizations")
    st.write("Key visualizations from the exploratory data analysis:")
    # Display images saved in visuals folder
    visuals_dir = Path("visuals")
    images = [
        ("Sentiment Distribution", visuals_dir / "sentiment_distribution.png"),
        ("Text Length Distribution", visuals_dir / "text_length_distribution.png"),
        ("Top Words per Sentiment", visuals_dir / "top_words_per_sentiment.png"),
    ]
    for title, img_path in images:
        if img_path.exists():
            st.subheader(title)
            st.image(str(img_path))
        else:
            st.warning(f"Image {img_path} not found.")


def show_business_insights():
    st.header("Business Insights & Recommendations")
    # Read the recommendations markdown file and display as text
    insights_path = Path("BUSINESS_RECOMMENDATIONS.md")
    if insights_path.exists():
        with open(insights_path, "r", encoding="utf‑         ") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.warning("Recommendations file not found.")


def show_about():
    st.header("About")
    st.write(
        """
        **Author:** Your Name Here  

        This Streamlit app is part of a portfolio project demonstrating
        end‑to‑end sentiment analysis on customer feedback.  The project
        includes data ingestion, preprocessing, exploratory analysis,
        modelling, evaluation, and deployment of an interactive web
        application.
        """
    )


def main():
    st.title("Customer Feedback Sentiment Analysis")
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Home",
            "Project Overview",
            "Prediction Interface",
            "Model Performance",
            "Visualizations",
            "Business Insights",
            "About",
        ],
    )
    baseline_model, advanced_model, vectorizer = get_models()
    baseline_report, advanced_report = load_metrics()

    if page == "Home":
        st.write(
            "Welcome to the Customer Feedback Sentiment Analysis app. Use the side bar to navigate through the project sections."
        )
    elif page == "Project Overview":
        show_overview()
    elif page == "Prediction Interface":
        show_prediction_page(baseline_model, advanced_model, vectorizer)
    elif page == "Model Performance":
        show_model_performance(baseline_report, advanced_report)
    elif page == "Visualizations":
        show_visualizations()
    elif page == "Business Insights":
        show_business_insights()
    elif page == "About":
        show_about()


if __name__ == "__main__":
    main()
