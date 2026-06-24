<!-- README.md generated for the Customer Feedback Sentiment Analysis project -->

# Customer Feedback Sentiment Analysis

## Project Overview

This project builds an end‑to‑end sentiment‑analysis pipeline to classify
customer feedback into **Positive**, **Neutral**, or **Negative** categories.
The goal is to automate the understanding of customer sentiment, enabling
businesses to improve products, services, and overall customer experience.
The project demonstrates best practices in data ingestion, preprocessing,
exploratory analysis, machine‑learning modelling, evaluation, and deployment
via an interactive Streamlit application.

## Business Problem

Organisations receive large volumes of unstructured feedback through
surveys, reviews, support tickets, and social media.  Manually analysing
this feedback is labour intensive and subjective.  Automated sentiment
analysis helps stakeholders quickly gauge customer emotions and identify
areas requiring attention.  By classifying feedback into positive,
neutral, and negative sentiment, companies can prioritise improvements
and highlight successes.

## Dataset Source

The dataset originates from a Kaggle customer feedback dataset.  Due to
download restrictions, a synthetic dataset was generated for this
portfolio project.  The synthetic data contains 30 unique customer
comments evenly distributed across the three sentiment categories.  Each
row includes a `Text` field (the feedback) and a `Sentiment` label.

## Methodology

1. **Data Acquisition:** A script (`src/download_data.py`) handles data
   download.  For the synthetic dataset, generation occurred via a
   separate script outside the repository.
2. **Data Preprocessing:** The preprocessing script
   (`src/preprocess.py`) removes duplicate rows, handles missing values,
   and trims whitespace.  The cleaned data is saved in `data/processed/`.
3. **Feature Engineering:** TF‑IDF vectorisation converts text into
   numerical features.  The fitted vectoriser, features matrix, and
   labels array are saved for reuse (`src/feature_engineering.py`).
4. **Model Training:** Two models are trained on the TF‑IDF features:
   - **Logistic Regression (baseline)**
   - **Random Forest (advanced)**
   The training script (`src/train_model.py`) splits the data into
   train/test sets, fits each model, evaluates on the test set, and
   saves the trained models.
5. **Evaluation:** The evaluation script (`src/evaluate_model.py`)
   computes accuracy, precision, recall, F1‑score, and confusion
   matrices for each model.  Results are stored in JSON files and
   summarised in `MODEL_EVALUATION.md` and `MODEL_COMPARISON.md`.
6. **Exploratory Data Analysis (EDA):** A markdown report
   (`EDA_REPORT.md`) details missing values, duplicate analysis,
   distribution of sentiments and text lengths, and frequent words per
   sentiment.  Custom images are generated using the Pillow library and
   stored in `visuals/`.
7. **Business Insights:** Insights and recommendations are compiled in
   `BUSINESS_RECOMMENDATIONS.md`, connecting analytical findings to
   business actions.
8. **Streamlit Application:** The `app/app.py` script builds an
   interactive web application that allows users to explore the project
   overview, visualisations, model performance, business insights, and
   perform sentiment predictions on new text.
9. **Documentation & Presentation:** Additional markdown files (e.g.,
   `PROJECT_CHARTER.md`) provide context, objectives, and structure.  An
   executive presentation is available in the `presentation/` folder.

## Exploratory Data Analysis

The EDA revealed that the cleaned dataset has no missing values and is
balanced across sentiment classes.  Customer comments are typically
short, ranging from 4 to 10 words.  Top words for positive sentiment
include *amazing*, *love*, and *great*, while negative comments feature
terms like *disappointed*, *poor*, and *unhappy*.  Detailed charts are
available in the `visuals/` folder and described in `EDA_REPORT.md`.

## Feature Engineering

TF‑IDF vectorisation was chosen to transform text into numerical
representations.  This approach preserves important words and phrases by
assigning higher weights to terms that are more distinctive within each
document while down‑weighting common words.

## Modeling

| Model               | Accuracy | Precision | Recall | F1‑Score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | 0.867    | 0.879     | 0.867  | 0.868    |
| Random Forest       | **0.900**| **0.911** | **0.900**| **0.899** |

The Random Forest model outperformed the logistic regression baseline on
all metrics.  Confusion matrices show that the Random Forest makes fewer
misclassifications, particularly between negative and neutral classes.
Detailed evaluation results are documented in `MODEL_EVALUATION.md`.

## Results & Business Impact

Deploying the sentiment classifier enables automated analysis of customer
feedback at scale.  Key benefits include:

- **Real‑Time Monitoring:** Quickly identify spikes in negative sentiment
  and respond to issues.
- **Prioritised Improvements:** Focus on problems most frequently
  mentioned in negative feedback (e.g., quality or shipping issues).
- **Marketing Insights:** Highlight strengths emphasised in positive
  feedback for promotional material and messaging.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rachel-Oyeyemi/Customer_Feedback_Dataset.git
   cd Customer_Feedback_Dataset
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download or generate data:**
   Place the raw CSV in `data/raw/` or use `src/download_data.py` (for
   demonstration, the synthetic dataset is already included).
4. **Preprocess and feature engineering:**
   ```bash
   python src/preprocess.py --input data/raw/sentiment_analysis.csv --output data/processed/sentiment_analysis_clean.csv
   python src/feature_engineering.py --input data/processed/sentiment_analysis_clean.csv --features data/processed/features.joblib --labels data/processed/labels.joblib --vectorizer data/processed/vectorizer.joblib
   ```
5. **Train models:**
   ```bash
   python src/train_model.py --features data/processed/features.joblib --labels data/processed/labels.joblib --baseline_model models/baseline_model.joblib --advanced_model models/advanced_model.joblib
   ```
6. **Evaluate models:**
   ```bash
   python src/evaluate_model.py --model models/baseline_model.joblib --features data/processed/features.joblib --labels data/processed/labels.joblib --output models/baseline_evaluation.json
   python src/evaluate_model.py --model models/advanced_model.joblib --features data/processed/features.joblib --labels data/processed/labels.joblib --output models/advanced_evaluation.json
   ```
7. **Run Streamlit app:**
   ```bash
   streamlit run app/app.py
   ```

## Future Improvements

- **Expand Dataset:** Collect more real customer feedback from multiple
  sources to train more robust models.
- **Advanced Models:** Explore transformer‑based models (e.g., BERT) and
  gradient boosting algorithms for improved accuracy.
- **Aspect‑Based Analysis:** Extend the system to identify sentiment
  toward specific product attributes (e.g., price, quality, customer
  service).
- **Topic Modelling:** Cluster feedback into themes to provide richer
  insights beyond overall sentiment.

## License

This project is for educational purposes and is distributed under the MIT
License.
