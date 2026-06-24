# Project Charter: Customer Feedback Sentiment Analysis

## Project Name

**Customer Feedback Dataset – NLP & Machine Learning**

## Business Problem

Organisations receive large volumes of unstructured feedback from customers via surveys, reviews, social media, and
support channels.  Manually reading and interpreting this feedback is time‑consuming and often subjective.  The
business needs a way to automatically classify customer feedback into sentiment categories (Positive, Neutral,
Negative) so that teams can quickly identify areas of customer satisfaction or concern and respond appropriately.

## Project Objectives

1. **Ingest and Process Data:** Download, clean, and prepare the customer feedback dataset for analysis.
2. **Explore the Data:** Perform exploratory analysis to understand the distribution, quality, and key patterns in
   the feedback.
3. **Build ML Models:** Develop baseline and advanced machine‑learning models to classify feedback sentiment.  The
   baseline model will use TF‑IDF features with logistic regression, while the advanced model will employ a random
   forest classifier on the same features.
4. **Evaluate Performance:** Assess models using appropriate metrics (accuracy, precision, recall, F1, ROC‑AUC) and
   select the best-performing model for deployment.
5. **Deliver Business Insights:** Translate modelling results into actionable insights and recommendations for
   stakeholders, highlighting key drivers of sentiment and potential areas for improvement.
6. **Deploy an Application:** Develop a Streamlit web application that allows users to enter new feedback, view
   sentiment predictions, and explore model performance and business insights.

## Stakeholders

- **Customer Experience Team:** Needs to monitor customer sentiment and act on negative feedback to improve service.
- **Product Managers:** Use insights to prioritise feature enhancements and bug fixes based on customer sentiment.
- **Marketing & Communications:** Tailor messaging and campaigns to address common themes in positive or negative
  feedback.
- **Data Science & Engineering Team:** Responsible for building and maintaining the data pipeline, models, and
  application.
- **Executive Leadership:** Measures the overall impact on customer satisfaction and business metrics.

## Success Metrics

The project will be considered successful when:

1. **Data Pipeline Stability:** Raw data is ingested and processed without errors, and duplicate entries are
   removed.
2. **Model Performance:** The baseline model achieves at least **70%** accuracy and the advanced model improves upon
   this by at least **5 percentage points**.  Precision, recall, and F1‑scores should be above **0.70** for each
   class.
3. **Actionable Insights:** The analysis identifies the top factors influencing customer sentiment and provides
   clear recommendations to stakeholders.
4. **Application Usability:** The Streamlit application is easy to use, delivers predictions in real time, and
   displays model performance and insights clearly.
5. **Business Adoption:** Stakeholders integrate the results into decision‑making processes, such as product
   roadmaps or customer success strategies.

## Expected Business Impact

Implementing sentiment analysis on customer feedback enables the business to:

- **Enhance Customer Satisfaction:** Quickly identify and address negative feedback to improve customer experience.
- **Inform Product Development:** Prioritise enhancements based on recurring themes and sentiment patterns in
  feedback.
- **Protect Brand Reputation:** Detect and respond to negative sentiment early to mitigate potential reputational
  damage.
- **Improve Operational Efficiency:** Reduce the manual effort required to read and classify feedback, freeing up
  resources for higher‑value tasks.

## Technical Architecture

The solution comprises the following components:

1. **Data Ingestion & Storage:** Raw data is downloaded from an external source (Kaggle) and stored in a `data/raw`
   directory.  Preprocessed and processed data are stored in `data/processed`.
2. **Preprocessing Pipeline:** A Python script cleans the data by removing duplicates, handling missing values, and
   normalising text.  Processed data is saved for feature engineering.
3. **Feature Engineering:** TF‑IDF vectorisation converts text into numerical features.  The fitted vectoriser and
   feature matrix are stored for reuse.
4. **Model Training & Evaluation:** Baseline and advanced models are trained on the engineered features.  Results
   and models are persisted in the `models/` directory.  Evaluation metrics are recorded for comparison.
5. **Visualization & Reporting:** Exploratory data analysis and modelling results are presented through Jupyter
   notebooks and markdown reports located in the `notebooks/` and root directories, with charts saved in `visuals/`.
6. **Streamlit Application:** A web interface allows users to input new feedback for sentiment prediction, explore
   model performance, and view business insights.
7. **Version Control:** All code, data (excluding large intermediate files), reports, and assets are stored in a
   GitHub repository with clear documentation and structure.

## End‑to‑End Workflow

1. **Download Data:** Use `src/download_data.py` to download the raw CSV into `data/raw/`.
2. **Preprocess Data:** Execute `src/preprocess.py` to remove duplicates, handle missing values, and save the
   cleaned dataset to `data/processed/`.
3. **Feature Engineering:** Run `src/feature_engineering.py` to generate TF‑IDF features and save them along with
   labels and the vectoriser.
4. **Model Training:** Use `src/train_model.py` to train the baseline logistic regression and advanced random
   forest models, saving them into `models/`.
5. **Model Evaluation:** Evaluate models using `src/evaluate_model.py` and document results in `MODEL_EVALUATION.md`.
6. **Exploratory Analysis:** Conduct EDA via Jupyter notebooks (`01_data_exploration.ipynb`) and summarise findings
   in `EDA_REPORT.md`.
7. **Business Insights & Reporting:** Translate analysis into business recommendations in `BUSINESS_RECOMMENDATIONS.md`.
8. **Streamlit Deployment:** Deploy the Streamlit app (`app/app.py`) to provide an interactive interface for
   stakeholders.
9. **Documentation & Presentation:** Write comprehensive documentation (`README.md`) and create an executive
   presentation in the `presentation/` directory.  Prepare resume bullets and interview materials.
