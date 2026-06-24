# Dataset Analysis

## Overview

The `Customer Feedback Dataset` consists of textual feedback from customers along with sentiment labels.  The data
was constructed synthetically for this portfolio project because the original Kaggle dataset was not directly
accessible via automation.  Each row contains a customer comment and a corresponding sentiment label indicating
whether the comment expresses **Positive**, **Neutral**, or **Negative** sentiment.

## Basic Characteristics

| Metric               | Value                                                     |
|----------------------|-----------------------------------------------------------|
| **Number of rows**   | 300                                                       |
| **Number of columns**| 2                                                         |
| **Column names**     | `Text` (customer comment) and `Sentiment` (label)         |
| **Data types**       | Both columns are of type `object` (string)                |
| **Missing values**   | No missing values in either column                        |
| **Duplicate rows**   | 270 duplicates (many comments appear more than once)      |
| **Class distribution** | 100 Positive, 100 Neutral, 100 Negative (balanced)       |

The dataset is balanced across the three sentiment classes, which simplifies model training because class imbalance
is not a concern.  However, a very high proportion of rows are duplicates because the synthetic data generation
process reused many sentences.  These duplicates should be removed during preprocessing to avoid training the model
on redundant information.

## Data Quality Issues

- **Duplicates:** Out of 300 rows, 270 are duplicates.  Duplicate feedbacks do not add new information and can
  inflate performance metrics artificially.  The preprocessing step will remove duplicates.
- **Limited Vocabulary:** Because the dataset is synthetic and reuses the same sentences many times, the diversity
  of vocabulary is limited.  In a real‑world scenario, customer feedback would contain a much richer variety of
  expressions.  The small vocabulary may restrict the ability of complex models to learn nuanced patterns.
- **No Missing Values:** There are no missing values in the raw data, so no imputation is necessary.
- **Balanced Classes:** The equal distribution of labels is advantageous for model evaluation because accuracy
  will not be inflated by majority class dominance.

## Target Variable

The target variable is the `Sentiment` column, which contains three possible values: **Positive**, **Neutral**, and
**Negative**.  This makes the problem a **multi‑class classification** task.

## Business Context

Sentiment analysis helps organisations understand customer opinions and emotions expressed in text feedback.  According
to a general description of sentiment analysis, it is an NLP technique that identifies emotions or opinions in text
and classifies the text as positive, negative or neutral【215926350891116†L27-L30】.  Implementing sentiment
analysis on customer feedback can improve customer satisfaction, support brand reputation management, and inform
product and marketing decisions【215926350891116†L198-L207】.  In this project we aim to build a system that can
automatically classify new customer feedback into sentiment categories, enabling stakeholders to monitor
customer sentiment at scale.

## Recommended Machine‑Learning Approach

Given that the data consists of short text documents labelled with sentiment, a traditional NLP pipeline using
bag‑of‑words or TF‑IDF representations coupled with linear classifiers is appropriate for a baseline model.  Logistic
regression often performs well for sentiment classification because it trains quickly and provides robust results【719788731322268†L305-L309】.  Other models such as support vector machines (SVMs) and random forests can also yield good
performance, especially when combined with appropriate feature engineering【215926350891116†L90-L98】.

### Baseline Model

**TF‑IDF + Logistic Regression.**

*Rationale:* Logistic regression is a simple yet powerful linear classifier that works well on high‑dimensional sparse
feature spaces like TF‑IDF vectors.  The model is easy to interpret, trains quickly, and serves as a strong
benchmark for more complex models【719788731322268†L305-L309】.

### Advanced Model

**Random Forest Classifier.**

*Rationale:* Random forests are ensemble models that can capture non‑linear relationships and interactions between
features.  They are robust to overfitting and do not require extensive parameter tuning.  When combined with
TF‑IDF features, a random forest can model complex patterns that a linear model might miss.  Other advanced models
such as support vector machines or gradient boosting (e.g., XGBoost, LightGBM) could also be considered, but random
forests provide a good balance between performance and computational cost for this small dataset.

## Evaluation Metrics

Because this is a multi‑class classification problem, the following metrics are recommended:

- **Accuracy:** Proportion of correctly predicted labels across all classes.  Useful as a general measure when
  classes are balanced.
- **Precision, Recall, and F1‑score:** Evaluate the classifier’s performance for each class.  Precision measures the
  fraction of positive predictions that are correct, recall measures the fraction of true positives that are
  captured, and the F1‑score balances precision and recall.
- **Confusion Matrix:** Provides a detailed breakdown of true vs. predicted classes for each sentiment, enabling
  examination of misclassification patterns.
- **ROC‑AUC:** When extended to multi‑class via one‑vs‑rest, this metric measures the model’s ability to
  differentiate between classes.

These metrics together offer a comprehensive view of model performance and will be used in later phases to assess
baseline and advanced models.
