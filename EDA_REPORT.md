# Exploratory Data Analysis (EDA)

This report summarises the exploratory analysis conducted on the cleaned
`Customer Feedback Dataset`.  The goal of the EDA is to understand the
structure, distribution, and key characteristics of the data before
modeling.

## 1. Missing Value Analysis

The cleaned dataset contains two columns, `Text` and `Sentiment`.  After
preprocessing, there are **no missing values** in either column.  This
simplifies further analysis since no imputation is required.

## 2. Duplicate Analysis

The raw dataset originally contained many duplicated entries.  During
preprocessing, duplicates were removed, reducing the dataset from 300 rows
to 30 unique rows.  Removing duplicates helps prevent models from being
biased toward repeated sentences and ensures that evaluation metrics reflect
unique feedback instances.

## 3. Distribution Analysis

### Sentiment Distribution

The class distribution in the cleaned dataset is perfectly balanced, with
ten samples for each sentiment class.  The bar chart below (stored in
`visuals/sentiment_distribution.png`) visually confirms this balance.

![Sentiment Distribution](visuals/sentiment_distribution.png)

### Text Length Distribution

To understand the variability in customer comments, we calculated the
number of words in each comment.  The histogram of word counts
(`visuals/text_length_distribution.png`) shows that most feedback messages
contain between 4 and 10 words, with a few shorter and longer comments.

![Distribution of Text Length](visuals/text_length_distribution.png)

## 4. Word Frequency Analysis

For each sentiment class we extracted the most frequently occurring words
after removing common English stop‑words.  The figure below lists the top
words for the **Positive**, **Neutral**, and **Negative** feedback classes.

![Top Words per Sentiment](visuals/top_words_per_sentiment.png)

Key observations:

- Positive feedback often includes words like **amazing**, **love**, and
  **great**, reflecting enthusiastic sentiment.
- Neutral comments frequently contain words such as **average** or
  **received**, suggesting moderate experiences.
- Negative feedback is characterised by words like **disappointed**,
  **poor**, and **unhappy**.

## 5. Correlation and Outlier Analysis

Because the dataset consists of unstructured text and a categorical label,
traditional numerical correlation analysis does not apply.  Instead, we
focus on textual patterns and class frequencies.  Outlier detection in
textual data typically involves identifying extremely short or long
documents or unusual vocabularies.  In this dataset, word counts are
mostly within a reasonable range, with no extreme outliers.

## 6. Target Analysis

The target variable (`Sentiment`) has an equal number of samples across
all three classes.  Balanced classes simplify model training and ensure
that accuracy and other metrics are meaningful.  Future work on larger,
real‑world data may require techniques like resampling or class weighting
to address class imbalance.

## 7. Summary

The EDA shows that the cleaned customer feedback dataset is small,
balanced, and free of missing values.  Text comments are concise,
typically under ten words.  Common words within each sentiment class
provide intuitive cues about what drives positive, neutral, and negative
feedback.  These insights guided the feature engineering and model
selection phases of the project.
