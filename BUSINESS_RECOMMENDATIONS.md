# Business Insights & Recommendations

## Executive Summary

This project developed a sentiment‑analysis system that classifies customer
feedback into **Positive**, **Neutral**, and **Negative** categories.  Using a
balanced dataset of unique customer comments, the team built and evaluated
both baseline and advanced machine‑learning models.  The advanced Random
Forest model achieved an accuracy of **90%**, outperforming the logistic
regression baseline.  Key insights from the exploratory analysis and model
results inform actionable recommendations to enhance customer experience and
business performance.

## Key Findings

1. **Balanced Sentiment Distribution:** The dataset contains an equal
   number of positive, neutral, and negative feedback instances, enabling
   fair model training and evaluation.
2. **Distinct Vocabulary by Sentiment:** Positive comments frequently
   include words like *amazing*, *love*, and *great*, while negative
   comments include *disappointed*, *poor*, and *unhappy*.  Neutral
   comments use more moderate terms like *average* and *received*.
3. **Model Performance:** The Random Forest model achieved better overall
   performance (accuracy 90%, precision 91%, recall 90%) than logistic
   regression (accuracy 86.7%).  It reduces misclassifications of
   neutral vs. negative feedback.

## Business Recommendations

1. **Monitor Sentiment Trends:** Deploy the trained sentiment classifier in
   production to automatically analyse incoming customer feedback in real
   time.  Monitor the proportion of negative feedback to identify
   emerging issues quickly.
2. **Prioritise Negative Themes:** Focus on the key drivers of negative
   feedback—such as poor quality or shipping delays—and prioritise
   initiatives to address these pain points.  Use keyword analysis to
   surface recurring issues for engineering and operations teams.
3. **Amplify Positive Feedback:** Highlight recurring positive themes like
   excellent performance and features in marketing materials and product
   messaging.  Encourage satisfied customers to share their experiences.
4. **Personalise Customer Response:** Integrate sentiment scores into the
   customer support workflow.  Negative comments should trigger alerts for
   rapid follow‑up, while positive feedback could be used to identify
   brand advocates.
5. **Invest in Data Collection:** Expand the dataset by aggregating
   feedback from multiple channels (surveys, social media, support
   tickets).  A larger and more diverse dataset will enable more
   sophisticated models (e.g., transformer‑based models like BERT) and
   support nuanced sentiment and topic analysis.

## Risk Assessment

- **Data Quality Risk:** The current dataset is synthetic and small.  If
  deployed in production without additional data, models may not generalise
  well to real customer feedback.  Mitigation: collect and label more
  real‑world feedback data.
- **Model Bias:** Models trained on limited vocabulary might misinterpret
  slang or domain‑specific phrases.  Mitigation: update the training data
  regularly and monitor model performance across different customer groups.
- **Operational Complexity:** Integrating sentiment analysis into existing
  systems requires coordination across data engineering, customer support,
  and product teams.  Mitigation: start with a pilot programme and
  incrementally scale up.

## Future Opportunities

- **Topic Modelling:** Beyond sentiment classification, apply topic
  modelling to cluster feedback into themes (e.g., customer service,
  product features) to provide deeper insights.
- **Aspect‑Based Sentiment Analysis:** Identify sentiment toward specific
  product attributes rather than the overall sentiment of the comment.
- **Real‑Time Dashboards:** Build dashboards that visualise sentiment
  trends over time and segment feedback by demographic or product line.
- **Integration with CRM:** Combine sentiment scores with customer
  lifetime value or purchase history to prioritise follow‑up actions.

The insights and recommendations outlined here will help the business
leverage customer feedback to improve products, services, and customer
satisfaction while building a foundation for more advanced analytics in
the future.
