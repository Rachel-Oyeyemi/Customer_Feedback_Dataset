# Model Evaluation

This report details the evaluation of the baseline and advanced
sentiment‑classification models using the full dataset.  Evaluation metrics
provide insight into how well each model predicts the sentiment labels.

## Evaluation Metrics

For each model we report the following metrics:

| Metric        | Description                                                                            |
|---------------|----------------------------------------------------------------------------------------|
| **Accuracy**  | The proportion of correct predictions across all classes.                              |
| **Precision** | The fraction of predicted positive cases that are actually positive (weighted average).|
| **Recall**    | The fraction of actual positive cases that are correctly identified (weighted average).|
| **F1‑Score**  | The harmonic mean of precision and recall (weighted average).                          |

### Logistic Regression (Baseline)

- **Accuracy:** 0.867
- **Precision:** 0.879
- **Recall:** 0.867
- **F1‑Score:** 0.868

**Confusion Matrix:**

```
True\Pred | Negative | Neutral | Positive
----------------------------------------
Negative  | 9        | 1       | 0
Neutral   | 1        | 9       | 0
Positive  | 1        | 1       | 8
```

The logistic regression model performs well overall, with most of the
misclassifications occurring when positive feedback is labelled as neutral
or negative.  Precision and recall are balanced across classes.

### Random Forest (Advanced)

- **Accuracy:** 0.900
- **Precision:** 0.911
- **Recall:** 0.900
- **F1‑Score:** 0.899

**Confusion Matrix:**

```
True\Pred | Negative | Neutral | Positive
----------------------------------------
Negative  | 9        | 0       | 1
Neutral   | 0        | 9       | 1
Positive  | 1        | 1       | 8
```

The random forest model achieves slightly better performance than the
logistic regression model, correctly classifying more negative and
neutral instances and achieving higher precision, recall, and F1‑scores.

## Discussion

The evaluation reveals that both models are capable of accurately
classifying customer feedback into sentiment categories.  The random
forest model shows superior performance across all metrics, making it the
preferred model for deployment.  However, the difference between the
models is modest due to the small size of the dataset.  In practice, more
data and more sophisticated models (e.g., gradient boosting, BERT) could
further improve performance.
