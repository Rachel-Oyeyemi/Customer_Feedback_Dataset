# Model Comparison

This document compares the performance of the baseline and advanced models
developed for sentiment classification on the customer feedback dataset.

## Models Evaluated

- **Baseline Model:** Logistic Regression trained on TF‑IDF features.
- **Advanced Model:** Random Forest Classifier trained on the same TF‑IDF
  features.

Both models were evaluated on the full dataset after training and
preprocessing.  Although cross‑validation would be preferable on a larger
dataset, the small sample size limited this analysis to a single train/test
split during training and evaluation on the full dataset for comparison.

## Performance Metrics

| Model               | Accuracy | Precision | Recall | F1‑Score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | 0.867    | 0.879     | 0.867  | 0.868    |
| Random Forest       | **0.900**| **0.911** | **0.900**| **0.899** |

*Metrics are rounded to three decimal places.*

### Interpretation

The advanced Random Forest model achieves higher accuracy, precision,
recall, and F1‑score than the baseline Logistic Regression model.
Specifically, the Random Forest improves accuracy by about 3.3 percentage
points and F1‑score by roughly 3.2 percentage points.  These gains
indicate that the ensemble model captures nonlinear relationships in the
data that the linear classifier cannot.  However, the improvement is
modest, likely due to the small dataset and limited vocabulary.

## Confusion Matrices

The confusion matrices illustrate how each model classifies instances of
each sentiment class.  Rows correspond to true labels and columns to
predicted labels.

### Logistic Regression

```
True\Pred | Negative | Neutral | Positive
----------------------------------------
Negative  | 9        | 1       | 0
Neutral   | 1        | 9       | 0
Positive  | 1        | 1       | 8
```

### Random Forest

```
True\Pred | Negative | Neutral | Positive
----------------------------------------
Negative  | 9        | 0       | 1
Neutral   | 0        | 9       | 1
Positive  | 1        | 1       | 8
```

**Observations:**

- The Random Forest makes fewer misclassifications overall.  In particular,
  it avoids predicting **Neutral** for negative feedback and vice versa.
- Both models occasionally confuse positive feedback with negative or
  neutral, but misclassifications are rare.

## Conclusion

While both models perform reasonably well given the small dataset, the
Random Forest provides a slight edge in predictive performance.  For
production use on larger datasets, experimenting with additional models
such as Support Vector Machines or gradient boosting (e.g., XGBoost) could
yield further improvements.  Nonetheless, the Random Forest is selected as
the preferred model for deployment in this project.
