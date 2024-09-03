# Classification Metrics

- These metrics are problem-specific.
- They are used for classification problems.
- They help determine how good the model is.


# Regression

For regression:
- Coefficient of Determination

# Classification Metrics

For classification problems, there are many metrics:

## Accuracy
- **Definition**: Ratio of correct and total entries.
- **Explanation**: It is the number of correct vs. incorrect entries for all classes.

### Limitation
Consider a dataset:
- 95 entries = Class A
- 5 entries = Class B

Our model will get good prediction for Class A but not for Class B.
- For a skewed dataset, our accuracy is 95%, but the model is bad.

## Confusion Matrix
- **Purpose**: Performance check of the model.
- **Definition**: This is a matrix built to check how many correct and incorrect entries are predicted for each class.

### Example
Consider a binary classification system:
- **Pred 0**: 
  - **True 0**
  - **False 0**
- **Pred 1**:
  - **True 1**
  - **False 1**

Sample Confusion Matrix for Binary Classification:

### Example of Confusion Matrix
Consider a binary classification system:

|          | Pred 0 | Pred 1 |
|----------|--------|--------|
| True 0   |        |        |
| True 1   |        |        |


Sample Confusion Matrix for Binary Classification:

|          | Pred A        | Pred B        |
|----------|---------------|---------------|
| True A   | True positive | False negative|
| True B   | False positive| True negative |


Now, let's say the testing data has 50 Class A and 50 Class B points.

|        | Pred A        | Pred B        |
|--------|---------------|---------------|
| True A | 40 (True positive)  | 10 (False negative) |
| True B |  8 (False positive) | 42 (True negative)  |

- Predicted from Class A, the values that were correct were 40 and for Class B were 45.
- Incorrect Class A = 10
- Incorrect Class B = 5

## Definitions
1. **True Positive**: \[Pred = True = Class positive\]
2. **True Negative**: \[Pred = True = Class negative\]
3. **False Positive**: \[Pred ≠ True = Class positive\]
    - They are actually negative but predicted as positive (wrong prediction).
4. **False Negative**: \[Pred ≠ True = Class negative\]
    - They are actually positive but predicted as negative (wrong prediction).


# Precision and Total Recall

These metrics are defined for each class separately and are better than accuracy.

Consider the confusion matrix:

|        | Pred A        | Pred B        |
|--------|---------------|---------------|
| True A | 40 (True positive)  | 10 (False negative) |
| True B |  8 (False positive) | 42 (True negative)  |

## Precision
- It is how accurate I am OR what % of predicted A are correct.
- Precision for a class A is how many values that are predicted A are correct.

For our confusion matrix:
- Precision (A) = % of predicted A that are A
- \[ \text{Precision (A)} = \frac{40}{40 + 8} = \frac{40}{48} \approx 0.8333 \]
- Precision (B) = % of predicted B that are B
- \[ \text{Precision (B)} = \frac{42}{42 + 10} = \frac{42}{52} \approx 0.8077 \]

## Recall
- It tells how many values I was able to recall.

- What % of A predicted were correctly predicted to be A?
- Recall (A) = How many values of A could be recalled to be A OR out of all actual values of A, how many were predicted correct.

## Example
- Recall (A) = \(\frac{40}{40 + 10} = \frac{40}{50} = 0.8\)
- Recall (B) = \(\frac{42}{42 + 8} = \frac{42}{50} = 0.84\)

> Both metrics need to be high for a good model.

## Example 2
Consider a confusion matrix:

|        | Pred A | Pred B |
|--------|--------|--------|
| True A | 95     | 0      |
| True B | 5      | 0      |

- Precision (A) = \(\frac{95}{95 + 5} = \frac{95}{100} = 0.95\)
- Recall (A) = \(\frac{95}{95} = 1\)
- Recall (B) = \(\frac{0}{5} = 0\)



\[ Fβ = (1 + β^2) \cdot \frac{PR}{β^2P + R} \]

### F1 Score

- Harmonic mean of precision and recall of each class.
- Used to tell a collective result rather than two individual metrics.

\[ H.M \text{ of a \& b} = \frac{2}{\frac{1}{x} + \frac{1}{y}} = \frac{2xy}{x + y} \]

- **Support**: How many actual true values of the class exist.
- Not required that much.


#### Fβ Score

- If the metric of precision or recall needs to be given a higher weightage, then this is used.
  