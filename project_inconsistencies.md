
### 1. Inconsistent Binarization of Unconstrained BPR Scores
 [x] The inconsistency is solved.

In `ConfAwareRankingMetrics.rank_metrics` (defined in [conf_aware_rank_metrics.py](file:///home/joel/Documents/time-aware-conf-learn-to-rank/recsysconfident/ml/ranking/conf_aware_rank_metrics.py)), the prediction scores are binarized using a hardcoded method:
```python
def binarize(self, true_relevances):
    return (true_relevances >= self.r_ratio).astype(int)  # self.r_ratio defaults to 0.75
```
During evaluation, the code does:
```python
# Truncate to top-k
pred_top_k = pred_ratings_sorted[:k]
...
# Binarize after sorting
binary_pred = self.binarize(pred_top_k)
```
* **The Inconsistency**: Learn-to-rank models (such as Matrix Factorization trained with BPR loss) output arbitrary, unconstrained real-valued preference scores (e.g., dot products of latent factors, which can be negative or positive numbers like `-2.5`, `0.4`, or `1.8`). Comparing these unconstrained scores against a fixed threshold of `0.75` is mathematically incorrect. 
* **The Consequence**: 
  - If a user's top-k predictions are all below `0.75`, `binary_pred` becomes all zeros, resulting in a recall of `0.0`.
  - If they are all above `0.75`, `binary_pred` becomes all ones.
  - This invalidates `average_precision_score(binary_true, binary_pred)` and `recall_score(binary_true, binary_pred)` since the binary threshold does not scale with BPR scores.

---

### 2. Broken Confidence Calibration Logic
[x] This consistency is solved.
In `ConfAwareRankingMetrics.calibrate_ratings` (defined in [conf_aware_rank_metrics.py](file:///home/joel/Documents/time-aware-conf-learn-to-rank/recsysconfident/ml/ranking/conf_aware_rank_metrics.py)), scores are calibrated using:
```python
def calibrate(row):
    r, c = row[self.data_info.r_pred_col], row[self.data_info.conf_pred_col]
    if r >= self.r_ratio * self.data_info.rate_range[1] and c >= c_t:
        return r + self.alpha * c
    else:
        return r
```
* **The Inconsistency**: `self.data_info.rate_range[1]` is the maximum rating (which is `5` for the `ml-100k` dataset). This makes the condition check whether the model's prediction `r >= 0.75 * 5` (i.e., `r >= 3.75`).
* **The Consequence**: Because BPR scores are unconstrained latent factor dot products (centering typically around `0.0`), the condition `r >= 3.75` is virtually never met. As a result, confidence calibration is silently bypassed, and the calibration code has no effect on model scores in Learn-to-Rank mode.

---

### 3. Invalid Distance Metrics (MAE and RMSE)

[x] This inconsistency is solved.

In `get_distance_metrics` (defined in [ranking_evaluation.py](file:///home/joel/Documents/time-aware-conf-learn-to-rank/recsysconfident/ml/eval/ranking_evaluation.py)), Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated directly on predicted ranking scores:
```python
y_true = non_negative_sampled_df[environ.dataset_info.relevance_col].values  # Binarized: [0.0, 1.0]
y_pred = non_negative_sampled_df[environ.dataset_info.r_pred_col].values       # Unconstrained BPR scores
mae_score = mae(y_true, y_pred)
rmse_score = rmse(y_true, y_pred)
```
* **The Inconsistency**: The ground truth relevance labels `y_true` are binarized (`0.0` or `1.0`), while the predicted preference scores `y_pred` are on an arbitrary, unconstrained scale.
* **The Consequence**: MAE and RMSE require the prediction and target to be on the same scale (e.g., both ratings from 1 to 5, or both probabilities between 0 and 1). Calculating distance metrics between binarized labels and unconstrained preference scores is mathematically invalid.

---

### 4. Non-Sensical BPR Error for Negative Test Samples

[x] This inconsistency is solved.
In `elementwise_pos_neg_scores` (defined in [elementwise_error.py](file:///home/joel/Documents/time-aware-conf-learn-to-rank/recsysconfident/ml/ranking/elementwise_error.py)), the BPR error is calculated for negative interactions (`relevance == 0`):
```python
neg_split_df = split_df[~pos_indices]  # Relevance == 0
neg_true, neg_pred, neg_conf = obtain_neg_scores(...)
neg_split_df.loc[:, "neg_pred"], neg_split_df.loc[:, "neg_conf"] = neg_pred, neg_conf
```
* **The Inconsistency**: The BPR loss formulation contrasts a positive item's score against a negative item's score: $-\log(\sigma(r_{pos} - r_{neg}))$. For `neg_split_df`, the code compares the predicted score of a negative item (from the dataset) against the predicted score of another sampled negative item.
* **The Consequence**: Calculating BPR errors on negative test samples has no physical meaning since both items are negative, and BPR has no preference constraint between two negative items.