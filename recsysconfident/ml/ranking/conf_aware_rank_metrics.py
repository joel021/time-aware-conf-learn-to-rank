"""
package: recsysconfident.ml.ranking.conf_aware_rank_metrics
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score, recall_score

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo

warnings.filterwarnings("ignore")

class ConfAwareRankingMetrics:

    def __init__(self, data_info: DatasetInfo, r_ratio: float=0.75, alpha: float = 5, learn_to_rank: bool = False):
        self.data_info = data_info
        self.r_ratio = r_ratio
        self.alpha = alpha
        self.learn_to_rank = learn_to_rank

    def binarize(self, true_relevances):
        return (true_relevances >= self.r_ratio).astype(int)

    def reciprocal_rank_at_k(self, pred_relevances, true_relevance, k):
        top_k_indices = np.argsort(-pred_relevances)[:k]
        for rank, idx in enumerate(top_k_indices, start=1):
            if true_relevance[idx] >= 1:
                return 1.0 / rank
        return 0.0

    def _get_true_pred_scores(self, df: pd.DataFrame) -> dict:
        user_true_pred_scores = df.groupby(self.data_info.user_col).apply(
            lambda x: (
                x[self.data_info.relevance_col].values,
                x[self.data_info.r_pred_col].values
            )
        ).to_dict()

        return user_true_pred_scores

    def rank_metrics(self, candidates_df: pd.DataFrame, k: int) -> list:
        user_true_pred_scores = self._get_true_pred_scores(candidates_df)
        metrics = []
        for user_key in user_true_pred_scores.keys():
            true_ratings, pred_ratings = user_true_pred_scores[user_key]

            sorted_indices = np.argsort(pred_ratings)[::-1]
            pred_ratings_sorted = np.array(pred_ratings)[sorted_indices]
            true_ratings_sorted = np.array(true_ratings)[sorted_indices]

            # Binarize all true relevances
            binary_true_all = self.binarize(true_ratings_sorted)

            # Truncate to top-k
            pred_top_k = pred_ratings_sorted[:k]
            binary_true_top_k = binary_true_all[:k]

            # Recommend top-k items: binary_pred is all ones for top-k items
            binary_pred_top_k = np.ones_like(pred_top_k, dtype=int)

            # Compute recall using the user's total true positive items
            # Recall@k = (number of relevant items in top-k) / (total relevant items for this user)
            total_positives = np.sum(binary_true_all)
            if total_positives > 0:
                recall = np.sum(binary_true_top_k) / total_positives
            else:
                recall = 0.0

            # Compute Average Precision on top-k
            # If binary_true_top_k has no positive class, AP is 0.0
            if np.sum(binary_true_top_k) > 0:
                ap = average_precision_score(binary_true_top_k, pred_top_k)
            else:
                ap = 0.0

            # Compute MRR at k
            mrr = self.reciprocal_rank_at_k(pred_ratings_sorted, binary_true_all, k)

            metrics.append([
                ndcg_score([true_ratings_sorted], [pred_ratings_sorted], k=k),
                ap,
                recall,
                mrr
            ])
        return metrics

    def users_mean_std_rank_metrics(self, candidates_df: pd.DataFrame, k: int) -> tuple:
        users_scores = self.rank_metrics(candidates_df, k)
        scores = np.array(users_scores)

        mean_metrics = np.mean(scores, axis=0)
        std_metrics = np.std(scores, axis=0)

        return mean_metrics, std_metrics
