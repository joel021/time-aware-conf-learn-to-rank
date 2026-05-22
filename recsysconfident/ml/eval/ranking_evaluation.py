"""
package: recsysconfident.ml.fit_eval
"""
import pandas as pd
from pandas import DataFrame

from recsysconfident.constants import RANK_SCORES_COL
from recsysconfident.data_handling.miscellaneous import min_max_norm
from recsysconfident.environment import Environment
from recsysconfident.ml.ranking.conf_aware_rank_metrics import ConfAwareRankingMetrics


def get_rank_scores(candidates_df: DataFrame, environ: Environment, k:int):
    conf_rank_calculator = ConfAwareRankingMetrics(environ.dataset_info, learn_to_rank=environ.learn_to_rank)
    rank_scores_mean, rank_scores_std = conf_rank_calculator.users_mean_std_rank_metrics(candidates_df, k)

    return rank_scores_mean, rank_scores_std

def ranking_scores(candidates_df: DataFrame, environ: Environment, k=10) -> dict:

    rank_scores_mean, rank_scores_std = get_rank_scores(candidates_df, environ, k)
    scores_dict = {
        f"mNDCG@{k}": f"{rank_scores_mean[0]:.5f}",
        f"stdNDCG@{k}": f"{rank_scores_std[0]:.5f}",
        f"mAP@{k}": f"{rank_scores_mean[1]:.5f}",
        f"stdP@{k}": f"{rank_scores_std[1]:.5f}",
        f"mRecall@{k}": f"{rank_scores_mean[2]:.5f}",
        f"stdRecall@{k}": f"{rank_scores_std[2]:.5f}",
        f"MRR@{k}": f"{rank_scores_mean[3]:.5f}",
        f"stdMRR@{k}": f"{rank_scores_std[3]:.5f}"
    }
    return scores_dict

def evaluate(split_df: pd.DataFrame, environ: Environment) -> dict:

    if RANK_SCORES_COL in split_df.columns:
        split_df.loc[:, environ.dataset_info.r_pred_col] = split_df[RANK_SCORES_COL]

    if not environ.learn_to_rank:
        split_df = min_max_norm(split_df, environ.dataset_info.relevance_col, environ.dataset_info.rate_range[1])

    ranking_10metrics = ranking_scores(split_df, environ, 10)
    ranking_3metrics = ranking_scores(split_df, environ,  3)

    return {**ranking_10metrics, **ranking_3metrics}
