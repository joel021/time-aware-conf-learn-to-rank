"""
package: recsysconfident.data_handling.datasets.datasets.py
"""
import os
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
from pandas import DataFrame, read_csv, concat
from sklearn.model_selection import train_test_split

from recsysconfident.utils.datasets import filter_positives, map_ids


class DatasetInfo:


    def __init__(self, user_col: str, item_col: str, rating_col: str, interactions_file: str, columns: List[str],
                 rate_range: List[float], database_name: str, split_run_uri: str, metadata_columns: List[str],
                 items_file: Optional[str] = None, sep: str = ",", has_head: bool = False,
                 timestamp_col: Optional[str] = None, batch_size: int = 1024, root_uri: str = "./",
                 k_folds: int = 5, split_position: int = 0):

        # Columns
        self.user_col: str = user_col
        self.item_col: str = item_col
        self.relevance_col: str = rating_col
        self.conf_pred_col: str = "conf_pred"
        self.r_pred_col: str = "r_pred"
        self.timestamp_col: Optional[str] = timestamp_col

        # Files and Paths
        self.root_uri: str = root_uri
        self.interactions_file: str = interactions_file
        self.items_file: Optional[str] = items_file
        self.database_name: str = database_name
        self.split_run_uri: str = split_run_uri

        # Dataset Structure Info
        self.columns: List[str] = columns
        self.metadata_columns: List[str] = metadata_columns
        self.rate_range: List[float] = rate_range
        self.sep: str = sep
        self.has_head: bool = has_head

        # DataFrames (Internal State)
        self.ratings_df: Optional[DataFrame] = None
        self.items_df: Optional[DataFrame] = None
        self.fit_df: Optional[DataFrame] = None
        self.val_df: Optional[DataFrame] = None
        self.test_df: Optional[DataFrame] = None
        self.items_per_user: Dict[Any, Tuple[set, list]] = {}

        # Sizing and Training Params
        self.n_users: int = 0
        self.n_items: int = 0
        self.batch_size: int = batch_size
        self.ratio_t: float = 0.78  # Threshold for filtering positives
        self.k_folds: int = k_folds
        self.split_position: int = split_position


    def build(self, ratings_df: DataFrame, items_df: Optional[DataFrame], shuffle: bool) -> None:
        """
        Initializes the dataset state with raw dataframes, performs splitting,
        and computes initial user-item interaction sets.
        """
        self.ratings_df = ratings_df
        self.items_df = items_df

        self._split_interactions(shuffle)
        self.items_per_user = self._get_user_item_sets(self.ratings_df)

        print(f"{len(list(self.items_per_user.keys()))} mapped users sequentially!")


    def _split_interactions(self, shuffle: bool) -> "DatasetInfo":
        """
        Splits the interactions into fit, validation, and test sets.
        Loads existing splits if files are present.
        """
        fit_path = f"{self.split_run_uri}/ratings.fit.csv"
        test_path = f"{self.split_run_uri}/ratings.test.csv"
        items_path = f"{self.split_run_uri}/items.csv"

        os.makedirs(self.split_run_uri, exist_ok=True)

        if os.path.exists(fit_path) and os.path.exists(test_path):
            # Load existing splits
            self.fit_df = read_csv(fit_path)
            self.test_df = read_csv(test_path)
            self.val_df = self.test_df.copy()
            self.ratings_df = concat([self.fit_df, self.test_df], ignore_index=True)
            self.n_users = int(self.ratings_df[self.user_col].max() + 1)
            self.n_items = int(self.ratings_df[self.item_col].max() + 1)

            if os.path.exists(items_path):
                self.items_df = read_csv(items_path)
                self.items_df.set_index(self.item_col, inplace=True, drop=True)
        else:
            self.ratings_df = filter_positives(self.ratings_df, self.relevance_col, self.ratio_t)

            self.ratings_df, self.items_df = map_ids(self.ratings_df, self.items_df, self.user_col, self.item_col)

            if self.items_df is not None:
                self.items_df.to_csv(items_path, index=False)
                self.items_df.set_index(self.item_col, inplace=True, drop=True)

            # Sort ratings by timestamp prior to splitting
            if self.timestamp_col and self.timestamp_col in self.ratings_df.columns:
                self.ratings_df = self.ratings_df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            else:
                print(f"Warning: timestamp column '{self.timestamp_col}' not found or not specified. Splitting without sorting.")

            splits = np.array_split(self.ratings_df, self.k_folds)

            if self.split_position + 1 >= self.k_folds:
                raise ValueError(f"split_position {self.split_position} + 1 is out of bounds for {self.k_folds} folds.")

            self.fit_df = concat(splits[:self.split_position + 1], ignore_index=True)
            self.test_df = splits[self.split_position + 1].copy()
            self.val_df = self.test_df.copy()

            self.fit_df.to_csv(fit_path, index=False)
            self.test_df.to_csv(test_path, index=False)

            # Set self.ratings_df to the concatenation of fit and test for consistency
            self.ratings_df = concat([self.fit_df, self.test_df], ignore_index=True)
            self.n_users = int(self.ratings_df[self.user_col].max() + 1)
            self.n_items = int(self.ratings_df[self.item_col].max() + 1)

        return self

    def _get_user_item_sets(self, df: DataFrame) -> Dict:
        """
        Computes a dictionary mapping users to their rated items and relevance scores.
        """
        user_item_dict = (
            df.groupby(self.user_col)
            .apply(lambda x: (set(x[self.item_col].tolist()), x[self.relevance_col].tolist()))
            .to_dict()
        )
        return user_item_dict

    def get_splits(self) -> Tuple[Optional[DataFrame], Optional[DataFrame], Optional[DataFrame]]:
        """
        Returns the fit, validation, and test dataframes.
        """
        return self.fit_df, self.val_df, self.test_df
