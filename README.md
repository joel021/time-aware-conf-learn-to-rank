# RecSysConfident

A Python framework for robust recommender system experiments. The framework is structured to evaluate recommendation models with a focus on Learn-to-Rank (BPR) approaches using time-aware cross-validation.

---

## 1. Installation & Setup

We recommend using **Poetry** to manage dependencies.

### Configure Poetry to create a local virtual environment
```bash
poetry config virtualenvs.in-project true
poetry install
```

### Manual Installation (Alternative)
```bash
pip install -r requirements.txt
# Install PyTorch and PyTorch Geometric dependencies:
pip install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

---

## 2. Running Experiments & Tests

### Run the Full Experiment Pipeline
Run the Matrix Factorization setup on `ml-100k`:
```bash
poetry run python main.py --setups ./setups/mf.json
```

### Run the Test Suite
Run all unit and integration tests (including the end-to-end main flow integration test):
```bash
poetry run pytest tests
```

---

## 3. Project Architecture

* **`recsysconfident`**: The core package.
  * **`data_handling`**: Preprocessing, dataset readers, and custom dataloaders.
    * `datasets/`: Contains dataset readers (`movie_lens_reader.py`, etc.) and `datasetinfo.py` which manages data partitioning and directory structures.
    * `dataloader/`: Contains dataloaders, such as `int_ui_ids_dataloader.py` for training batches.
  * **`ml`**: Machine learning components.
    * `fitting/`: Handles the fit and early stopping loops for different models.
    * `models/`: Implementations of recommendation models (MF, LightGCN, DGAT, etc.) registering their architectures and compatible dataloaders.
    * `ranking/`: Code for rank metrics and element-wise/BPR loss error calculation.
    * `eval/`: Pipeline evaluation steps and metrics export.
  * **`utils`**: General helper scripts for file operations and configuration parsing.

---

## 4. Key Design Choices & Architecture

### A. Time Series Cross-Validation
- **Strategy**: Instead of randomized Monte Carlo cross-validation, the framework uses a chronological **Time Series Cross-Validation** strategy.
- **Data Splitting**:
  - The dataset is sorted chronologically and partitioned into $k$ splits.
  - For fold $i$, folds $[0 \dots i]$ are used for training (`ratings.fit.csv`), and fold $i+1$ is used for evaluation/testing (`ratings.test.csv`).
  - No physical validation split is written to disk. Early stopping is performed using a duplicated in-memory `val_df` mapped from the test set.

### B. Enforced Learn-to-Rank (LTR) Focus
- The system is configured globally to operate under a Learn-to-Rank approach (`learn_to_rank = True`).
- **Binarization**: Ratings above a specified relevance threshold are binarized to `1.0` (positives), and ratings below the threshold are mapped to `0.0` (negatives). Both types of interactions are preserved in the training and testing datasets on disk.
- **Negative Feedback in Training**:
  - To properly compute BPR loss, the user's positive interactions dictionary (`self.items_per_user`) is built using only positive interactions (`relevance == 1`).
  - Consequently, low-relevance (`relevance == 0`) and unobserved items are correctly treated as negatives by the BPR negative sampler.
  - Training dataloaders filter incoming training data to keep only `relevance == 1` samples, while test dataloaders evaluate on the full, unfiltered test splits.

### C. Distance Metrics & BPR Errors under LTR
- **Distance Metrics**: Since BPR scores are unconstrained real values representing relative preference (and not absolute rating predictions), standard distance metrics like MAE and RMSE are bypassed and return `0.0` when running in Learn-to-Rank mode.
- **BPR Error Calculation**: BPR errors are restricted exclusively to actual positive interactions (`relevance == 1`). Negative test interactions (`relevance == 0`) have their negative predicted score set to `0.0` and their elementwise BPR error zeroed out.

### D. Removal of Confidence Calibration
- The confidence calibration feature has been completely removed from the system. `conf_calibration` is globally set to `False` by default, and all threshold-tuning, calibration loops, and calibrated metrics outputs have been simplified/deleted to ensure performance and simplicity.