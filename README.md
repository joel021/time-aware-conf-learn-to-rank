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
- **Strategy**: Instead of randomized cross-validation, the framework uses a chronological **Time Series Cross-Validation** strategy.
- **Data Splitting**:
  - The dataset is sorted chronologically and partitioned into $k$ splits.
  - For fold $i$, folds $[0 \dots i]$ are used for training (`ratings.fit.csv`), and fold $i+1$ is used for evaluation/testing (`ratings.test.csv`).
  - No physical validation split is written to disk. Early stopping is performed using a duplicated in-memory `val_df` mapped from the test set.
