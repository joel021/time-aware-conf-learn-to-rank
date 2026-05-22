import argparse
import json

import torch

from recsysconfident.environment import Environment
from recsysconfident.ml.eval.inference_error_analysis import export_elementwise_error
from recsysconfident.ml.eval.ranking_evaluation import evaluate
from recsysconfident.setup import Setup
from recsysconfident.utils.files import export_metrics, export_setup, read_json, \
    setup_and_model_exists, setup_model_results_exists
from recsysconfident.setup_manager import setup_fit

def run_all_setups(setups: dict, split_position: int=0, k_folds: int=5):

    for value in setups.values():
        setup_params = value.copy()
        if 'k_folds' not in setup_params:
            setup_params['k_folds'] = k_folds
        setup = Setup(**setup_params)
        setup.set_split_position(split_position)
        main(setup)

def run_k_folds(setups: dict, split_position: int, k: int):
    for i in range(split_position, k - 1):
        print(f"Running fold {i}.")
        run_all_setups(setups, i, k)


def main(setup: Setup, shuffle_train_split: bool = False):
    """
    shuffle_train_split: whether shuffle the train split or use sorted by timestamp
    """
    print(setup.to_dict())
    if setup_model_results_exists(setup.instance_dir) and not setup.reevaluate:
        print("All results already obtained. Skip.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    environ = Environment(model_name=setup.model_name,
                          database_name=setup.database_name,
                          instance_dir=setup.instance_dir,
                          split_position=setup.split_position,
                          batch_size=setup.batch_size,
                          conf_calibration=setup.conf_calibration,
                          min_inter_per_user=setup.min_inter_per_user,
                          learn_to_rank=setup.learn_to_rank,
                          k_folds=setup.k_folds
                          ).read_split_datasets(shuffle_train_split)

    model, fit_dl, val_dl, test_dl = environ.get_model_dataloaders()

    if setup.fit_mode == 0 and not setup_and_model_exists(setup.instance_dir):

        model = setup_fit(setup, model, fit_dl, val_dl, environ, device)


    export_setup(environ, setup.to_dict())
    eval_df, test_df = export_elementwise_error(model, environ, device)
    conf_10_3threshold = (None, None)
    eval_metrics = evaluate(eval_df, environ, conf_10_3threshold)
    if 'conf_threshold@10' in eval_metrics:
        conf_10_3threshold = (float(eval_metrics['conf_threshold@10']), float(eval_metrics['conf_threshold@3']))
    test_metrics = evaluate(test_df, environ, conf_10_3threshold)
    
    export_metrics(environ, {"eval": eval_metrics, "test": test_metrics})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups", type=str, default="setups.json",
                        help="Path to setups JSON file containing starting parameters")

    args = parser.parse_args()
    config_data = read_json(args.setups)

    if isinstance(config_data, dict) and "setups" in config_data:
        k_folds = config_data.get("k_folds", 5)
        split_position = config_data.get("split_position", 0)
        setups = config_data.get("setups", {})
    else:
        k_folds = 5
        split_position = 0
        setups = config_data if isinstance(config_data, dict) else {}

    run_k_folds(setups, split_position, k_folds)
