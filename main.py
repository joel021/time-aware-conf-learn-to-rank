import argparse
import glob
import json
from types import SimpleNamespace

import torch

from recsysconfident.environment import Environment
from recsysconfident.ml.eval.inference_error_analysis import export_elementwise_error
from recsysconfident.ml.eval.ranking_evaluation import evaluate
from recsysconfident.setup import Setup
from recsysconfident.utils.files import export_metrics, export_setup, read_json, \
    setup_and_model_exists, setup_model_results_exists
from recsysconfident.setup_manager import setup_fit

def run_all_setups(setups: dict, split_position: int=0, shuffle: bool=False):

    for value in setups.values():
        setup = Setup(**value)
        setup.set_split_position(split_position)
        main(setup, shuffle)

def run_k_folds(setups: dict, split_position: int, k: int):
    for i in range(split_position, k):
        print(f"Running fold {i}.")
        run_all_setups(setups, i, not i == 0) #Use sorted splitting for the first fold.


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
                          learn_to_rank=setup.learn_to_rank
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

def handle_setup_instance(setup_instance_path):
    """Run a specific setup instance from a provided JSON file."""
    setup_json = read_json(setup_instance_path)
    setup = Setup(**setup_json)
    main(setup)

def handle_reevaluate(runs_folder: str):
    """Re-evaluate all setups found in the runs directory."""
    setups_uri_list = glob.glob(f"{runs_folder}/**/setup-[0-9].json")
    for setup_uri in setups_uri_list:
        with open(setup_uri, 'r') as f:
            setup_data = json.load(f)

        setup_data['instance_dir'] = setup_uri[:setup_uri.rindex("/")]
        setup_data['reevaluate'] = True
        setup = Setup(**setup_data)

        main(setup)

def handle_all_setups(setups):
    """Run all predefined setups from the setups JSON file."""
    run_all_setups(setups)

def handle_k_folds(setups, split_position, k_folds):
    """Run k-fold cross-validation for all predefined setups."""
    if k_folds <= 0:
        raise ValueError("k_folds must be greater than 0")
    run_k_folds(setups, split_position, k_folds)

def handle_single_setup(setup_data):
    """Run a single predefined setup from the setups JSON file."""
    setup = Setup(**setup_data)
    main(setup)

def handle_custom_setup(args):
    """Run a custom setup using command-line arguments."""
    setup = Setup(
        model_name=args.model_name,
        database_name=args.database_name,
        split_position=args.split_position,
        fit_mode=args.fit_mode,
        batch_size=args.batch_size,
        conf_calibration=bool(args.conf_calibration),
        learning_rate=args.learning_rate,
        learn_to_rank=bool(args.learn_to_rank)
        # Include additional parameters as needed
    )
    main(setup)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups", type=str, default="setups.json",
                        help="Path to setups JSON file containing starting parameters")

    args = parser.parse_args()
    config_data = read_json(args.setups)

    if isinstance(config_data, dict) and "setups" in config_data:
        setup_name = config_data.get("setup_name", "none")
        k_folds = config_data.get("k_folds", 0)
        database_name = config_data.get("database_name", "ml-1m")
        fit_mode = config_data.get("fit_mode", 0)
        model_name = config_data.get("model_name", "mf")
        setup_instance = config_data.get("setup_instance", None)
        split_position = config_data.get("split_position", 0)
        learning_rate = config_data.get("learning_rate", 1e-3)
        batch_size = config_data.get("batch_size", 1024)
        conf_calibration = config_data.get("conf_calibration", 0)
        runs_folder = config_data.get("runs_folder", "./runs")
        learn_to_rank = config_data.get("learn_to_rank", 0)
        setups = config_data.get("setups", {})
    else:
        setup_name = "none"
        k_folds = 0
        database_name = "ml-1m"
        fit_mode = 0
        model_name = "mf"
        setup_instance = None
        split_position = 0
        learning_rate = 1e-3
        batch_size = 1024
        conf_calibration = 0
        runs_folder = "./runs"
        learn_to_rank = 0
        setups = config_data if isinstance(config_data, dict) else {}

        if setups:
            if len(setups) == 1:
                setup_name = list(setups.keys())[0]
            else:
                setup_name = "all"

    config = SimpleNamespace(
        setup_name=setup_name,
        k_folds=k_folds,
        database_name=database_name,
        fit_mode=fit_mode,
        model_name=model_name,
        setup_instance=setup_instance,
        split_position=split_position,
        learning_rate=learning_rate,
        batch_size=batch_size,
        conf_calibration=conf_calibration,
        runs_folder=runs_folder,
        learn_to_rank=learn_to_rank,
        setups=setups
    )

    if config.setup_instance:
        handle_setup_instance(config.setup_instance)
    elif config.setup_name == "reevaluate":
        handle_reevaluate(config.runs_folder)
    elif config.setup_name == "all":
        handle_all_setups(config.setups)
    elif config.setup_name == "k_folds":
        handle_k_folds(config.setups, config.split_position, config.k_folds)
    elif config.setup_name in config.setups:
        handle_single_setup(config.setups[config.setup_name])
    else:
        handle_custom_setup(config)
