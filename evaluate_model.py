# Run this script with the following command
# python -m evaluate_model -p params.xlsx

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning msgs from tensorflow imports

import time
# import matplotlib.pyplot as plt
from pprint import pformat, pprint
import pandas as pd

from utility import evaluation, common, parameter_setup, data_prep
from models.model_trainer import ModelTrainer
from models import model_utility


def create_trainer(params, cnn_net, loaded_state):
    device_obj = model_utility.get_device(gpu_id=0, params=params)
    trainer = ModelTrainer(cnn_net, device_obj, params, loaded_state)
    return trainer


def create_results_df(all_params):
    col_names = []
    row_names = []

    for param_set in all_params:
        col_names.append(param_set['model_name'])
        row_names.append(param_set['dataset_name'])

    # Keep only the unique dataset and model names
    col_names = list(set(col_names))
    row_names = list(set(row_names))

    row_init_vals = [-1] * len(row_names)

    data = {name: row_init_vals for name in col_names}

    results_df = pd.DataFrame(data, columns=col_names, index=row_names)

    # results_df.index.name = 'dataset'
    results_df.sort_index(inplace=True)

    return results_df


def print_results(accuracies):
    pass


def run_experiment(params):
    # ------------------------
    # Set some basic params needed in model_trainer & data_prep
    params['distributed_training'] = 0
    params['gpu_id'] = 0
    params['num_gpus_per_node'] = 1
    params['testset_only'] = True

    parameter_setup.sanitize_params(params)
    common.init_logging(params, create_folder=False)
    # common.init_tensorboard(params)

    print('Experiment parameters')
    print(pformat(params))

    # ------------------------
    # Load data and model
    trainset, testset = data_prep.load_data(params)

    cnn_net, loaded_state = model_utility.load_model(params, set_weights=True)

    trainer = create_trainer(params, cnn_net, loaded_state=None)

    # ------------------------
    # Test model
    print('^^^^^^^^^^^^^^^^^^^')
    accuracy, group_accuracy = trainer.test_network(testset)
    print(f'Standard accuracy: {accuracy: .2f}%')
    print('^^^^^^^^^^^^^^^^^^^')

    common.save_all_figures(params['output_dir'])

    return accuracy


def main():
    print('Starting main')

    # ------------------------
    # Initialize params
    params_filename, other_args = parameter_setup.get_cmd_line_args()
    all_params = parameter_setup.get_params_from_file(params_filename)

    first_param_set = all_params[0]

    parameter_setup.sanitize_params(first_param_set)  # To get defaults
    common.init_logging(first_param_set)
    common.init_tensorboard(first_param_set)

    print('\n******************************\n')
    t0 = time.time()
    print(f'Running all {len(all_params)} experiments')

    print(f'Saving output to: {first_param_set["output_dir"]}')

    print('\n******************************\n')

    results_df = create_results_df(all_params)

    for i, param_set in enumerate(all_params):
        print('\n--------------\n')
        print(f'Running experiment {i+1}')

        param_set["output_dir"] = first_param_set["output_dir"] # All output goes to the same folder
        param_set = parameter_setup.override_cmd_line_args(param_set, other_args)
        parameter_setup.sanitize_params(param_set)

        accuracy = run_experiment(param_set)

        results_df.loc[param_set['dataset_name'], param_set['model_name']] = accuracy

        print(f'\nExperiment {i+1} complete.')
        print(f'Output saved to: {param_set["output_dir"]}')
        print('\n--------------\n')

    print('\n******************************\n')
    time_taken = time.time() - t0
    pprint(f'Finished running all {len(all_params)} experiments')
    pprint(f'Time taken for all experiments: {time_taken:.1f} sec, {time_taken/60:.1f} min, {time_taken/3600:.1f} hrs')

    print('\n******************************\n')
    
    print(results_df)


if __name__ == '__main__':
    main()
