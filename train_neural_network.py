# Run this script with the following command
# python train_neural_network.py sunanda_params.yaml
import os
import time
import warnings
import torch

torch.set_printoptions(threshold=3)
warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning msgs from tensorflow imports

# import matplotlib.pyplot as plt
from pprint import pformat, pprint

from utility import evaluation, common, parameter_setup, data_prep, misc, visualization, attack_data_prep
from utility.distributed_model_trainer import DistributedModelTrainer
from models import model_utility
#
import evaluate_model


def test_model(params):

    eval_params = dict(params)

    model_path = params['output_dir'] + '/final_model_state.pt'
    eval_params['model_state_path'] = model_path
    eval_params['distributed_training'] = 0
    # eval_params['dataset_percent'] = 1  # Use the entire test set - useful in label efficiency experiments

    print(f'Loading trained model and evaluating on natural and attack datasets. model_path: {model_path}')

    # ------------------------
    # Load data and model
    trainset, testset = data_prep.load_data(eval_params)

    natural_loader, attacks_loader = attack_data_prep.load_attack_data(eval_params)

    cnn_net, _ = model_utility.load_model(eval_params, set_weights=True)

    trainer = evaluate_model.create_trainer(eval_params, cnn_net, loaded_state=None)

    # ------------------------
    # Test model
    train_time_mins = params['train_time_secs'] / 60
    train_time_hrs = params['train_time_secs'] / 60 / 60

    natural_accuracy, natural_group_accuracy = trainer.test_network(testset)

    attack_accuracy = -1
    robustness_str = ''
    avg_robustness_metrics = (-1, -1, -1)
    all_robustness_metrics = (-1, -1, -1)

    if attacks_loader is not None:
        attack_accuracy, attack_group_accuracy = trainer.test_network(attacks_loader)
        avg_robustness_metrics, all_robustness_metrics = trainer._test_robustness_one_epoch(natural_loader, attacks_loader)

        (avg_spearman, avg_kendall, avg_intersection) = avg_robustness_metrics
        robustness_str = f'Robustness metrics:\n' \
                         f'\tAverage Spearman: {avg_spearman: .2f}\n' \
                         f'\tAverage Kendall: {avg_kendall: .2f}\n' \
                         f'\tAverage Intersection: {100 * avg_intersection: .2f}%'

    print('\n===================== Results =====================\n')

    results_str = f'Standard (natural) accuracy: {natural_accuracy: .2f}% \n' \
                  f'Attack accuracy: {attack_accuracy: .2f}% \n' \
                  f'Train time: {train_time_mins: .1f} minutes, {train_time_hrs: .1f} hours \n' \
                  f'{robustness_str}'

    print(results_str)

    print('\n--- Group accuracy (natural) ---\n')
    for group_no, acc in natural_group_accuracy.items():
        print(f'\tGroup {group_no}: {acc: .2f}%')

    tb_writer = common.get_tensorboard_writer()

    if tb_writer is not None:
        tb_writer.add_text('Experiment results', results_str.replace('\n', '<br>'), -1)

    print('\n===================================================\n')

    all_metrics = {
        'natural_accuracy': natural_accuracy,
        'attack_accuracy': attack_accuracy,
        'train_time_mins': train_time_mins,
        'train_time_hrs': train_time_hrs,
        'robustness_str': robustness_str,
        'avg_robustness_metrics': avg_robustness_metrics,
        'all_robustness_metrics': all_robustness_metrics,
    }

    return all_metrics


def run_experiment(params):
    tb_writer = common.get_tensorboard_writer()
    print('Experiment parameters')
    print(pformat(params))
    tb_writer.add_text('Experiment params', pformat(params).replace('\n', '<br>'), -1)

    # ------------------------
    # Train model
    t0 = time.time()
    distributed_trainer = DistributedModelTrainer(params)
    distributed_trainer.run_training()
    # pprint(distributed_trainer.pipelines)
    params['train_time_secs'] = time.time() - t0

    # At this point, all spawned processes have finished running (joined).
    # i.e. this is the parent process of this node's run

    # Only one node will do these final actions (no point of having repeated outputs)
    node_id = int(os.environ['NODE_ID'])
    if node_id == 0:
        test_model(params)

        # ------------------------
        # Visualization - For now, use Tensorboard for viz

        # evaluation.plot_training_history(history)
        # evaluation.plot_entropy_history(history)
        # evaluation.plot_lr_history(history)

        # ig_callback = distributed_trainer.get_primary_model_trainer().ig_maps_callback
        # print(ig_callback.ig_maps)
        # visualization.plot_ig_maps_sequence(ig_callback.ig_maps)

        # plt.show()
        common.save_all_figures(params['output_dir'])


def main():
    print('Starting main')

    # ------------------------
    # Init
    params_filename, other_args = parameter_setup.get_cmd_line_args()
    all_params = parameter_setup.get_params_from_file(params_filename)

    print('\n******************************\n')
    t0 = time.time()
    pprint(f'Running all {len(all_params)} experiments')
    print('\n******************************\n')

    for i, param_set in enumerate(all_params):
        print('\n******************************\n')
        print(f'Running experiment {i+1}')

        param_set = parameter_setup.override_cmd_line_args(param_set, other_args)
        parameter_setup.sanitize_params(param_set)

        common.init_logging(param_set)
        common.init_tensorboard(param_set)

        print(f'Saving output to: {param_set["output_dir"]}')

        run_experiment(param_set)

        print('--------------')
        print(f'Experiment {i+1} complete.')
        print(f'Output saved to: {param_set["output_dir"]}')
        print(f'Sleeping 3 seconds for cool off')
        time.sleep(3)

    print('\n******************************\n')
    time_taken = time.time() - t0
    pprint(f'Finished running all {len(all_params)} experiments')
    pprint(f'Time taken for all experiments: {time_taken:.1f} sec, {time_taken/60:.1f} min, {time_taken/3600:.1f} hrs')

    print('\n******************************\n')


if __name__ == '__main__':
    main()
