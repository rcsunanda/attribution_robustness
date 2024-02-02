# from torch.utils.tensorboard import SummaryWriter
import hashlib
import os
import psutil

import numpy as np
from tensorboardX import SummaryWriter

import time
import torch.optim as optim
import torch
from torch.nn.parallel import DistributedDataParallel

# import tensorboard as tb

# try:
#     import tensorflow as tf
#     # tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# except Exception as e:
#     print('Tensorflow not installed, this Tensorboard fix is not needed')

from tqdm import tqdm
from pprint import pprint

from int_tl.feature_significance import compute_integrated_gradients
from models import model_utility
from int_tl.loss_functions import get_loss_function
from utility import common, misc, attribution_robustness_metrics
from utility.group_error_calculator import GroupErrorCalculator


class ModelTrainer:
    """A class that trains a neural network
    """
    def __init__(self, network, device_obj, params, loaded_state=None):
        self.params = params    # For any later use
        self.network = network
        self.device_obj = device_obj
        self.distributed = params['distributed_training']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.test_batch_size = params['test_batch_size']
        self.early_stop_patience = params['early_stop_patience']
        self.gradient_clipping_value = params['gradient_clipping_value']
        self.do_gradient_clipping = self.gradient_clipping_value > 0.0

        if self.do_gradient_clipping:
            print(f'\n!!!! Gradient clipping enabled with value: {self.gradient_clipping_value} !!!!\n')

        # During training, do max 40 test set evals or at least every 2 epochs
        self.test_set_eval_epochs = max(2, self.epochs // 40)

        # During training, do max 10 attack set evals (error and *robustness metrics*) or at least every 5 epochs
        self.attack_set_eval_epochs = max(5, self.epochs // 10)

        self.epoch = -1    # Track current epoch (-1 = not started)
        self.steps = 0    # Track current step (0 = not started)

        self.is_primary = not self.distributed or (self.distributed and self.params['gpu_id'] == 0)

        if self.distributed:
            self.batch_size = int(self.batch_size / params['num_gpus_per_node'])

        optimizer_name = params['optimizer']
        loss_function_name = params['loss_function']
        test_loss_function_name = params['test_loss_function']

        self.input_gradients = False

        if loss_function_name in ['gradient_regularization', 'ig_regularization']:
            self.input_gradients = True

        if loss_function_name == 'changing':
            for key, val in params['changing_loss_functions'].items():
                if val in ['gradient_regularization', 'ig_regularization']:
                    self.input_gradients = True

        if loss_function_name == 'every_other':
            for val in params['changing_loss_functions']:
                if val in ['gradient_regularization', 'ig_regularization']:
                    self.input_gradients = True

        self.verbose = params['verbose']
        self.mixed_precision = bool(params['mixed_precision'])

        if self.is_primary:
            common.init_logging(params, create_folder=False)
            common.init_tensorboard(params)   # Only needed for distributed training
            self.tb_writer = common.get_tensorboard_writer()

            self.process = psutil.Process()  # Current Python process (self)
            # self._record_memory_usage(-1)  # Record memory usage before this object is initialized

        train_type = params['train_type']
        assert train_type in ['standard', 'adversarial']
        self.adv_train = True if train_type == 'adversarial' else False

        self.enable_progress_bar = self.verbose

        self.network = self.network.to(device_obj)  # Move model to device upfront (may fail for large models)

        self.train_loss_function = get_loss_function(loss_function_name, self.network, device_obj, self.batch_size, params)
        self.test_loss_function = get_loss_function(test_loss_function_name, self.network, device_obj, self.test_batch_size, params)

        self.optimizer, self.lr_scheduler = self._create_optimizer(optimizer_name, self.network, params)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        if loaded_state is not None:
            self._load_states_to_trainer(loaded_state)

        if self.distributed:
            gpu_id = params['gpu_id']
            self.network = DistributedDataParallel(self.network, device_ids=[gpu_id])
            self.enable_progress_bar = self.verbose and gpu_id == 0

        if self.adv_train:
            from art.attacks.evasion import ProjectedGradientDescent    # To prevent issues on Sharcnet
            from art.estimators.classification import PyTorchClassifier

            input_shape = params['input_shape']
            num_classes = params['num_classes']
            # AT parameters from Salman et al
            epsilon = 3.0
            alpha = epsilon * 2 / 3
            iterations = 3
            adv_loss_function = torch.nn.CrossEntropyLoss()
            classifier = PyTorchClassifier(model=self.network, loss=adv_loss_function, input_shape=input_shape, nb_classes=num_classes)
            self.attack_generator = ProjectedGradientDescent(estimator=classifier, norm=2, eps=epsilon, eps_step=alpha, max_iter=iterations, batch_size=1000, verbose=False)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

        # Not supported for Python 3.11 with PuTorch 2.0 stable builds. Try the nightly builds
        # self.network = torch.compile(self.network)

        # if self.is_primary:
        #     self._record_memory_usage(0)   # Record memory usage after this object is initialized, but before training

    # Note that in distributed mode, multiple processes run this training loop in parallel
    # A copy of the model exists in each process, and different batches are passed to them by the dataloader sampler
    # Synchronization happens in the loss.backward() call. Therefore, avoid other backward() calls within the loop
    # Gradients are then updated in each model copy
    # Due to this multiprocessing, some actions must only be done on the primary process
    # (eg: printing info, tensorboard logging, checkpointing, saving the model etc.)
    def train(self, train_loader, test_loader, natural_loader=None, attacks_loader=None):
        self.ig_maps_callback = misc.IGMapCallback(self.network, test_loader)

        # torch.autograd.set_detect_anomaly(True)   # To detect tensor modifications that prevent grad calculation

        num_steps_in_epoch = min(1, len(train_loader.dataset) // self.batch_size)

        # Record memory 10 times per epoch (to allow enough tracking),
        # but also enforce an upperbound frequency of 15 steps-per-record (to prevent too many system calls)
        self.memory_record_steps = max(15, num_steps_in_epoch // 10)

        t0 = time.time()

        self.network.train()  # To set layers (eg: Dropout, BatchNorm) to train mode

        # History variables
        lr_hist = []

        curr_best_test_loss = 1000000
        epochs_since_best_model = 0
        best_epoch = 0

        best_model_state = None  # To prevent 'referenced before assignment' errors after loop

        print('Training neural network')

        for epoch in range(1, self.epochs + 1):  # loop over the dataset multiple epochs
            self.epoch = epoch
            epoch_start_time = time.time()

            if self.distributed:
                train_loader.sampler.set_epoch(epoch)   # To use a different random ordering for each epoch

            # self._log_network_checksum()

            train_loss, train_error = self._train_one_epoch(train_loader)

            last_lr = 0
            if self.lr_scheduler is not None:
                last_lr = self.lr_scheduler.get_last_lr()[0]

            lr_hist.append(last_lr)

            epochs_since_best_model += 1
            best_model_marker = ''

            if self.is_primary:
                self.results = self._do_evaluations(test_loader, natural_loader, attacks_loader)

                # if test_loss < curr_best_test_loss:
                #     curr_best_test_loss = test_loss
                #     best_epoch = epoch
                #     epochs_since_best_model = 0
                #     best_model_marker = '***'
                #     best_model_state = {'epoch': epoch,
                #                         'model_state_dict': dict(self.network.state_dict()),  # Make a deep copy
                #                         'optimizer_state_dict': dict(self.optimizer.state_dict()),
                #                         'scaler_state_dict': self.scaler.state_dict(),
                #                         'train_loss_function': self.train_loss_function,
                #                         'perf_metrics': self.results}

                self._checkpoint()  # Handle intermediate checkpoints (every n epochs)

                epoch_time_secs = time.time() - epoch_start_time

                # Update the results dict with training-set related metrics, so they can be recorded in tensorflow
                self.results['train_loss'] = train_loss
                self.results['train_error'] = train_error
                self.results['epoch_time_secs'] = epoch_time_secs

                epoch_info = f'[epoch: {epoch}/{self.epochs}]\t' \
                             f'time: {epoch_time_secs:.1f}\t' \
                             f'train_loss: {train_loss:.4f}\t' \
                             f'train_error: {train_error:.2f}%\t'

                if self.do_test_set_eval:
                    epoch_info += f'test_loss: {self.results["test_loss"]:.4f}\t' \
                                 f'test_error: {self.results["test_error"]:.2f}%\t' \
                                 f'{best_model_marker}'

                print(epoch_info)

                self._record_tensorboard_metrics()

            if epochs_since_best_model >= self.early_stop_patience:
                print(f'No improvement to best model in {epochs_since_best_model} epochs. Early stopping...')
                break

            # if epoch % 10 == 0:
            #     self.ig_maps_callback.on_callback(epoch)

        history = {}
        history['learning_rate_hist'] = lr_hist

        time_to_train = time.time() - t0

        if self.is_primary:
            print(f'Finished Training. Time taken: {time_to_train:.1f} sec, {time_to_train / 60:.1f} min, {time_to_train/3600:.1f} hrs')
            self._checkpoint(is_final=True)  # Final checkpoint
            self.tb_writer.close()
            print('\n=============================================\n')

        # # Restore & save best model (smallest test loss)
        # if best_model_state is not None:
        #     # Restore model weights to best set
        #     self.network.load_state_dict(best_model_state['model_state_dict'])
        #
        #     if self.is_primary:
        #         model_path = self.params['output_dir'] + '/best_model_state.pt'
        #         torch.save(best_model_state, model_path)
        #         print(f'Best model state (from epoch {best_epoch}) saved to: {model_path}')

        return history

    def test_network(self, test_loader):
        # Move model to device upfront (may fail for large models)
        self.network.to(self.device_obj)
        self.network.eval()     # switch to evaluate mode

        num_samples = 0
        num_correct_preds = 0
        num_incorrect_preds = 0

        group_error_calculator = GroupErrorCalculator(self.device_obj)

        for inputs, labels, *metadata in test_loader:
            inputs = inputs.to(self.device_obj)
            labels = labels.to(self.device_obj)

            # with torch.no_grad():  # We do not intend to call backward(), as we don't backprop and optimize
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.network(inputs)

            _, y_pred = torch.max(outputs.data, 1)

            num_samples += y_pred.shape[0]
            num_correct_preds += (y_pred == labels).sum().item()

            is_incorrect_preds = (y_pred != labels)
            num_incorrect_preds += is_incorrect_preds.sum().item()

            if len(metadata) > 0:
                group_error_calculator.add_batch(metadata, labels, is_incorrect_preds)

        accuracy = num_correct_preds * 100 / num_samples
        print(f'num_samples: {num_samples}, num_correct_preds: {num_correct_preds}, num_incorrect_preds: {num_incorrect_preds}')

        group_errors = group_error_calculator.compute_group_errors()
        group_accuracy = group_error_calculator.convert_to_group_accuracies(group_errors)

        return accuracy, group_accuracy

    def _train_one_epoch(self, train_loader):
        num_batches = 0
        num_samples = 0
        train_loss_epoch = 0.0
        train_loss_classif_term = 0.0
        train_loss_reg_term = 0.0
        incorrect_preds = 0

        # train_loss_total_grad_sum = 0.0
        # train_loss_classif_term_grad_sum = 0.0
        # train_loss_reg_term_grad_sum = 0.0

        self.network.train()   # switch to train mode

        for inputs, labels, *metadata in tqdm(train_loader, disable=not self.enable_progress_bar):
            # -------------------------------------

            inputs.requires_grad = self.input_gradients

            if self.adv_train:
                # print('Generating adversarial examples')
                inputs = self.attack_generator.generate(x=inputs)
                inputs = torch.from_numpy(inputs)

            inputs = inputs.to(self.device_obj, non_blocking=True)
            labels = labels.to(self.device_obj, non_blocking=True)

            # -------------------------------------

            self.optimizer.zero_grad()  # To prevent gradient accumulation

            # forward + backward + optimize

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.network(inputs)

                train_loss_batch = self.train_loss_function(outputs, labels, inputs, self.epoch)
                # To clear out gradients accumulated during loss function computation (eg: IG, IG attacks etc)

            # If loss function involved computing gradients, we must clear them for correct loss gradient backprop
            self.optimizer.zero_grad()

            # In distributed training, synchronizing of the gradients occur here
            self.scaler.scale(train_loss_batch).backward()

            # Gradient clipping should be done after the backward() call and before the optimizer.step() call
            if self.do_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.gradient_clipping_value)

            # self._log_network_checksum()

            # Apply optimizer rule and update network params (using the param gradients saved after the backward() call)
            # optimizer.step()
            self.scaler.step(self.optimizer)

            self.scaler.update()

            # ------------------------------------------------

            train_loss_epoch += train_loss_batch.item()
            train_loss_classif_term += self.train_loss_function.train_loss_classif_term
            train_loss_reg_term += self.train_loss_function.train_loss_reg_term

            _, labels_pred = torch.max(outputs.data, 1)
            incorrect_preds += (labels_pred != labels).sum().item()

            num_batches += 1
            num_samples += len(inputs)

            self.steps += 1

            if self.is_primary:
                if self.steps % self.memory_record_steps == 0:
                    self._record_memory_usage(self.steps)   # Record memory *after* each n steps

        if self.is_primary:
            # We calculate the gradients of the loss function terms only for the last batch (to get some idea)
            self._log_network_checksum()
            # self._log_loss_term_grads(inputs, labels)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        train_loss_epoch = float(train_loss_epoch / num_batches)

        self.train_loss_classif_term = float(train_loss_classif_term / num_batches)
        self.train_loss_reg_term = float(train_loss_reg_term / num_batches)

        train_error = incorrect_preds * 100 / num_samples

        return train_loss_epoch, train_error

    def _do_evaluations(self, test_loader, natural_loader, attacks_loader):
        do_test_set_eval = False
        do_attack_set_eval = False

        # print(f'epoch: {epochs}, test_set_eval_epochs: {self.test_set_eval_epochs}, ')

        if self.epoch % self.test_set_eval_epochs == 0:
            do_test_set_eval = True

        if self.epoch % self.attack_set_eval_epochs == 0:
            do_attack_set_eval = True

        # All evaluations are done for the first the last epochs to round up the plots
        if self.epoch == 1 or self.epoch == self.epochs:
            do_test_set_eval = True
            do_attack_set_eval = True

        results = {}

        if do_test_set_eval:
            test_loss, test_error, group_errors = self._test_one_epoch(test_loader)
            results['test_loss'] = test_loss
            results['test_error'] = test_error
            results['group_errors'] = group_errors

        if do_attack_set_eval and attacks_loader is not None:
            attacks_loss, attacks_error, attacks_group_errors = self._test_one_epoch(attacks_loader)
            (avg_spearman, avg_kendall, avg_intersection), _ = self._test_robustness_one_epoch(natural_loader,attacks_loader)

            results['attacks_loss'] = attacks_loss
            results['attacks_error'] = attacks_error
            results['attacks_group_errors'] = attacks_group_errors
            results['robustness_spearman'] = avg_spearman
            results['robustness_kendall'] = avg_kendall
            results['robustness_intersection'] = avg_intersection

        # Set these variables here, so they can be used to check if tensorboard metrics must be plotted for the epoch
        self.do_test_set_eval = do_test_set_eval
        self.do_attack_set_eval = do_attack_set_eval and attacks_loader is not None

        return results

    def _test_one_epoch(self, test_loader):
        # return -1, -1, {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}     # Dummy function call for perf debugging

        num_samples = 0
        num_batches = 0
        test_loss_epoch = 0.0
        num_incorrect_preds = 0

        self.network.eval()     # switch to evaluate mode

        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(self.input_gradients)    # Similar to with no_grad(), but able to enable/ disable it

        group_error_calculator = GroupErrorCalculator(self.device_obj)

        for inputs, labels, *metadata in test_loader:
            inputs.requires_grad = self.input_gradients

            inputs = inputs.to(self.device_obj, non_blocking=True)
            labels = labels.to(self.device_obj, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.network(inputs)

            _, y_pred = torch.max(outputs.data, 1)

            test_loss_epoch += self.test_loss_function(outputs, labels, inputs, self.epoch).item()

            is_incorrect_preds = (y_pred != labels)
            num_incorrect_preds += is_incorrect_preds.sum().item()

            num_batches += 1
            num_samples += len(inputs)

            if len(metadata) > 0:
                group_error_calculator.add_batch(metadata, labels, is_incorrect_preds)

        test_loss = float(test_loss_epoch / num_batches)
        test_error = num_incorrect_preds * 100 / num_samples

        torch.set_grad_enabled(prev_grad_state)     # Critical: restore grad_enabled status

        group_errors = group_error_calculator.compute_group_errors()

        return test_loss, test_error, group_errors

    def _test_robustness_one_epoch(self, natural_loader, attack_loader):
        # return -1, -1, -1     # Dummy function call for perf debugging

        # print('_test_robustness_one_epoch')

        eval_top_K = self.params.get('eval_top_K', 100)     # 100 for CIFAR-10, 1000 for ImageNet

        self.network.eval()     # switch to evaluate mode

        # print('Computing IG maps for natural images')
        natural_ig = self._compute_ig_maps_normalized(natural_loader)

        # print('Computing IG maps for attack images')
        attacks_ig = self._compute_ig_maps_normalized(attack_loader)

        spearman_list = []
        kendall_list = []
        intersection_list = []

        # print('Computing robustness metrics on IG maps')

        for x_natural_ig, x_attack_ig in zip(natural_ig, attacks_ig):
            if x_natural_ig.numel() > 2000:
                assert eval_top_K > 500, 'Large images must have a large K value for top-k evaluation'

            spearman = attribution_robustness_metrics.compute_spearman_rank_correlation(x_natural_ig, x_attack_ig)
            kendall = attribution_robustness_metrics.compute_kendalls_correlation(x_natural_ig, x_attack_ig)
            inersection = attribution_robustness_metrics.compute_topK_intersection(x_natural_ig, x_attack_ig, K=eval_top_K)

            spearman_list.append(spearman)
            kendall_list.append(kendall)
            intersection_list.append(inersection)

        avg_spearman = np.mean(spearman_list)
        avg_kendall = np.mean(kendall_list)
        avg_intersection = np.mean(intersection_list)

        return (avg_spearman, avg_kendall, avg_intersection), (spearman_list, kendall_list, intersection_list)

    def _create_optimizer(self, optimizer_name, network, params):
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        momentum = params['momentum']
        lr_scheduler_num_steps = params['lr_scheduler_num_steps']
        lr_scheduler_step_size = max(params['epochs'] // lr_scheduler_num_steps, 1)
        lr_scheduler_gamma = params.get('lr_scheduler_gamma', 1)  # Disabled by default

        lr_scheduler = None

        if optimizer_name == 'sgd':
            print(f'Creating SGD optimizer. lr: {learning_rate}, momentum: {momentum}, weight_decay: {weight_decay}')
            optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size,
                                                     gamma=lr_scheduler_gamma)
        elif optimizer_name == 'adam':
            print(f'Creating Adam optimizer. lr: {learning_rate}, weight_decay: {weight_decay}')
            optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adagrad':
            print('Creating AdaGrad optimizer. lr: {learning_rate}, weight_decay: {weight_decay}')
            optimizer = optim.Adagrad(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

        return optimizer, lr_scheduler

    def _load_states_to_trainer(self, loaded_state):

        # Since we are expecting to train a neural network, we don't need classifier weights (they are random init)
        model_utility.load_weights_to_network(self.params['model'], self.network, loaded_state, load_classifier_weights=False)

        # Loading the optimizer attempts to resume training causes shape mismatch errors
        # self.optimizer.load_state_dict(loaded_state['optimizer_state_dict'])

        if 'scaler_state_dict' in loaded_state:
            self.scaler.load_state_dict(loaded_state['scaler_state_dict'])

        perf_metrics = loaded_state['perf_metrics']
        print('Perf metrics from the loaded model')
        pprint(perf_metrics)

    def _log_network_checksum(self):
        # print(f'Epoch {self.epoch}. Network info')

        state_dict = self.network.state_dict()
        checksum = hashlib.md5(str(state_dict).encode()).hexdigest()
        # print(f'\tNetwork all weights checksum: {checksum}')

        self.tb_writer.add_text('network_weights_checksum', checksum, self.epoch)

    def _log_loss_term_grads(self, inputs, labels):
        outputs = self.network(inputs)
        train_loss_batch = self.train_loss_function(outputs, labels, inputs)
        classif_term = self.train_loss_function.train_loss_classif_term
        reg_term = self.train_loss_function.train_loss_reg_term

        weights_sum, total_loss_grad_sum = self.__compute_loss_term_grad_sum(train_loss_batch)

        classif_term_grad_sum = 0
        reg_term_grad_sum = 0

        # Separately calculate grads for classification and regularization terms, only if reg_term is a tensor
        if torch.is_tensor(reg_term):
            _, classif_term_grad_sum = self.__compute_loss_term_grad_sum(classif_term)
            _, reg_term_grad_sum = self.__compute_loss_term_grad_sum(reg_term)
        
        self.tb_writer.add_scalar('training/network_weights_sum', weights_sum, self.epoch)

        self.tb_writer.add_scalars('training/loss_terms_gradient_sums', {'classif_term_grad': classif_term_grad_sum,
                                                'reg_term_grad': reg_term_grad_sum,
                                                'total_loss_grad': total_loss_grad_sum}, self.epoch)

    def __compute_loss_term_grad_sum(self, loss_term):
        # The backward() call in this function causes the training loop to hang in distributed training mode
        assert False
        self.optimizer.zero_grad()
        loss_term.backward(retain_graph=True)
        weights_sum, weights_grad_sum = self.__weight_sum_grad_sum()
        return weights_sum, weights_grad_sum

    def __weight_sum_grad_sum(self):
        weight_sum = 0.0
        weight_grad_sum = 0.0

        # Iterate over parameters and access their gradients
        for name, parameter in self.network.named_parameters():
            weight_sum += torch.sum(parameter)
            if parameter.grad is not None:
                g = parameter.grad
                weight_grad_sum += torch.sum(g)

        return weight_sum, weight_grad_sum

    def _compute_ig_maps_normalized(self, data_loader):
        for inputs, labels in data_loader:
            x_shape = inputs[0].shape
            break

        ig_batches = []
        baselines = torch.zeros((1, *x_shape)).to(self.device_obj)

        m = 24

        for inputs, labels in data_loader:
            inputs = inputs.to(self.device_obj, non_blocking=True)
            labels = labels.to(self.device_obj, non_blocking=True)
            # inputs.requires_grad = True

            ig_batch = compute_integrated_gradients(self.network, inputs, labels, baselines, m)

            ig_batch = torch.sum(torch.abs(ig_batch), dim=1)  # Sum across channels
            ig_batch = ig_batch / torch.sum(ig_batch)  # Normalize the heatmap

            ig_batches.append(ig_batch)

        ig_set = torch.cat(ig_batches, dim=0)

        return ig_set

    def _checkpoint(self, is_final=False):
        n = min(self.params['checkpoint_every_n_epochs'], self.epochs)
        output_dir = self.params['output_dir']

        if n <= 0 and not is_final:    # Checkpointing is disabled & this is not the end of training
            return

        if self.epoch % n != 0 and not is_final:    # Not a checkpoint-able epoch & this is not the end of training
            return

        if is_final:
            model_path = os.path.join(output_dir, f'final_model_state.pt')
        else:
            model_path = os.path.join(output_dir, f'checkpoint_epoch_{self.epoch}.pt')

        # Remove all previous checkpoints
        curr_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if "checkpoint" in f]
        [os.remove(filepath) for filepath in curr_files]

        # Save new checkpoint
        model_state = {'epoch': self.epoch,
                         'model_state_dict': dict(self.network.state_dict()),  # Make a deep copy
                         'optimizer_state_dict': dict(self.optimizer.state_dict()),
                         'scaler_state_dict': self.scaler.state_dict(),
                         'train_loss_function': self.train_loss_function,
                         'perf_metrics': self.results}

        torch.save(model_state, model_path)
        print(f'Checkpoint saved to: {model_path}')

    def _record_memory_usage(self, step):
        program_rss_memory = self.process.memory_info().rss / 1024 / 1024   # in MB
        system_memory = psutil.virtual_memory().used / 1024 / 1024  # in MB

        self.tb_writer.add_scalar('system/program_rss_memory', program_rss_memory, step)
        self.tb_writer.add_scalar('system/system_memory', system_memory, step)

        if self.device_obj.type == 'cuda':
            gpu_memory_mb = torch.cuda.memory_reserved(self.device_obj) / 1024 / 1024  # in MB
            self.tb_writer.add_scalar('system/gpu_memory_mb', gpu_memory_mb, step)

    def _record_tensorboard_metrics(self):
        r = self.results
        ep = self.epoch

        # Training set metrics (available every epoch)

        self.tb_writer.add_scalar('training/train_loss', r["train_loss"], ep)
        self.tb_writer.add_scalar('training/train_error', r["train_error"], ep)
        self.tb_writer.add_scalar('system/epoch_time_secs', r["epoch_time_secs"], ep)
        # self.tb_writer.add_scalar('learning_rate', last_lr, self.epoch)

        self.tb_writer.add_scalars('training/train_loss_terms', {'classification': self.train_loss_classif_term,
                                                        'regularization': self.train_loss_reg_term,
                                                        'total': r["train_loss"]}, ep)

        if self.do_test_set_eval:
            self.tb_writer.add_scalar('test/test_loss', r["test_loss"], ep)
            self.tb_writer.add_scalar('test/test_error', r["test_error"], ep)

            # # Additional plots to see train + test set loss & error on same plot
            # self.tb_writer.add_scalars('training/train_and_test_loss', {'train_loss': r["train_loss"], 'test_loss': r["test_loss"]}, ep)
            # self.tb_writer.add_scalars('training/train_and_test_error', {'train_error': r["train_error"], 'test_error': r["test_error"]}, ep)

            group_errors = r['group_errors']
            if sum(group_errors.values()) > -4.0:  # At least one group has a valid error value
                group_errors_curve = {f'group_{key}': error_val for key, error_val in group_errors.items()}
                self.tb_writer.add_scalars('test/group_errors', group_errors_curve, ep)

        if self.do_attack_set_eval:
            self.tb_writer.add_scalar('attacks/attacks_loss', r["attacks_loss"], ep)
            self.tb_writer.add_scalar('attacks/attacks_error', r["attacks_error"], ep)
            self.tb_writer.add_scalar('attacks/robustness_spearman', r["robustness_spearman"], ep)
            self.tb_writer.add_scalar('attacks/robustness_kendall', r["robustness_kendall"], ep)
            self.tb_writer.add_scalar('attacks/robustness_intersection', r["robustness_intersection"], ep)
