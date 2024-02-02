import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from int_tl import ig_attack, gradient_attack
from int_tl.eg_explainer import PathExplainerTorch
from int_tl.layer_loss_functions import LayerGradientRegularizationLoss, LayerIGRegularizationLoss, \
    LayerGradAttackRegularizationLoss
from utility import parameter_setup


def get_loss_function(loss_function_name, network, device_obj, batch_size, params):
    extra_str = None
    if '--' in loss_function_name:
        extra_str = loss_function_name.split('--')[-1]  # Get the extra string after the '--' separator

    loss_function_name = loss_function_name.split('--')[0]  # Get the loss function name before the '--' separator

    if loss_function_name == 'cross_entropy':
        # loss_function = nn.CrossEntropyLoss()
        loss_function = CrossEntropyLoss(device_obj)
    elif loss_function_name == 'gradient_regularization':
        loss_function = GradientRegularizationLoss(batch_size, params['grad_reg_alpha'], device_obj)
    elif loss_function_name == 'ig_regularization':
        loss_function = IGRegularizationLoss(network, params, device_obj, extra_str)
    elif loss_function_name == 'gradient_attack_regularization':
        loss_function = GradAttackRegularizationLoss(network, params, device_obj)
    elif loss_function_name == 'eg_regularization':
        loss_function = EGRegularizationLoss(network, params, device_obj)

    # Layer regularization losses
    elif loss_function_name == 'layer_gradient_regularization':
        loss_function = LayerGradientRegularizationLoss(network, batch_size, params, device_obj)
    elif loss_function_name == 'layer_ig_regularization':
        loss_function = LayerIGRegularizationLoss(network, params, device_obj)
    elif loss_function_name == 'layer_gradient_attack_regularization':
        loss_function = LayerGradAttackRegularizationLoss(network, params, device_obj)

    elif loss_function_name == 'changing':
        loss_function = ChangingLoss(network, params, device_obj)
    elif loss_function_name == 'every_other':
        loss_function = EveryOtherChangingLoss(network, params, device_obj)
    else:
        assert False, f'Unknown loss function: {loss_function_name}'

    return loss_function


# --------------------------------------

class CrossEntropyLoss:
    def __init__(self, device_obj):
        print('CrossEntropyLoss.init()')

        self.ce_loss_obj = nn.CrossEntropyLoss().to(device_obj)
        self.train_loss_classif_term = -1
        self.train_loss_reg_term = 0

    def __call__(self, output, target, x, epoch):
        # print('CrossEntropyLoss')
        self.train_loss_classif_term = self.ce_loss_obj(output, target)
        return self.train_loss_classif_term


# --------------------------------------

class GradientRegularizationLoss:
    def __init__(self, batch_size, alpha, device_obj):
        print('GradientRegularizationLoss.init()')

        self.alpha = alpha
        self.ce_loss_obj = nn.CrossEntropyLoss(reduction='none').to(device_obj)
        self.v = torch.ones(batch_size*2).to(device_obj) # To get element-wise gradients of vector valued function (non-reduced loss function)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

    def __call__(self, output, target, x, epoch):
        # print(f'output.shape: {output.shape}')
        # print(f'target.shape: {target.shape}')

        partial_loss = self.ce_loss_obj(output, target)
        batch_size = x.shape[0]

        assert False    # Possibly need to switch to eval mode --> self.network.eval() # forward runs in IG must be run in eval mode

        # a = partial_loss.reshape(batch_size, 1, 1, 1)
        # b = a.tile(1, 3, 32, 32)

        v = self.v[0: batch_size]   # Because some batches may be smaller than original batch_size (when dataset is very small)

        # # x = x.reshape(batch_size, -1)
        # print(partial_loss.shape)
        # print(x.shape)
        # print(v.shape)

        # Note: create_graph is necessary to be able to compute the
        # derivative of the regularization term wrt model params in loss.backward() call of the optimizer
        # Without it, the derivatives will be zero (will yield silently incorrect results)
        grad = torch.autograd.grad(partial_loss, x, grad_outputs=v, create_graph=True, only_inputs=True)[0]

        # print(f'grad.shape: {grad.shape}')

        self.train_loss_classif_term = torch.mean(partial_loss)

        grad_reg_term = torch.sum(torch.square(grad)) / batch_size
        self.train_loss_reg_term = self.alpha * grad_reg_term

        # print(f'partial_loss: {self.train_loss_classif_term}, grad_reg_term: {grad_reg_term}')

        total_loss = self.train_loss_classif_term + self.train_loss_reg_term
        
        # return self.train_loss_reg_term
        # return self.train_loss_classif_term
        return total_loss


# --------------------------------------

class GradAttackRegularizationLoss:
    def __init__(self, network, params, device_obj):
        print('GradAttackRegularizationLoss.init()')
        self.network = network

        self.grad_attack_reg_lambda = params['grad_attack_reg_lambda']
        self.epsilon = params['grad_attack_reg_epsilon']
        self.alpha = params['grad_attack_reg_alpha']
        self.num_iter = params['grad_attack_reg_adv_num_iter']

        self.ce_loss_obj = nn.CrossEntropyLoss().to(device_obj)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

    # Train with cross entropy + IG norm
    def __call__(self, output, target, x, epoch):
        self.train_loss_classif_term = self.ce_loss_obj(output, target)

        # ------------------------------------
        # Create IG attack examples
        # x.grad.zero_()

        prev_mode = self.network.training  # save original mode to restore
        self.network.eval()     # forward runs in IG must be run in eval mode

        XX_attacks = []
        for example, y in zip(x, target):
            example = example.clone().detach()  # Is this required?

            x_attack = gradient_attack.create_gradient_attack_example(self.network, example, y, self.epsilon, self.alpha, self.num_iter)
            XX_attacks.append(x_attack)

        XX_attacks = torch.stack(XX_attacks)  # Turn the list of tensors into a single tensor with all attack examples

        # ------------------------------------

        grad_diff_norm_sum = gradient_attack.compute_gradient_diff_norm_sum(self.network, x, target, XX_attacks)
        # grad_diff_norm_sum = 0

        batch_size = x.shape[0]
        ig_norm_mean = grad_diff_norm_sum / batch_size

        self.train_loss_reg_term = self.grad_attack_reg_lambda * ig_norm_mean

        total_loss = self.train_loss_classif_term + self.train_loss_reg_term

        # print(f'partial_loss: {self.train_loss_classif_term}, ig_norm_mean: {ig_norm_mean}, ig_reg_loss_term: {self.train_loss_reg_term}, total_loss: {total_loss}')

        # x.grad.zero_()

        self.network.train(prev_mode)  # Restore original mode

        return total_loss


# --------------------------------------

class IGRegularizationLoss:
    def __init__(self, network, params, device_obj, extra_str):
        print('IGRegularizationLoss.init()')

        self.network = network

        self.ig_reg_lambda = params['ig_reg_lambda']
        self.norm_ig_steps = params['ig_norm_ig_steps']
        self.epsilon = params['ig_epsilon']
        self.alpha = params['ig_alpha']
        self.num_iter = params['ig_adv_num_iter']
        self.adv_ig_steps = params['ig_adv_ig_steps']
        ig_loss_func = params['ig_loss_func']

        self.coefficient_1 = 1
        if extra_str is not None and extra_str == 'percent_lambda':
            assert 0 <= self.ig_reg_lambda <= 1
            self.coefficient_1 = 1 - self.ig_reg_lambda
            print(f'IGRegularizationLoss.init(). coefficient_1: {self.coefficient_1}')

        self.ce_loss_obj = nn.CrossEntropyLoss().to(device_obj)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

        # self.loss_func = self.ce_ig_norm

        print(f'IGRegularizationLoss. Loss function = {ig_loss_func}')

        if ig_loss_func == 'ce_ig_norm':
            self.loss_func = self.ce_ig_norm
        elif ig_loss_func == 'ce_of_attacks':
            self.loss_func = self.ce_of_attacks
        elif ig_loss_func == 'ce_of_natural_and_attacks':
            self.loss_func = self.ce_of_natural_and_attacks
        elif ig_loss_func == 'ce_of_natural_and_ce_of_attacks_to_natural_output':
            self.loss_func = self.ce_of_natural_and_ce_of_attacks_to_natural_output
        elif ig_loss_func == 'ce_grad_diff_norm':
            self.loss_func = self.ce_grad_diff_norm
        else:
            assert False

    def __call__(self, output, target, x, epoch):
        return self.loss_func(output, target, x)

    # Train with cross entropy + IG norm
    def ce_ig_norm(self, output, target, x):
        self.train_loss_classif_term = self.coefficient_1 * self.ce_loss_obj(output, target)

        # ------------------------------------
        # Create IG attack examples
        # x.grad_wrt_x_attacks.zero_()

        prev_mode = self.network.training  # save original mode to restore
        self.network.eval()     # forward runs in IG must be run in eval mode

        XX_attacks = ig_attack.compute_pgd_attacks(self.network, x, target, self.epsilon, self.alpha, self.num_iter, self.adv_ig_steps)

        # print(f'\tce_ig_norm. XX_attacks: {misc.tensor_checksum(XX_attacks)}')

        # ------------------------------------

        # ig_norm_sum = ig_attack.compute_ig_norm_sum_in_batches(self.network, x, target, XX_attacks, self.norm_ig_steps, batch_size=16)
        ig_norm_sum = ig_attack.compute_ig_norm_sum(self.network, x, target, XX_attacks, self.norm_ig_steps)
        # print(ig_norm_sum)
        # ig_norm_sum = 0

        batch_size = x.shape[0]
        ig_norm_mean = ig_norm_sum / batch_size

        self.train_loss_reg_term = self.ig_reg_lambda * ig_norm_mean

        # ------ Debugging ------
        # grad_wrt_x = torch.autograd.grad(ig_norm_sum, x, retain_graph=True)[0]
        # grad_wrt_x_attacks = torch.autograd.grad(ig_norm_sum, XX_attacks, retain_graph=True)[0]
        # print(f'grad_wrt_x.sum(): {torch.sum(grad_wrt_x)}')
        # print(f'grad_wrt_x_attacks.sum(): {torch.sum(grad_wrt_x_attacks)}')
        # -----------------------

        total_loss = self.train_loss_classif_term + self.train_loss_reg_term

        # print(f'partial_loss: {self.train_loss_classif_term}, ig_norm_mean: {ig_norm_mean},
        # ig_reg_loss_term: {self.train_loss_reg_term}, total_loss: {total_loss}')

        self.network.train(prev_mode)  # Restore original mode

        # return self.train_loss_reg_term
        # return self.train_loss_classif_term
        return total_loss

    # Train with cross entropy loss of only the IG attacks of a batch
    def ce_of_attacks(self, output, target, x):
        XX_attacks = ig_attack.compute_pgd_attacks(self.network, x, target, self.epsilon, self.alpha, self.num_iter, self.adv_ig_steps)

        # May need a grad_zero() call here ???

        XX_attacks_output = self.network(XX_attacks)

        attacks_ce_loss = self.ce_loss_obj(XX_attacks_output, target)

        return attacks_ce_loss

    # Train with cross entropy loss of both natural examples the IG attacks of a batch (wrt true label y)
    def ce_of_natural_and_attacks(self, output, target, x):
        # self.train_loss_classif_term = self.coefficient_1 * self.ce_loss_obj(output, target).item()

        XX_attacks = ig_attack.compute_pgd_attacks(self.network, x, target, self.epsilon, self.alpha, self.num_iter, self.adv_ig_steps)

        # May need a grad_zero() call here ???

        XX_attacks_output = self.network(XX_attacks)

        # self.train_loss_reg_term = self.ig_reg_lambda * self.ce_loss_obj(XX_attacks_output, target).item()

        # attacks_ce_loss = self.train_loss_classif_term + self.train_loss_reg_term

        attacks_ce_loss = self.coefficient_1 * self.ce_loss_obj(output, target) + \
                          self.ig_reg_lambda * self.ce_loss_obj(XX_attacks_output, target)

        return attacks_ce_loss

    # Train with cross entropy loss of natural examples + CE of IG attacks wrt logits of natural examples
    def ce_of_natural_and_ce_of_attacks_to_natural_output(self, output, target, x):
        self.train_loss_classif_term = self.coefficient_1 * self.ce_loss_obj(output, target)

        XX_attacks = ig_attack.compute_pgd_attacks(self.network, x, target, self.epsilon, self.alpha, self.num_iter,
                                                   self.adv_ig_steps)

        # May need a grad_zero() call here ???

        XX_attacks_output = self.network(XX_attacks)
        output_probs = F.softmax(output, dim=1)  # Convert output logits to probabilities

        self.train_loss_reg_term = self.ig_reg_lambda * self.ce_loss_obj(XX_attacks_output, output_probs)

        attacks_ce_loss = self.train_loss_classif_term + self.train_loss_reg_term

        return attacks_ce_loss

    def ce_grad_diff_norm(self, output, target, x):
        self.train_loss_classif_term = self.coefficient_1 * self.ce_loss_obj(output, target)

        # ------------------------------------
        prev_mode = self.network.training  # save original mode to restore
        self.network.eval()  # forward runs in IG must be run in eval mode

        XX_attacks = ig_attack.compute_pgd_attacks(self.network, x, target, self.epsilon, self.alpha, self.num_iter,
                                                   self.adv_ig_steps)

        # ------------------------------------

        grad_diff_norm = ig_attack.compute_grad_diff_norm(self.network, x, target, XX_attacks)

        batch_size = x.shape[0]
        grad_diff_norm_mean = grad_diff_norm / batch_size

        self.train_loss_reg_term = self.ig_reg_lambda * grad_diff_norm_mean

        total_loss = self.train_loss_classif_term + self.train_loss_reg_term

        self.network.train(prev_mode)  # Restore original mode

        return total_loss


# --------------------------------------

# Regularize to have small squared norm of Expected Gradients maps
class EGRegularizationLoss:
    def __init__(self, network, params, device_obj):
        print('EGRegularizationLoss.init()')

        self.network = network

        self.eg_reg_lambda = params['eg_reg_lambda']
        self.eg_num_refs = params['eg_num_refs']  # the k value
        self.eg_batch_size = params['eg_batch_size']

        self.ce_loss_obj = nn.CrossEntropyLoss(reduction='none').to(device_obj)
        self.eg_explainer = PathExplainerTorch(network)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

    def __call__(self, output, target, x, epoch):
        partial_loss = self.ce_loss_obj(output, target)
        batch_size = x.shape[0]

        prev_mode = self.network.training  # save original mode to restore
        self.network.eval()  # forward runs in EG must be run in eval mode

        # !!!!!!!! Can we take Expected Gradients of the loss term? !!!!!!!!!!

        # ----- Compute Expected Gradients and squared norm sum

        eg_squared_norm_sum = 0

        for i in range(0, batch_size, self.eg_batch_size):
            XX = x[i: i + self.eg_batch_size]
            yy = target[i: i + self.eg_batch_size]

            XX_eg = self.eg_explainer.attributions(input_tensor=XX,
                                           baseline=XX,
                                           num_samples=self.eg_num_refs,
                                           use_expectation=True,
                                           output_indices=yy)

            eg_squared_norm_sum += torch.sum(torch.square(XX_eg)).item()

        # --------------------

        self.train_loss_classif_term = torch.mean(partial_loss)
        eg_reg_term = eg_squared_norm_sum / batch_size

        self.train_loss_reg_term = self.eg_reg_lambda * eg_reg_term

        total_loss = self.train_loss_classif_term + self.train_loss_reg_term

        # print(f'partial_loss: {self.train_loss_classif_term},\teg_reg_term: {eg_reg_term},\ttotal_loss: {total_loss}')

        self.network.train(prev_mode)  # Restore original mode

        return total_loss


class ChangingLoss:
    def __init__(self, network, params, device_obj):
        print('ChangingLoss.init()')
        self.verify_changing_loss_functions(params)
        self.schedule = self.get_changing_loss_functions_shedule(params, network, device_obj)

        self.loss_functions = [func for epoch_range, func in self.schedule.items()]

        change_points = np.array([start for (start, end) in self.schedule.keys()])
        self.change_points = np.sort(change_points) # Redundant, for safety

        self.network = network

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

        self.current_loss_function = self.loss_functions[0]
        print(f'Starting with loss function: {self.current_loss_function}')

    def verify_changing_loss_functions(self, params):
        schedule = params['changing_loss_functions']
        assert type(schedule) == dict, 'Parameter "changing_loss_functions" must be a dictionary'

        first_range = list(schedule.keys())[0]
        last_range = list(schedule.keys())[-1]

        assert first_range[0] == 1, 'First epoch range must start at 1'
        assert last_range[1] == params['epochs'], 'Last epoch range must end at the last epoch'

        last_range_end = 0
        for epoch_range, loss_function in schedule.items():
            assert loss_function in parameter_setup.allowed_values['loss_function'], 'Invalid loss function in changing_loss_functions'
            range_start = epoch_range[0]
            # print(f'epoch_range: {epoch_range}, range_start: {range_start}, last_range_end: {last_range_end}')
            assert range_start == last_range_end + 1, 'Epoch ranges must be in ascending order with no gaps or overlaps'
            last_range_end = epoch_range[1]

    def get_changing_loss_functions_shedule(self, params, network, device_obj):
        schedule = params['changing_loss_functions']
        batch_size = params['batch_size']

        schedule_functions = {}

        for epoch_range, loss_function_name in schedule.items():
            loss_function_obj = get_loss_function(loss_function_name, network, device_obj, batch_size, params)
            schedule_functions[epoch_range] = loss_function_obj

        return schedule_functions

    def get_loss_function_for_epoch(self, epoch):
        idx = np.searchsorted(self.change_points, epoch, side='right') - 1
        assert 0 <= idx < len(self.loss_functions), f'Invalid loss function index: {idx}'
        return self.loss_functions[idx]

    def __call__(self, output, target, x, epoch):
        # print(f'ChangingLoss.__call__(), epoch: {epoch}')

        loss_func = self.get_loss_function_for_epoch(epoch)

        if self.current_loss_function != loss_func:
            print(f'Changing loss function to: {loss_func}')
            self.current_loss_function = loss_func

        loss = loss_func(output, target, x, epoch)

        self.train_loss_classif_term = loss_func.train_loss_classif_term
        self.train_loss_reg_term = loss_func.train_loss_reg_term

        return loss


class EveryOtherChangingLoss:
    def __init__(self, network, params, device_obj):
        print('EveryOtherChangingLoss.init()')
        self.verify_changing_loss_functions(params)

        self.loss_functions = self.get_changing_loss_functions(params, network, device_obj)

        self.network = network

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

        self.curr_epoch = 1
        self.curr_loss_idx = 0
        print(f'Starting with loss function: {self.loss_functions[0]}')

    def verify_changing_loss_functions(self, params):
        loss_func_names = params['changing_loss_functions']

        assert type(loss_func_names) == list, 'Parameter "changing_loss_functions" must be a list of 2 loss functions'
        assert len(loss_func_names) == 2, 'Parameter "changing_loss_functions" must have 2 loss functions'

    def get_changing_loss_functions(self, params, network, device_obj):
        loss_func_names = params['changing_loss_functions']
        batch_size = params['batch_size']

        loss_functions = []

        for loss_function_name in loss_func_names:
            loss_function_obj = get_loss_function(loss_function_name, network, device_obj, batch_size, params)
            loss_functions.append(loss_function_obj)

        return loss_functions

    def get_loss_function_and_switch(self, epoch):
        if epoch != self.curr_epoch:
            self.curr_loss_idx = (self.curr_loss_idx + 1) % 2   # Switch between 0 and 1
            print(f'Switching loss function to: {self.loss_functions[self.curr_loss_idx]}')

        self.curr_epoch = epoch

        loss_function = self.loss_functions[self.curr_loss_idx]

        return loss_function

    def __call__(self, output, target, x, epoch):
        # print(f'EveryOtherChangingLoss.__call__(), epoch: {epoch}')

        loss_func = self.get_loss_function_and_switch(epoch)

        loss = loss_func(output, target, x, epoch)

        self.train_loss_classif_term = loss_func.train_loss_classif_term
        self.train_loss_reg_term = loss_func.train_loss_reg_term

        return loss
