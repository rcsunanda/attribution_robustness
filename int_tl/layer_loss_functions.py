import torch
import torch.nn as nn

from int_tl import layer_ig_attack, layer_gradient_attack
from int_tl.eg_explainer import PathExplainerTorch
from int_tl.layer_feature_significance import compute_layer_gradients


class LayerGradientRegularizationLoss:
    def __init__(self, network, batch_size, params, device_obj):
        print('LayerGradientRegularizationLoss.init()')

        self.network = network

        self.alpha = params['grad_reg_alpha']
        self.ce_loss_obj = nn.CrossEntropyLoss(reduction='none').to(device_obj)
        self.v = torch.ones(batch_size*2).to(device_obj) # To get element-wise gradients of vector valued function (non-reduced loss function)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

        self.layer_activations_tensor = None
        self.layer_activations_tensor_debug = None

        block_idx = params['block_idx']
        sub_block_idx = params['sub_block_idx']

        assert len(block_idx) == len(sub_block_idx)

        for b_id, sb_id in zip(block_idx, sub_block_idx):
            assert 4 <= b_id <= 7   # ResNet-18 block index check
            assert 0 <= sb_id <= 1  # ResNet-18 block index check

            # Register backward hooks for all given layers (so that gradients can be summed)
            print(f'b_id: {b_id}, sb_id: {sb_id}')
            network.conv_base[b_id][sb_id].conv2.register_full_backward_hook(self._backward_hook_sum)

        # Turn this on only within the gradient call in __call__()
        # Turn this on only within the gradient call in __call__()
        self.gradient_sum_mode = False
        self.grad_squared_sum = 0.0

        first_b_id = block_idx[0]
        first_sb_id = sub_block_idx[0]

        # Register a forward hook to the *conv1* first (shallowest) layer, so we can differentiate w.r.t. to it
        # We need to register this forward hook at the *conv1* layer (one before) instead of the conv2 layer,
        # so that all backward hooks are triggered
        self.network.conv_base[first_b_id][first_sb_id].conv1.register_forward_hook(self._forward_hook_save)

        # For debugging (to check if backward hooks give the same gradients)
        self.network.conv_base[first_b_id][first_sb_id].conv2.register_forward_hook(self._forward_hook_save_debug)


    def _forward_hook_save(self, module, input, output):
        # Save the layer activations during forward pass (so we can differentiate w.r.t. to it)
        self.layer_activations_tensor = output

    def _forward_hook_save_debug(self, module, input, output):
        # Save the layer activations during forward pass (so we can differentiate w.r.t. to it)
        self.layer_activations_tensor_debug = output

    def _backward_hook_sum(self, module, grad_ouput, grad_output):
        # print(f'\t====> self.gradient_sum_mode: {self.gradient_sum_mode}')

        if self.gradient_sum_mode is False:
            return

        self.grad_squared_sum += torch.sum(torch.square(grad_output[0])).item()

        # sq = torch.square(grad_output[0])
        # sq_sum = torch.sum(sq)
        # print(f'\t====> grad_output.shape: {grad_output[0].shape}')
        # print(f'\t====> grad_output: min: {torch.min(grad_output[0])}, max: {torch.max(grad_output[0])}')
        # print(f'\t====> squared_grad_output: min: {torch.min(sq)}, max: {torch.max(sq)}')
        # print(f'\t====> sq_sum: {sq_sum}')

    def __call__(self, output, target, x):
        # print(self.grad_squared_sum)
        # assert self.grad_squared_sum == 0   # To ensure we start out with 0

        partial_loss = self.ce_loss_obj(output, target)

        batch_size = x.shape[0]

        # a = partial_loss.reshape(batch_size, 1, 1, 1)
        # b = a.tile(1, 3, 32, 32)

        v = self.v[0: batch_size]   # Because some batches may be smaller than original batch_size (when dataset is very small)

        # # x = x.reshape(batch_size, -1)
        # print(partial_loss.shape)
        # print(x.shape)
        # print(v.shape)

        self.grad_squared_sum = 0   # To ensure next gradient sums start with 0
        self.gradient_sum_mode = True

        dummy = torch.autograd.grad(partial_loss, self.layer_activations_tensor, grad_outputs=v, retain_graph=True)[0]
        # print(f'grad.shape: {grad.shape}')
        self.gradient_sum_mode = False

        # grad = torch.autograd.grad(partial_loss, self.layer_activations_tensor_debug, grad_outputs=v, retain_graph=True)[0]

        # grad = torch.zeros(10, 10)

        # grad_reg_term = torch.sum(torch.square(grad)) / batch_size

        # sq = torch.square(grad)
        # sq_sum = torch.sum(torch.square(grad))
        #
        # print(f'\t@@@@> grad_output.shape: {grad.shape}')
        # print(f'\t@@@@> grad_output: min: {torch.min(grad)}, max: {torch.max(grad)}')
        # print(f'\t@@@@> squared_grad_output: min: {torch.min(sq)}, max: {torch.max(sq)}')
        # print(f'\t@@@@> sq_sum: {sq_sum}')

        # Need item() to prevent memory leak, but it will also make the loss function indifferentiable
        # We comment this out to save compute. Uncomment when you need to debug
        # self.train_loss_classif_term = torch.mean(partial_loss).item()    # Need item() to prevent memory leak

        grad_reg_term = self.grad_squared_sum / batch_size

        self.train_loss_reg_term = self.alpha * grad_reg_term

        # print(f'partial_loss: {self.train_loss_classif_term}, grad_reg_term: {grad_reg_term}')

        total_loss = torch.mean(partial_loss) + self.train_loss_reg_term

        return total_loss


# --------------------------------------

class LayerGradAttackRegularizationLoss:
    def __init__(self, network, params, device_obj):
        print('LayerGradAttackRegularizationLoss.init()')

        self.network = network

        self.grad_attack_reg_lambda = params['grad_attack_reg_lambda']
        self.epsilon = params['grad_attack_epsilon']
        self.alpha = params['grad_attack_alpha']
        self.num_iter = params['grad_attack_adv_num_iter']

        self.ce_loss_obj = nn.CrossEntropyLoss().to(device_obj)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

    # Train with cross entropy + IG norm
    def __call__(self, output, target, x):
        self.train_loss_classif_term = self.ce_loss_obj(output, target)

        # ------------------------------------
        # Create IG attack examples
        # x.grad.zero_()

        prev_mode = self.network.training  # save original mode to restore
        self.network.eval()     # forward runs in IG must be run in eval mode

        XX_attacks = []
        for example, y in zip(x, target):
            example = example.clone().detach()  # Is this required?

            x_attack = layer_gradient_attack.create_layer_gradient_attack_example(self.network, example, y, self.epsilon, self.alpha, self.num_iter)
            XX_attacks.append(x_attack)

        XX_attacks = torch.stack(XX_attacks)  # Turn the list of tensors into a single tensor with all attack examples

        # ------------------------------------

        grad_diff_norm_sum = layer_gradient_attack.compute_layer_gradient_diff_norm_sum(self.network, x, target, XX_attacks)
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

class LayerIGRegularizationLoss:
    def __init__(self, network, params, device_obj):
        print('LayerIGRegularizationLoss.init()')

        self.network = network

        self.ig_reg_lambda = params['ig_reg_lambda']
        self.norm_ig_steps = params['ig_norm_ig_steps']
        self.epsilon = params['ig_epsilon']
        self.alpha = params['ig_alpha']
        self.num_iter = params['ig_adv_num_iter']
        self.adv_ig_steps = params['ig_adv_ig_steps']

        self.block_idx = params['block_idx']
        self.sub_block_idx = params['sub_block_idx']

        assert 4 <= self.block_idx <= 7  # ResNet-18 block index check
        assert 0 <= self.sub_block_idx <= 1  # ResNet-18 block index check

        self.ce_loss_obj = nn.CrossEntropyLoss().to(device_obj)

        self.train_loss_classif_term = -1
        self.train_loss_reg_term = -1

    # Train with cross entropy + IG norm
    def __call__(self, output, target, x):
        self.train_loss_classif_term = self.ce_loss_obj(output, target)

        # ------------------------------------
        # Create IG attack examples
        # x.grad.zero_()

        prev_mode = self.network.training  # save original mode to restore
        self.network.eval()     # forward runs in IG must be run in eval mode

        XX_attacks = []
        for example, y in zip(x, target):
            example = example.clone().detach()  # Is this required?

            x_attack = layer_ig_attack.create_layer_ig_attack_example(self.network, example, y, self.block_idx, self.sub_block_idx,
                                                                      self.epsilon, self.alpha, self.num_iter, self.adv_ig_steps)
            XX_attacks.append(x_attack)

        XX_attacks = torch.stack(XX_attacks)  # Turn the list of tensors into a single tensor with all attack examples

        # ------------------------------------

        ig_norm_sum = layer_ig_attack.compute_layer_ig_norm_sum_in_batches(self.network, x, target, XX_attacks,
                                               self.block_idx, self.sub_block_idx, self.norm_ig_steps, batch_size=16)
        # ig_norm_sum = 0

        batch_size = x.shape[0]
        ig_norm_mean = ig_norm_sum / batch_size

        self.train_loss_reg_term = self.ig_reg_lambda * ig_norm_mean

        total_loss = self.train_loss_classif_term + self.train_loss_reg_term

        # print(f'partial_loss: {self.train_loss_classif_term}, ig_norm_mean: {ig_norm_mean}, ig_reg_loss_term: {self.train_loss_reg_term}, total_loss: {total_loss}')

        # x.grad.zero_()

        self.network.train(prev_mode)  # Restore original mode

        return total_loss

    # # Train with cross entropy loss of only the IG attacks of a batch
    # def __call__(self, output, target, x):
    #     # Create IG attack examples
    #     XX_attacks = []
    #     for example, y in zip(x, target):
    #         x_attack = ig_attack.create_ig_attack_example(self.network, example, y, self.epsilon, self.alpha, self.num_iter, self.adv_ig_steps)
    #         XX_attacks.append(x_attack)
    #
    #     XX_attacks = torch.stack(XX_attacks)  # Turn the list of tensors into a single tensor with all attack examples
    #
    #     # May need a grad_zero() call here ???
    #
    #     XX_attacks_output = self.network(XX_attacks)
    #
    #     attacks_ce_loss = self.ce_loss_obj(XX_attacks_output, target)
    #
    #     print(f'attacks_ce_loss: {attacks_ce_loss}')
    #
    #     return attacks_ce_loss


# --------------------------------------
#
# # Regularize to have small squared norm of Expected Gradients maps
# class LayerEGRegularizationLoss:
#     def __init__(self, network, params, device_obj):
#           print('LayerEGRegularizationLoss.init()')

#         self.network = network
#
#         self.eg_reg_lambda = params['eg_reg_lambda']
#         self.eg_num_refs = params['eg_num_refs']  # the k value
#         self.eg_batch_size = params['eg_batch_size']
#
#         self.ce_loss_obj = nn.CrossEntropyLoss(reduction='none').to(device_obj)
#         self.eg_explainer = PathExplainerTorch(network)
#
#         self.train_loss_classif_term = -1
#         self.train_loss_reg_term = -1
#
#     def __call__(self, output, target, x):
#         partial_loss = self.ce_loss_obj(output, target)
#         batch_size = x.shape[0]
#
#         prev_mode = self.network.training  # save original mode to restore
#         self.network.eval()  # forward runs in EG must be run in eval mode
#
#         # !!!!!!!! Can we take Expected Gradients of the loss term? !!!!!!!!!!
#
#         # ----- Compute Expected Gradients and squared norm sum
#
#         eg_squared_norm_sum = 0
#
#         for i in range(0, batch_size, self.eg_batch_size):
#             XX = x[i: i + self.eg_batch_size]
#             yy = target[i: i + self.eg_batch_size]
#
#             XX_eg = self.eg_explainer.attributions(input_tensor=XX,
#                                            baseline=XX,
#                                            num_samples=self.eg_num_refs,
#                                            use_expectation=True,
#                                            output_indices=yy)
#
#             eg_squared_norm_sum += torch.sum(torch.square(XX_eg)).item()
#
#         # --------------------
#
#         self.train_loss_classif_term = torch.mean(partial_loss)
#         eg_reg_term = eg_squared_norm_sum / batch_size
#
#         self.train_loss_reg_term = self.eg_reg_lambda * eg_reg_term
#
#         total_loss = self.train_loss_classif_term + self.train_loss_reg_term
#
#         # print(f'partial_loss: {self.train_loss_classif_term},\teg_reg_term: {eg_reg_term},\ttotal_loss: {total_loss}')
#
#         self.network.train(prev_mode)  # Restore original mode
#
#         return total_loss
