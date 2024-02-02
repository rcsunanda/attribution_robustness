import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import os
import numpy as np
#
from models.cnn import ConvolutionalNetwork
from models.ann import FullyConnectedNetwork
from models.resnets import ResNet18
from models.model_trainer import ModelTrainer
from models.model_utility import create_network

from utility import data_prep, attack_data_prep


class Pipeline:
    """ A class that wraps functionality of a training pipeline (data loading, training & testing)
    One object of this class must be created per process (in DistributedModelTrainer._initiate_pipeline)
    The main function of this class is run_pipeline()
    """
    def __init__(self, gpu_id, params):
        self.gpu_id = gpu_id
        self.params = dict(params)
        self._setup_process(gpu_id)
        self._set_random_seed(params['random_seed'])
        self._set_device(gpu_id, params)

    def run_pipeline(self,):
        # ---------------------------------
        # Load data

        trainset, testset, *metadata = data_prep.load_data(self.params)

        natural_loader, attacks_loader = attack_data_prep.load_attack_data(self.params)

        # display_images(trainset)
        # data_prep.visualize_data_samples(trainset, num_samples=8, tb_writer=params['tb_writer'], tb_tag='train_samples')
        # data_prep.visualize_data_projections(trainset, num_samples=100, tb_writer=params['tb_writer'])

        # ---------------------------------
        network = create_network(self.params)

        loaded_state = None
        if self.params['transfer_type'] in ['fixed_feature', 'full_network']:
            print('Loading state for transfer (model & scalar states)')
            # Map model to be loaded to specified single gpu (required to prevent overlaps into the same gpu (0) where model was saved from)
            map_loc = {'cuda:0': f'cuda:{self.gpu_id}'}
            loaded_state = torch.load(self.params['model_state_path'], map_location=map_loc)

        # visualize_network(network, trainset, params['input_shape'], params['tb_writer'])

        # ---------------------------------
        # Actual training loop for this process

        self.trainer = ModelTrainer(network, self.device_obj, self.params, loaded_state)
        self.history = self.trainer.train(trainset, testset, natural_loader, attacks_loader)
        self.trained_model = network

        # ---------------------------------

        self._cleanup()

    def _setup_process(self, gpu_id):
        # Prepare process variables and initialize process
        node_id = self.params['node_id']
        num_gpus_per_node = self.params['num_gpus_per_node']
        world_size = self.params['world_size']
        master_proc_string = self.params['master_proc_string']

        global_rank = node_id * num_gpus_per_node + gpu_id

        self.params['gpu_id'] = gpu_id
        self.params['global_rank'] = global_rank

        print(f'Running distributed training process with global_rank={global_rank} on node_id: {node_id}, gpu_id: {gpu_id}')

        if self.params['distributed_training']:
            dist.init_process_group(backend='nccl', init_method=master_proc_string, rank=global_rank, world_size=world_size)

    def _set_random_seed(self, seed):
        # Set random seed (important for all processes to initialize network to same weights)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _set_device(self, gpu_id, params):
        # Set device (CPU or GPU with correct ID)

        assert params['device'] in ['cpu', 'cuda']

        if params['device'] == 'cuda':
            print(f'Running on GPU (CUDA)')
            assert torch.cuda.is_available()
            params['device'] = f'cuda:{gpu_id}'
            torch.cuda.set_device(params['device'])
        elif params['device'] == 'cpu':
            print('Running on CPU')

        self.device_obj = torch.device(params['device'])  # eg: cpu, cuda:0, cuda:1

    def _cleanup(self):
        if self.params['distributed_training']:
            dist.destroy_process_group()


# -------------------------------------------------------------------------------------------


class DistributedModelTrainer:
    def __init__(self, params):
        self.params = params
        self.pipelines = {}
        self._setup_distributed_env_params()

    def run_training(self):
        if self.params['distributed_training']:
            self._run_distributed_training()
        else:
            self._run_single_node_training()

    def get_primary_model_trainer(self):
        print(f'self.pipelines: {self.pipelines}')
        return self.pipelines[0].trainer

    def _setup_distributed_env_params(self):
        # ----------------------------------------
        # Following environment variables must be set with proper values on each node
        node_id = int(os.environ['NODE_ID'])
        num_gpus_per_node = int(os.environ['NUM_GPUS_PER_NODE'])
        num_nodes = int(os.environ['NUM_NODES'])
        master_address = os.environ['MASTER_ADDR']
        master_port = 12358  # We hard code the port assuming it is free
        # ----------------------------------------

        world_size = num_nodes * num_gpus_per_node
        master_proc_string = f'tcp://{master_address}:{master_port}'

        self.params['node_id'] = node_id
        self.params['num_gpus_per_node'] = num_gpus_per_node
        self.params['num_nodes'] = num_nodes
        self.params['world_size'] = world_size
        self.params['master_proc_string'] = master_proc_string

        print(
            f'node_id: {node_id}, num_gpus_per_node: {num_gpus_per_node}, num_nodes: {num_nodes}, world_size: {world_size}')
        print(f'master_proc_string: {master_proc_string}')

    def _run_distributed_training(self):
        num_processes = self.params['num_gpus_per_node']
        node_id = self.params['node_id']
        print(f'Spawning {num_processes} processes on node_id: {node_id}')
        # mp.spawn(_run_pipeline, nprocs=num_processes, args=(self.params,), join=True)
        self.pipelines[-1] = 'Test add before spawn'
        mp.spawn(self._initiate_pipeline, nprocs=num_processes, args=(self.params,), join=True)
        self.pipelines[-2] = 'Test add after spawn'

    def _run_single_node_training(self):
        print(f'Running single node training')
        self._initiate_pipeline(0, self.params)

    def _initiate_pipeline(self, gpu_id, params):
        """This function runs 1 instance per process (1 process per GPU - this is the unit function of parallelism)
        This function should be given to mp.spawn() to be called once per process
        It can also be called manually to run in the current thread (as a usual function)"""
        # print(f'TEST. gpu_id: {gpu_id}, type(params): {type(params)}')
        pipeline = Pipeline(gpu_id, params)
        self.pipelines[gpu_id] = pipeline
        pipeline.run_pipeline()




