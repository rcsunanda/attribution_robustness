import logging
import sys, os
import time
from tqdm import tqdm
from pprint import pprint

from tensorboardX import SummaryWriter

tb_writer = None


# Class to redirect all print() outputs and errors to both terminal and log file
class Logger(object):
    initialized_files = []

    def __init__(self, log_filename, stream):
        Logger.initialized_files.append(log_filename)
        self.terminal = stream

        self.log_filename = log_filename
        self.log_file = open(log_filename, "w")

        self.is_error_stream = (stream == sys.stderr)
        print(f'\n------ Initializing Logger object for stream {stream}. '
              f'is_error_stream: {self.is_error_stream}. log_filename: {log_filename} ------\n')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):    # needed for python 3 compatibility.
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        print(f'\n------ Closing Logger object for file: {self.log_filename}. ')
        self.log_file.close()

        # Restore the system output terminals to their original state
        if self.is_error_stream:
            sys.stderr = self.terminal
        else:
            sys.stdout = self.terminal


def init_logging(params, create_folder=True):
    exp_id = params['exp_id']
    job_id = params['job_id']

    if create_folder:
        output_dir = f'{params["output_dir"]}/{exp_id}_{job_id}_{time.strftime("%b%d-%H_%M_%S")}'
        params['output_dir'] = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = params['output_dir']

    clear_all_figures()
    clear_all_dataframes()

    # A pair of files for each process [main process and primary training loop process] (stdout and stderr)
    pid = os.getpid()

    log_filename = f'{output_dir}/experiment_log_{pid}.txt'
    err_filename = f'{output_dir}/experiment_err_{pid}.txt'

    print(f'Logger.initialized_files: {Logger.initialized_files}')

    if log_filename not in Logger.initialized_files:
        if isinstance(sys.stdout, Logger):  # A previously initialized logger exists
            sys.stdout.close()

        sys.stdout = Logger(log_filename, sys.stdout)

    if err_filename not in Logger.initialized_files:
        if isinstance(sys.stderr, Logger):  # A previously initialized logger exists
            sys.stderr.close()

        sys.stderr = Logger(err_filename, sys.stderr)


# ---------------------------------------------------------
# Tensorboard SummaryWriter global object management
def init_tensorboard(params):
    tb_output_dir = params['output_dir'] + '/tensorboard'
    global tb_writer

    # None check allows multiple potential inits, so single-node and distributed training can have the same code path
    # The path check allows for multiple experiments (from Excel param rows) to be run in the same script run
    if tb_writer is None or not os.path.exists(tb_output_dir):
        tb_writer = SummaryWriter(tb_output_dir, comment='AAAAAAAAAAA', flush_secs=20)

    return tb_writer


def get_tensorboard_writer():
    global tb_writer
    # assert tb_writer is not None, 'Tensorboard writer not initialized'
    if tb_writer is None:
        print('\n!!! Tensorboard writer not initialized !!!\n')

    return tb_writer

# ---------------------------------------------------------
# Global item management (figures, dataframes to save etc)

# A global list to keep figures and dataframes (to be saved at the end of the program)
figures_list = []
dataframes_list = []

def add_figure_to_save(fig, name=None):
    print('Adding figure to save: ', name)
    figures_list.append((fig, name))

def add_dataframe_to_save(df, name=None):
    dataframes_list.append((df, name))

def clear_all_figures():
    for (fig, name) in figures_list:
        fig.clear()
    figures_list.clear()

def clear_all_dataframes():
    dataframes_list.clear()


def save_all_figures(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    filenames = []

    for i, (fig, name) in enumerate(figures_list):
        figname = ''
        if name: figname = '_' + name

        filename = dir + '/fig_' + str(i+1) + figname + '.png'
        fig.savefig(filename)

        filenames.append(filename)

    print(f'All figures ({len(filenames)}) saved to {dir}')
    pprint(f'Figure filenames: {filenames}')


def save_all_dataframes(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    for i, (df, name) in enumerate(dataframes_list):
        dfname = ''
        if name: dfname = '_' + name

        filename = dir + '/df_' + str(i+1) + dfname + '.csv'
        df.to_csv(filename)
    print('All dataframes saved to {}'.format(dir))

