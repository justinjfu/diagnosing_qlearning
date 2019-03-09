"""
This script runs a sweep over number of samples (using replay buffer or pure on-policy data)

Plot results using plot_sampling.py 
"""
import time
import argparse

from debugq.algos import exact_fqi, sampling_fqi, replay_buffer_fqi, stopping
from debugq.models import q_networks
from debugq.envs import random_obs_wrapper, time_limit_wrapper
from debugq.envs import env_suite
from debugq import pytorch_util as ptu

from rlutil.logging import log_utils
from rlutil import hyper_sweep


def main(exp_prefix='exp', validation_stop=True, layers=(32, 32), repeat=0, env_name='grid1', sampling_type=None, **alg_args):
    env = env_suite.get_env(env_name)

    if layers == 'tabular':
        network = q_networks.TabularNetwork(env)
    else:
        network = q_networks.FCNetwork(env, layers=layers)
    ptu.initialize_network(network)

    stop_mode = (stopping.AtolStop(), stopping.RtolStop())

    if sampling_type.endswith('512'):
        num_samples = 512
    elif sampling_type.endswith('256'):
        num_samples = 256
    elif sampling_type.endswith('128'):
        num_samples = 128
    elif sampling_type.endswith('64'):
        num_samples = 64
    elif sampling_type.endswith('32'):
        num_samples = 32
    elif sampling_type.endswith('16'):
        num_samples = 16
    elif sampling_type.endswith('8'):
        num_samples = 8
    elif sampling_type.endswith('4'):
        num_samples = 4

    alg_args.update({
        'min_project_steps': 10,
        'max_project_steps': 200,
        'lr': 5e-3,
        'discount': 0.95,
        'n_steps': 1,
        'batch_size': 128,
        'stop_modes': stop_mode,
        'time_limit': env.time_limit,
        'env_name': env_name,
        'backup_mode': 'sampling',
        'layers': str(layers),
        'num_samples': num_samples,
        'log_sampling_type': sampling_type
    })

    if sampling_type.startswith('buffer'):
        fqi = replay_buffer_fqi.TabularBufferDQN(env, network, log_proj_qstar=False, **alg_args)
    elif sampling_type.startswith('sample'):
        fqi = sampling_fqi.WeightedSamplingFQI(env, network, log_proj_qstar=False, **alg_args)

    with log_utils.setup_logger(algo=fqi, exp_prefix=exp_prefix, log_base_dir='/data') as log_dir:
        print('Logging to %s' % log_dir)
        total_num_samples = 0
        k = 0
        try:
            while (total_num_samples < 128*200) and (k <= 2000):
                fqi.update(step=k)
                total_num_samples += num_samples
                k += 1
        except:
            log_utils.save_exception()

if __name__ == "__main__":
    args = {
        'exp_prefix': ['overfitting2'],
        'weighting_scheme': ['none'],
        'sampling_policy': ['pi', 'random'],
        'sampling_type': ['buffer4', 'buffer8', 'buffer32', 'buffer16', 'sample32', 'sample128', 'sample256', 'sample512'],
        'resample_per_step': [False],
        'layers': [(256, 256)],
        'ent_wt': [0.1],
        'env_name': env_suite.ENV_KEYS,
        'repeat': list(range(5)),
    }
    hyper_sweep.run_sweep_serial(main, args, repeat=1)
