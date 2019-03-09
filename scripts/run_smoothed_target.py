"""
This script runs the alpha-smoothed backup experiment.

Plot results using plot_smoothed_target.py
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


def main(exp_prefix='exp', algo='exact', layers=(32, 32), repeat=0,
        env_name='grid1', **alg_args):
    env = env_suite.get_env(env_name)

    if layers == 'tabular':
        network = q_networks.TabularNetwork(env)
    else:
        network = q_networks.FCNetwork(env, layers=layers)
    ptu.initialize_network(network)

    alg_args.update({
        'min_project_steps': 10,
        'max_project_steps': 300,
        'lr': 5e-3,
        'discount': 0.95,
        'n_steps': 1,
        'backup_mode': 'exact',
        'stop_modes': (stopping.AtolStop(), stopping.RtolStop()),
        'time_limit': env.time_limit,
        'env_name': env_name,
        'layers': str(layers),
    })
    fqi = exact_fqi.WeightedExactFQI(env, network, log_proj_qstar=True, **alg_args)
    with log_utils.setup_logger(algo=fqi, exp_prefix=exp_prefix, log_base_dir='/data') as log_dir:
        print('Logging to %s' % log_dir)
        try:
            for k in range(600):
                fqi.update(step=k)
        except:
            log_utils.save_exception()

if __name__ == "__main__":
    prefix = 'fast_exact_fqi_weighting'
    args = {
        'exp_prefix': [prefix + '_TEST'],
        'weighting_scheme': ['uniform', 'pi', 'buffer_infinite', 'robust_prioritized'],
        'layers': [(4,4), (16,16), (64,64), (256,256), 'tabular'],
        'target_mode': ['tq'],
        'weight_states_only': [False],
        'smooth_target_tau': [0.01, 0.05, 0.1, 0.5, 1.0],
        'ent_wt': [0.0, 0.1],
        'env_name': env_suite.ENV_KEYS,
        'repeat': list(range(7)),
    }
    hyper_sweep.run_sweep_serial(main, args, repeat=1)
