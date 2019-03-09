"""Simple test - make sure there are no silly runtime errors. """
import unittest
from parameterized import parameterized

from debugq.algos import sampling_fqi, stopping
from debugq.envs import random_obs_wrapper
from debugq.envs import time_limit_wrapper
from debugq.models import q_networks
from debugq import pytorch_util as ptu

from rlutil.envs.tabular_cy import tabular_env
from rlutil.logging import log_utils

class TestSamplingFQI(unittest.TestCase):
    def setUp(self):
        self.env_tab = tabular_env.CliffwalkEnv(10)
        self.env_obs = random_obs_wrapper.RandomObsWrapper(self.env_tab, 8)
        self.env = time_limit_wrapper.TimeLimitWrapper(self.env_obs, 50)

        self.network = q_networks.LinearNetwork(self.env)
        ptu.initialize_network(self.network)

        self.alg_args = {
            'min_project_steps': 10,
            'max_project_steps': 20,
            'lr': 5e-3,
            'discount': 0.95,
            'n_steps': 1,
            'num_samples': 32,
            'stop_modes': (stopping.AtolStop(), stopping.RtolStop()),
            'backup_mode': 'sampling',
            'ent_wt': 0.01,
        }

    @parameterized.expand([(mode,) for mode in sampling_fqi.WeightedSamplingFQI.WEIGHT_MODES])
    def testWeights(self, mode):
        fqi = sampling_fqi.WeightedSamplingFQI(self.env, self.network, weighting_scheme=mode, **self.alg_args)
        log_utils.reset_logger()
        fqi.update(1)

    @parameterized.expand([(mode,) for mode in sampling_fqi.PolicySamplingFQI.SAMPLING_MODES])
    def testSamplingModes(self, mode):
        fqi = sampling_fqi.PolicySamplingFQI(self.env, self.network, sampling_policy=mode, **self.alg_args)
        log_utils.reset_logger()
        fqi.update(1)


if __name__ == "__main__":
    unittest.main()