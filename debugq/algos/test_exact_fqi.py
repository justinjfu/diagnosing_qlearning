"""Simple test - make sure there are no silly runtime errors. """
import unittest
import copy
import numpy as np
from parameterized import parameterized

from debugq.algos import exact_fqi, stopping
from debugq.envs import random_obs_wrapper
from debugq.envs import time_limit_wrapper
from debugq.models import q_networks
from debugq import pytorch_util as ptu

from rlutil.envs.tabular_cy import tabular_env
from rlutil.logging import log_utils

class TestExactFQI(unittest.TestCase):
    def setUp(self):
        self.env_tab = tabular_env.CliffwalkEnv(5)
        self.env_obs = random_obs_wrapper.RandomObsWrapper(self.env_tab, 16)
        self.env = time_limit_wrapper.TimeLimitWrapper(self.env_obs, 10)

        self.network = q_networks.LinearNetwork(self.env)
        ptu.initialize_network(self.network)

        self.alg_args = {
            'min_project_steps': 10,
            'max_project_steps': 20,
            'lr': 5e-3,
            'discount': 0.95,
            'n_steps': 1,
            'num_samples': 32,
            'weighting_scheme': 'uniform',
            'stop_modes': (stopping.AtolStop(), stopping.RtolStop()),
            'backup_mode': 'exact',
            'ent_wt': 0.01,
        }

    def testProjQstar(self):
        fqi = exact_fqi.ExactFQI(self.env, self.network, log_proj_qstar=True, **self.alg_args)
        fqi.update(1)

    @parameterized.expand([(weight_mode,) for weight_mode in exact_fqi.WeightedExactFQI.WEIGHT_MODES])
    def testWeights(self, weight_mode):
        if weight_mode == 'pi*proj':
            return
        alg_args = copy.copy(self.alg_args)
        alg_args['weighting_scheme'] = weight_mode
        
        log_utils.reset_logger()
        fqi = exact_fqi.WeightedExactFQI(self.env, self.network, **alg_args)
        fqi.update(1)
        log_utils.reset_logger()
        fqi = exact_fqi.WeightedExactFQI(self.env, self.network, weight_states_only=True, **alg_args)
        fqi.update(1)
        

    def testUnknownWeightScheme(self):
        self.alg_args['weighting_scheme'] = 'unknown'
        with self.assertRaises(ValueError):
            fqi = exact_fqi.WeightedExactFQI(self.env, self.network, **self.alg_args)
            fqi.update(1)
    
    @parameterized.expand([(weight_mode,) for weight_mode in exact_fqi.WeightedExactFQI.WEIGHT_MODES])
    def testWeightStatesOnly(self, weight_mode):
        log_utils.reset_logger()
        if weight_mode in ['pi*proj', 'robust_adversarial', 'robust_prioritized']:
            return
        alg_args = copy.copy(self.alg_args)
        alg_args['weight_states_only'] = True
        alg_args['weighting_scheme'] = weight_mode
        fqi = exact_fqi.WeightedExactFQI(self.env, self.network, **alg_args)
        fqi.pre_project()
        states, _, _, _, weights = fqi.get_sample_states()
        self.assertEquals(weights.shape, (self.env.num_states, self.env.num_actions))
        chk_weights = np.abs(weights - weights[:,0][:,np.newaxis])
        self.assertEquals(np.sum(chk_weights), 0)
        self.assertAlmostEquals(np.sum(weights), 1.0)

    @parameterized.expand([(weight_mode,) for weight_mode in exact_fqi.WeightedExactFQI.WEIGHT_MODES])
    def testWeightStateAction(self, weight_mode):
        log_utils.reset_logger()
        if weight_mode in ['pi*proj', 'robust_adversarial', 'robust_prioritized']:
            return
        alg_args = copy.copy(self.alg_args)
        alg_args['weight_states_only'] = False
        alg_args['weighting_scheme'] = weight_mode
        fqi = exact_fqi.WeightedExactFQI(self.env, self.network, **alg_args)
        fqi.pre_project()
        states, _, _, _, weights = fqi.get_sample_states()
        self.assertEquals(weights.shape, (self.env.num_states, self.env.num_actions))
        self.assertAlmostEquals(np.sum(weights), 1.0)


if __name__ == "__main__":
    unittest.main()
