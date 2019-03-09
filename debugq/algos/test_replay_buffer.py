"""Simple test - make sure there are no silly runtime errors. """
import unittest
import numpy as np
from parameterized import parameterized

from debugq.algos import replay_buffer_fqi, sampling_fqi, stopping
from debugq.envs import random_obs_wrapper
from debugq.envs import time_limit_wrapper
from debugq.models import q_networks
from debugq import pytorch_util as ptu

from rlutil.envs.tabular_cy import tabular_env
from rlutil import math_utils
from rlutil.logging import log_utils

class TestTabularReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.env = tabular_env.CliffwalkEnv(10)

    def testProb(self):
        buffer = replay_buffer_fqi.TabularReplayBuffer(self.env)
        buffer.add(0, 1, 0, 1.0)
        probs = buffer.probs()
        self.assertAlmostEqual(probs[1,1,0] , 0.0)
        self.assertAlmostEqual(probs[0,1,0] , 1.0)

        buffer.add(1, 1, 0, 0.0)
        probs = buffer.probs()
        self.assertAlmostEqual(probs[1,1,0] , 0.5)
        self.assertAlmostEqual(probs[0,1,0] , 0.5)

    def testSampleSingle(self):
        buffer = replay_buffer_fqi.TabularReplayBuffer(self.env)
        buffer.add(0, 1, 0, 1.0)
        
        samples = buffer.sample(10)
        self.assertTrue(np.all(samples[0] == 0)) # s
        self.assertTrue(np.all(samples[1] == 1)) # a
        self.assertTrue(np.all(samples[2] == 0)) # ns 
        self.assertTrue(np.all(samples[3] == 1.0)) # r 

    def testSample1(self):
        buffer = replay_buffer_fqi.TabularReplayBuffer(self.env)
        buffer.add(0, 1, 0, 1.0)
        buffer.add(1, 0, 1, 0.0)
        
        with math_utils.np_seed(0):
            samples = buffer.sample(100)
        self.assertEqual(np.sum(samples[0]), 49 ) # s
        self.assertEqual(np.sum(samples[1]), 51 ) # a
        self.assertEqual(np.sum(samples[2]), 49 ) # ns
        self.assertEqual(np.sum(samples[3]), 51 ) # r

    def testSample2(self):
        buffer = replay_buffer_fqi.TabularReplayBuffer(self.env)
        buffer.add(0, 1, 0, 1.0)
        buffer.add(0, 1, 0, 1.0)
        buffer.add(1, 0, 1, 0.0)
        
        with math_utils.np_seed(0):
            samples = buffer.sample(100)
        self.assertEqual(np.sum(samples[0]), 29 ) # s
        self.assertEqual(np.sum(samples[1]), 71 ) # a
        self.assertEqual(np.sum(samples[2]), 29 ) # ns
        self.assertEqual(np.sum(samples[3]), 71 ) # r


class TestReplayBufferFQI(unittest.TestCase):
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
        fqi = replay_buffer_fqi.TabularBufferDQN(self.env, self.network, weighting_scheme=mode, **self.alg_args)
        log_utils.reset_logger()
        fqi.update(1)

    @parameterized.expand([(mode,) for mode in sampling_fqi.PolicySamplingFQI.SAMPLING_MODES])
    def testSamplingModes(self, mode):
        fqi = replay_buffer_fqi.TabularBufferDQN(self.env, self.network, sampling_policy=mode, **self.alg_args)
        log_utils.reset_logger()
        fqi.update(1)

if __name__ == "__main__":
    unittest.main()