"""Simple test - make sure there are no silly runtime errors. """
import unittest
import numpy as np
from parameterized import parameterized

from debugq.envs import env_suite

class TestAllEnvs(unittest.TestCase):

    @parameterized.expand([(env, ) for env in env_suite.ENV_KEYS])
    def testEnv(self, env_name):
        env = env_suite.get_env(env_name)
        s0 = env.reset_state()
        s1 = env.reset_state()
        # initial states deterministic
        self.assertEqual(s0, s1)

        obs0 = env.reset()
        self.assertTrue(np.all(env.observation(s0) == obs0))
        # don't error on step
        env.step(0)


class TestRandomSeed(unittest.TestCase):
    """ Test if fixed random seeding produces deterministic environments."""

    def testSeedGrid(self):
        env1 = env_suite.get_env('grid16randomobs')
        env2 = env_suite.get_env('grid16onehot')
        # should share the same gridspec
        self.assertTrue(np.allclose(env1.transition_matrix(), env2.transition_matrix()))

    def testSeedObs(self):
        env1 = env_suite.get_env('grid16randomobs')
        env2 = env_suite.get_env('grid16randomobs')
        # should share the same observations
        self.assertTrue(np.allclose(env1.observation(0), env2.observation(0)))


if __name__ == "__main__":
    unittest.main()

