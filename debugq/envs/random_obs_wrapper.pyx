# distutils: language=c++
import numpy as np
import gym.spaces

from debugq.envs cimport env_wrapper
from rlutil.envs.tabular_cy cimport tabular_env

cdef class RandomObsWrapper(env_wrapper.TabularEnvWrapper):
    def __init__(self, tabular_env.TabularEnv env, int dim_obs):
        super(RandomObsWrapper, self).__init__(env)
        self._observations = np.random.random((env.num_states, dim_obs)).astype(np.float32)*2 - 1
        #self._observations = np.random.randn(env.num_states, dim_obs).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=np.min(self._observations),
                                                high=np.max(self._observations),
                                                shape=(dim_obs,),
                                                dtype=np.float32)

    cpdef observation(self, int state):
        return self._observations[state]


cdef class LocalObsWrapper(env_wrapper.TabularEnvWrapper):
    def __init__(self, tabular_env.TabularEnv env, int dim_obs):
        super(LocalObsWrapper, self).__init__(env)
        #self._observations = np.zeros((env.num_states, dim_obs), dtype=np.float32)
        trans_matrix = np.sum(env.transition_matrix(), axis=1) / env.num_actions
        self._observations = np.random.randn(env.num_states, dim_obs).astype(np.float32)
        for k in range(10):
            cur_obs_mat = self._observations[:,:] #np.random.randn(env.num_states, dim_obs).astype(np.float32)
            for state in range(env.num_states):
                new_obs = trans_matrix[state].dot(cur_obs_mat)
                self._observations[state] = new_obs
        self.observation_space = gym.spaces.Box(low=np.min(self._observations),
                                                high=np.max(self._observations),
                                                shape=(dim_obs,),
                                                dtype=np.float32)
        
    cpdef observation(self, int state):
        return self._observations[state]


cdef class OneHotObsWrapper(env_wrapper.TabularEnvWrapper):
    def __init__(self, tabular_env.TabularEnv env):
        super(OneHotObsWrapper, self).__init__(env)
        self.dim = int(np.sqrt(env.num_states))
        self.observation_space = gym.spaces.Box(low=0.,
                                                high=1.,
                                                shape=(2*self.dim,),
                                                dtype=np.float32)

    cpdef observation(self, int state):
        obs = np.zeros(self.dim*2, dtype=np.float32)
        obs[state % self.dim] = 1.0
        obs[self.dim + (state // self.dim)] = 1.0
        return obs
