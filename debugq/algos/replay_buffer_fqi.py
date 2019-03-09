import numpy as np
import torch

from debugq.algos import sampling_fqi
import debugq.pytorch_util as ptu
from rlutil.envs.tabular import q_iteration
import random
from rlutil.logging import logger

PROB_EPS = 1e-8


class ReplayBufferFQI(sampling_fqi.PolicySamplingFQI):
    def __init__(self, env, network, replay_buffer=None, batch_size=32, **kwargs):
        super(ReplayBufferFQI, self).__init__(env, network, **kwargs)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def pre_project(self):
        super(ReplayBufferFQI, self).pre_project()
        # add samples to replay buffer
        self.replay_buffer.add_all(self.batch_s, self.batch_a,
                                   self.batch_ns, self.batch_r)
        logger.record_tabular('replay_buffer_len', len(self.replay_buffer))

    def get_sample_states(self, itr=0):
        samples = self.replay_buffer.sample(self.batch_size)
        weights = self.compute_weights(samples, itr=itr)
        return samples + (weights, )

    def compute_weights(self, samples, itr=0):
        return np.ones(self.batch_size)


class WeightedBufferFQI(ReplayBufferFQI):
    def __init__(self, env, network, weighting_scheme='none', 
                 **kwargs):
        super(WeightedBufferFQI, self).__init__(env, network, **kwargs)
        self.wscheme = weighting_scheme
        self.vfn = q_iteration.logsumexp(
            self.ground_truth_q, alpha=self.ent_wt)
        self.optimal_visit_sa = q_iteration.compute_visitation(self.env, self.ground_truth_q, ent_wt=self.ent_wt,
                                                               discount=self.discount, env_time_limit=self.time_limit)

        self.warmstart_adversarial_q = np.zeros_like(self.ground_truth_q)

    def pre_project(self):
        super(WeightedBufferFQI, self).pre_project()
        self.sa_weights = sampling_fqi.compute_sa_weights(self, self.wscheme, self.replay_buffer.probs_sa())
        self.validation_sa_weights = self.sa_weights * self.replay_buffer.probs_sa()

        # validation loss
        self.buffer_validation_sa_weights = self.replay_buffer.probs_sa()
        sample_visit_sa = q_iteration.compute_visitation(self.env, self.sampling_q, ent_wt=self.ent_wt,
            discount=self.discount, env_time_limit=self.time_limit)
        self.onpolicy_validation_sa_weights = sample_visit_sa

    def compute_weights(self, samples, itr=0):
        return sampling_fqi.compute_weights(self, samples, itr=itr)

    def post_project(self):
        # Log validation loss
        expected_loss = np.sum(self.onpolicy_validation_sa_weights * (self.current_q - self.all_target_q_np)**2)
        logger.record_tabular('validation_loss_sampling', expected_loss)
        expected_loss = np.sum(self.validation_sa_weights * (self.current_q - self.all_target_q_np)**2)
        logger.record_tabular('validation_loss_reweighted', expected_loss)
        expected_loss = np.sum(self.buffer_validation_sa_weights * (self.current_q - self.all_target_q_np)**2)
        logger.record_tabular('validation_loss_buffer', expected_loss)



class DQN(WeightedBufferFQI):
    def __init__(self, env, network, replay_buffer_size=1000, **kwargs):
        replay_buffer = SimpleReplayBuffer(replay_buffer_size)
        super(DQN, self).__init__(env, network,
                                  replay_buffer=replay_buffer, **kwargs)


class TabularBufferDQN(WeightedBufferFQI):
    def __init__(self, env, network, replay_buffer_size=1000, **kwargs):
        replay_buffer = TabularReplayBuffer(env)
        super(TabularBufferDQN, self).__init__(
            env, network, replay_buffer=replay_buffer, **kwargs)



class ReplayBuffer(object):
    def __init__(self):
        pass

    def add_all(self, s, a, ns, r):
        for i in range(len(s)):
            self.add(s[i], a[i], ns[i], r[i])

    def add(self, s, a, ns, r):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()
    
    def probs(self):
        raise NotImplementedError()

    def probs_sa(self):
        return np.sum(self.probs(), axis=2)


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self._s = np.zeros(capacity, dtype=np.int32)
        self._a = np.zeros(capacity, dtype=np.int32)
        self._ns = np.zeros(capacity, dtype=np.int32)
        self._r = np.zeros(capacity, dtype=np.float32)
        self._wt = np.zeros(capacity, dtype=np.float32)
        self._cur_idx = 0
        self._len = 0

    def all(self):
        return self._s[:len(self)], self._a[:len(self)], \
            self._ns[:len(self)], self._r[:len(self)]

    def add(self, s, a, ns, r):
        self._s[self._cur_idx] = s
        self._a[self._cur_idx] = a
        self._ns[self._cur_idx] = ns
        self._r[self._cur_idx] = r
        self._cur_idx += 1
        if self._cur_idx >= self.capacity:
            self._cur_idx = 0
        self._len = min(self.capacity, self._len + 1)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), size=batch_size)
        return self._s[idxs], self._a[idxs], self._ns[idxs], self._r[idxs]

    def __len__(self):
        return self._len


class TabularReplayBuffer(ReplayBuffer):
    """
    A replay buffer which stores transitions in tabular form. The capacity is essentially machine precision, and
    allows for easy computation of probabilities.
    """

    def __init__(self, env):
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self._transitions = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=np.int32)
        self._reward = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=np.float32)
        self._len = 0

        self.normalized = False
        self._normalized_transitions = np.zeros_like(self._transitions)
        self._nonzero_transitions = None
        self._nonzero_probs = None

    def add(self, s, a, ns, r):
        self._transitions[s, a, ns] += 1
        self._reward[s, a, ns] = r
        self._len += 1
        self.normalized = False

    def probs(self):
        if not self.normalized:
            self._normalized_transitions = self._transitions / float(self._len)
            self._nonzero_transitions = np.where(self._normalized_transitions)
            self._nonzero_probs = self._normalized_transitions[self._nonzero_transitions]
            self.normalized = True
        return self._normalized_transitions

    def sample(self, batch_size):
        probs = self.probs()
        items = np.random.choice(
            len(self._nonzero_transitions[0]), size=batch_size, p=self._nonzero_probs)
        idxs = [x[items] for x in self._nonzero_transitions]
        s_ = idxs[0]
        a_ = idxs[1]
        ns_ = idxs[2]
        r_ = np.array([self._reward[s, a, ns]
                       for s, a, ns in zip(s_, a_, ns_)])
        return s_, a_, ns_, r_

    def __len__(self):
        return self._len
