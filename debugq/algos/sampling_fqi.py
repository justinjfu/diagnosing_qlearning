import torch
import six
import numpy as np
import random

from rlutil.logging import logger, hyperparameterized
from rlutil.envs.tabular_cy import q_iteration
from rlutil.envs.tabular import q_iteration as q_iteration_py

from debugq.algos import fqi, utils
import debugq.pytorch_util as ptu

def stack_observations(env):
    obs = []
    for s in range(env.num_states):
        obs.append(env.observation(s))
    return np.stack(obs)

class UniformSamplingFQI(fqi.FQI):
    def __init__(self, env, network, num_samples=32, **kwargs):
        super(UniformSamplingFQI, self).__init__(env, network, **kwargs)
        self.all_observations= stack_observations(env)
        self.num_samples = num_samples
    
    def get_sample_states(self, itr=0):
        batch_idxs = np.random.randint(0, self.env.num_states, size=(self.num_samples))
        raise NotImplementedError('sample actions, nxt states')
        return batch_idxs, np.ones_like(batch_idxs)

class PolicySamplingFQI(fqi.FQI):
    SAMPLING_MODES = ['pi', 'pi*', 'random', 'adversarial']
    def __init__(self, env, network, num_samples=32, resample_per_step=False, time_limit=50,
        sampling_policy='pi', **kwargs):
        super(PolicySamplingFQI, self).__init__(env, network, **kwargs)
        self.num_samples = num_samples
        self.resample = resample_per_step
        self.sampling_policy = sampling_policy
        self.time_limit = time_limit
        self._total_samples = 0

    def pre_project(self):
        if self.sampling_policy == 'adversarial':
            q_vals = ptu.to_numpy(self.evaluate_qvalues(np.arange(0, self.env.num_states), None, mode=fqi.MULTIPLE_HEADS))
            errors = np.abs(q_vals - self.all_target_q_np) ** 0.5
            # pick adversarial distribution - reward is bellman error
            adversarial_qs = q_iteration.softq_iteration_custom_reward(self.env, reward=errors, num_itrs=self.time_limit, discount=self.discount, ent_wt=self.ent_wt, atol=1e-5)
            self.adversarial_qs = adversarial_qs
        self.batch_s, self.batch_a, self.batch_ns, self.batch_r = self.collect_samples()
        self._total_samples += len(self.batch_s)
        logger.record_tabular('total_samples', self._total_samples)

    def collect_samples(self, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        # run rollouts
        batch_s = []
        batch_a = []
        batch_ns = []
        batch_r = []
        if self.sampling_policy == 'pi':
            self.sampling_q = self.current_q
        elif self.sampling_policy == 'pi*':
            self.sampling_q = self.ground_truth_q
        elif self.sampling_policy == 'random':
            self.sampling_q = np.zeros_like(self.ground_truth_q)
        elif self.sampling_policy == 'adversarial':
            self.sampling_q = self.adversarial_qs
        else:
            raise ValueError('Unknown sampling policy: %s' % self.sampling_policy)
        while len(batch_s) < num_samples:
            states, actions, next_states, rewards = utils.run_rollout(self.env, self.sampling_q, ent_wt=self.ent_wt)
            batch_s.extend(states)
            batch_a.extend(actions)
            batch_ns.extend(next_states)
            batch_r.extend(rewards)
        
        ret = []
        batch_idx = np.arange(len(batch_s))
        np.random.shuffle(batch_idx)
        batch_idx = batch_idx[:num_samples]

        for batch in [batch_s, batch_a, batch_ns, batch_r]:
            ret.append(np.array(batch)[batch_idx])
        return tuple(ret)

    def get_sample_states(self, itr=0):
        if self.resample:
            self.collect_samples()
        samples = (self.batch_s, self.batch_a, self.batch_ns, self.batch_r)
        weights = self.compute_weights(samples, itr=itr)
        return samples + (weights, )
    
    def compute_weights(self, samples, itr=0):
        return np.ones(len(samples[0]))


class WeightedSamplingFQI(PolicySamplingFQI):
    WEIGHT_MODES = ['uniform', 'pi*', 'pi', 'robust_adversarial', 'robust_prioritized','robust_adversarial_fast', 'none']
    def __init__(self, env, network, weighting_scheme='none',
        **kwargs):
        super(WeightedSamplingFQI, self).__init__(env, network, **kwargs)
        self.wscheme = weighting_scheme
        self.vfn = q_iteration_py.logsumexp(self.ground_truth_q, alpha=self.ent_wt)
        self.optimal_visit_sa = q_iteration_py.compute_visitation(self.env, self.ground_truth_q, ent_wt=self.ent_wt,
            discount=self.discount, env_time_limit=self.time_limit)
        self.warmstart_adversarial_q = np.zeros_like(self.ground_truth_q)
    
    def pre_project(self):
        super(WeightedSamplingFQI, self).pre_project()
        self.sample_visit_sa = q_iteration_py.compute_visitation(self.env, self.sampling_q, ent_wt=self.ent_wt,
            discount=self.discount, env_time_limit=self.time_limit)
        self.sa_weights = compute_sa_weights(self, self.wscheme, self.sample_visit_sa)
        self.validation_sa_weights = self.sa_weights * self.sample_visit_sa

    def compute_weights(self, samples, itr=0):
        return compute_weights(self, samples, itr=itr)
    
    def post_project(self):
        # Log validation loss
        expected_loss = np.sum(self.validation_sa_weights * (self.current_q - self.all_target_q_np)**2)
        logger.record_tabular('validation_loss_reweighted', expected_loss)
        expected_loss = np.sum(self.sample_visit_sa * (self.current_q - self.all_target_q_np)**2)
        logger.record_tabular('validation_loss_sampling', expected_loss)


def compute_sa_weights(fqi, wscheme, sample_visit_sa):
    if wscheme == 'uniform':
        sa_weights = 1.0 / (sample_visit_sa + 1e-6)
    elif wscheme == 'pi*':
        sa_weights = fqi.optimal_visit_sa / (sample_visit_sa + 1e-6)
    elif wscheme == 'pi':
        current_visit_sa = q_iteration_py.compute_visitation(fqi.env, fqi.current_q, ent_wt=fqi.ent_wt,
            discount=fqi.discount, env_time_limit=fqi.time_limit)
        sa_weights = current_visit_sa / (sample_visit_sa + 1e-6)
    elif wscheme in ['robust_prioritized', 'robust_adversarial','robust_adversarial_fast', 'none']:
        sa_weights = np.ones_like(fqi.current_q)
    else:
        raise ValueError('Unkown weighting scheme: %s' % wscheme)
    return sa_weights


def compute_weights(self, samples, itr=0, clip_min=1e-6, clip_max=100.0):
    if self.wscheme == 'robust_prioritized':
        q_vals = self.evaluate_qvalues(samples[0], samples[1]).detach()
        target_qs = self.evaluate_target(samples[0], samples[1], samples[2], samples[3]).detach()
        error = torch.abs(q_vals - target_qs)
        weights = ptu.to_numpy(error)
    elif self.wscheme == 'robust_adversarial':
        # solve for max_pi [bellman error]
        # compute bellman errors
        q_vals = ptu.to_numpy(self.evaluate_qvalues(np.arange(0, self.env.num_states), None, mode=fqi.MULTIPLE_HEADS))
        errors = np.abs(q_vals - self.all_target_q_np)
        # pick adversarial distribution - reward is bellman error
        adversarial_qs = q_iteration.softq_iteration_custom_reward(self.env, reward=errors, num_itrs=self.time_limit, discount=self.discount, ent_wt=self.ent_wt,
            warmstart_q=self.warmstart_adversarial_q, atol=1e-5)
        self.warmstart_adversarial_q = adversarial_qs
        visit_sa = q_iteration_py.compute_visitation(self.env, adversarial_qs, ent_wt=self.ent_wt, discount=self.discount, env_time_limit=self.time_limit)
        weights = visit_sa[samples[0], samples[1]]
    elif self.wscheme == 'robust_adversarial_fast':
        if itr % 10 == 0:
            # solve for max_pi [bellman error]
            # compute bellman errors
            q_vals = ptu.to_numpy(self.evaluate_qvalues(np.arange(0, self.env.num_states), None, mode=fqi.MULTIPLE_HEADS))
            errors = np.abs(q_vals - self.all_target_q_np)
            # pick adversarial distribution - reward is bellman error
            adversarial_qs = q_iteration.softq_iteration_custom_reward(self.env, reward=errors, num_itrs=self.time_limit, discount=self.discount, ent_wt=self.ent_wt,
                warmstart_q=self.warmstart_adversarial_q, atol=1e-5)
            self.warmstart_adversarial_q = adversarial_qs
            self.adv_visit_sa = q_iteration_py.compute_visitation(self.env, adversarial_qs, ent_wt=self.ent_wt, discount=self.discount, env_time_limit=self.time_limit)
        weights = self.adv_visit_sa[samples[0], samples[1]]
    else:
        weights = self.sa_weights[samples[0], samples[1]]
    weights = (weights / np.sum(weights))  # normalize
    weights = np.minimum(weights, clip_max)
    weights = np.maximum(weights, clip_min)
    weights = (weights / np.sum(weights))  # normalize
    return weights