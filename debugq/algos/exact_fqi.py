import six
import numpy as np

from rlutil.logging import logger, hyperparameterized 
from rlutil.envs.tabular_cy import q_iteration
from rlutil.envs.tabular import q_iteration as q_iteration_py

from debugq.algos import fqi
import debugq.pytorch_util as ptu


class ExactFQI(fqi.FQI):
    def __init__(self, env, network, **kwargs):
        super(ExactFQI, self).__init__(env, network, **kwargs)
        assert self.backup_mode == fqi.BACKUP_EXACT
    
    def get_sample_states(self, itr=0):
        return np.arange(0, self.env.num_states), None, None, None, np.ones((self.env.num_states, ))


class WeightedExactFQI(fqi.FQI):
    WEIGHT_MODES = ['uniform', 'pi*', 'pi*proj', 'pi', 'online', 'robust_prioritized', 'robust_adversarial', 'random', 'buffer_infinite', 'buffer10']
    def __init__(self, env, network, weighting_scheme='uniform', weight_states_only=False, time_limit=100, **kwargs):
        super(WeightedExactFQI, self).__init__(env, network, **kwargs)
        assert self.backup_mode == fqi.BACKUP_EXACT
        self.wscheme = weighting_scheme
        self.time_limit = time_limit
        self.weight_states_only = weight_states_only

        self.visit_sa = q_iteration_py.compute_visitation(self.env, self.ground_truth_q, ent_wt=self.ent_wt,
            discount=self.discount, env_time_limit=self.time_limit)
        self.warmstart_adversarial_q = self.ground_truth_q[:,:]

        self.opt_proj_visit_sa = q_iteration_py.compute_visitation(self.env, self.ground_truth_q_proj, ent_wt=self.ent_wt,
            discount=self.discount, env_time_limit=self.time_limit)

        self.buffer_sa = np.zeros_like(self.ground_truth_q)
        self.buffer_n = 0

        self.buffer10 = []

        self.prev_q_target = np.zeros_like(self.ground_truth_q)
        self.prev_q_value = np.zeros_like(self.ground_truth_q)
        self.prev_loss = 0
        self.prev_weights = np.zeros_like(self.ground_truth_q)
    
    def pre_project(self):
        if self.wscheme == 'pi':
            self.pi_visit_sa = q_iteration_py.compute_visitation(self.env, self.current_q, ent_wt=self.ent_wt,
                discount=self.discount, env_time_limit=self.time_limit)
        elif self.wscheme == 'random':
            self.pi_visit_sa = q_iteration_py.compute_visitation(self.env, np.zeros_like(self.current_q), ent_wt=self.ent_wt,
                discount=self.discount, env_time_limit=self.time_limit)
        elif self.wscheme == 'buffer_infinite':
            pi_visit_sa = q_iteration_py.compute_visitation(self.env, self.current_q, ent_wt=self.ent_wt,
                discount=self.discount, env_time_limit=self.time_limit)
            self.buffer_n += 1
            self.buffer_sa *= (self.buffer_n-1 ) / self.buffer_n
            self.buffer_sa += pi_visit_sa / self.buffer_n
        elif self.wscheme == 'buffer10':
            pi_visit_sa = q_iteration_py.compute_visitation(self.env, self.current_q, ent_wt=self.ent_wt,
                discount=self.discount, env_time_limit=self.time_limit)
            self.buffer10.append(pi_visit_sa)
            self.buffer10 = self.buffer10[-10:]
            self.buffer_sa = np.mean(self.buffer10, axis=0)

    def get_sample_states(self, itr=0):
        if itr % 5 == 0:  # compute weights
            weights = None
            if self.wscheme == 'uniform':
                weights = np.ones((self.env.num_states, self.env.num_actions))
            elif self.wscheme == 'buffer_infinite':
                weights = self.buffer_sa
            elif self.wscheme == 'buffer10':
                weights = self.buffer_sa
            elif self.wscheme == 'pi*':
                weights = self.visit_sa
            elif self.wscheme == 'pi*proj':
                assert self.log_proj_qstar
                weights = self.opt_proj_visit_sa
            elif self.wscheme == 'random':
                weights = self.pi_visit_sa
            elif self.wscheme == 'pi':
                weights = self.pi_visit_sa
            elif self.wscheme == 'online':
                q_vals = ptu.to_numpy(self.evaluate_qvalues(np.arange(0, self.env.num_states), None))
                visit_sa = q_iteration_py.compute_visitation(self.env, q_vals, ent_wt=self.ent_wt,
                    discount=self.discount, env_time_limit=self.time_limit)
                weights = visit_sa
            elif self.wscheme == 'robust_prioritized':
                q_vals = ptu.to_numpy(self.evaluate_qvalues(np.arange(0, self.env.num_states), None))
                errors = np.abs(q_vals - self.all_target_q_np)
                weights = errors
            elif self.wscheme == 'robust_adversarial':
                # solve for max_pi [bellman error]
                # compute bellman errors
                q_vals = ptu.to_numpy(self.evaluate_qvalues(np.arange(0, self.env.num_states), None))
                errors = np.abs(q_vals - self.all_target_q_np)
                # pick adversarial distribution - reward is bellman error
                adversarial_qs = q_iteration.softq_iteration_custom_reward(self.env, reward=errors, num_itrs=self.time_limit, discount=self.discount, ent_wt=self.ent_wt,
                    warmstart_q=self.warmstart_adversarial_q, atol=1e-5)
                self.warmstart_adversarial_q = adversarial_qs
                visit_sa = q_iteration_py.compute_visitation(self.env, adversarial_qs, ent_wt=self.ent_wt, discount=self.discount, env_time_limit=self.time_limit)
                weights = visit_sa
            else:
                raise ValueError("Unknown weighting scheme: %s" %self.wscheme)

            if self.weight_states_only:
                weights = np.sum(weights, axis=1)
                weights = np.repeat(weights[:, np.newaxis], self.env.num_actions, axis=-1)
            self.weights = (weights / np.sum(weights)) # normalize
        if itr == 0:
            entropy = -np.sum(self.weights * np.log(self.weights + 1e-6))
            logger.record_tabular('weight_entropy', entropy)
            unif = np.ones_like(self.weights) / float(self.weights.size)
            max_entropy = -np.sum(unif * np.log(unif))
            logger.record_tabular('weight_entropy_normalized', entropy / max_entropy)
        return np.arange(0, self.env.num_states), None, None, None, self.weights

    def post_project(self):
        #raise NotImplementedError("TODO: measure distributional shift - loss under next and ")
        if not self.weight_states_only:
            prev_loss = np.sum(self.prev_weights * (self.prev_q_target - self.prev_q_value)**2)
            shift_loss = np.sum(self.weights * (self.prev_q_target - self.prev_q_value)**2)
            logger.record_tabular('distributional_shift_old_loss', prev_loss)
            logger.record_tabular('distributional_shift_new_loss', shift_loss)
            logger.record_tabular('distributional_shift_diff_loss', shift_loss - prev_loss)
            logger.record_tabular('distributional_shift_abs_diff_loss', np.abs(shift_loss - prev_loss))
            logger.record_tabular('distributional_shift_tv', 0.5*np.sum(np.abs(self.weights - self.prev_weights)))
            logger.record_tabular('fit_qvalue_weighted_mean', np.sum(self.weights * self.current_q))

            # update
            self.prev_weights = self.weights
            self.prev_q_target = self.all_target_q_np
            self.prev_q_value = self.current_q
