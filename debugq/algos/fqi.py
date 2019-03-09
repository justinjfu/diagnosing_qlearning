import six
import time
import torch
import debugq.pytorch_util as ptu
import numpy as np

from rlutil.logging import logger, hyperparameterized, log_utils
from rlutil.envs.tabular import q_iteration
from rlutil.envs.tabular_cy import q_iteration as q_iteration_cy

from debugq.algos import utils, stopping

BACKUP_EXACT = 'exact'
BACKUP_SAMPLING = 'sampling'
MULTIPLE_HEADS = 'heads'
FLAT = 'flat'

def weighted_q_diff(q1, q2, state_weights):
    return (q1 - q2) * np.expand_dims(state_weights, axis=1)

def compute_exact_projection(target_q, network, states, weights, N=10000, robust=True, **optimizer_args):
    network = network
    optimizer = torch.optim.Adam(network.parameters(), **optimizer_args)
    pt_target_q = ptu.tensor(target_q)
    for k in range(N):
        q_values = network(states)
        if robust:
            s_weights = torch.sum(torch.abs(q_values - pt_target_q), dim=1).detach()
            s_weights = s_weights / torch.sum(s_weights) * len(s_weights)
        else:
            s_weights = ptu.tensor(weights) 
        critic_loss = torch.mean(s_weights * torch.mean((q_values-pt_target_q)**2, dim=1))

        network.zero_grad()
        critic_loss.backward()
        optimizer.step()
        if k % 1000 == 0:
            logger.log('Itr %d exact projection loss: %f' % (k, ptu.to_numpy(critic_loss)))
    proj_q = ptu.to_numpy(network(states))
    network.reset_weights()

    logger.log('Exact projection abs diff: %f' % (np.mean(np.abs(weighted_q_diff(target_q, proj_q, weights)))))
    return proj_q 


@six.add_metaclass(hyperparameterized.Hyperparameterized)
class FQI(object):
    def __init__(self, env, network, min_project_steps=5, max_project_steps=50, lr=1e-3, 
                 discount=0.99, n_steps=1, ent_wt=1.0,
                 stop_modes=tuple(),
                 backup_mode=BACKUP_EXACT,
                 n_eval_trials=50,
                 log_proj_qstar=False,
                 target_mode='tq',
                 optimizer='adam',
                 smooth_target_tau=1.0,
                 **kwargs
                 ):
        self.env = env
        self.network = network
        self.discount = discount
        self.ent_wt = ent_wt
        self.n_eval_trials = n_eval_trials
        self.lr = lr
        self.max_q = 1e5

        self.target_mode = target_mode

        self.backup_mode = backup_mode
        if backup_mode == BACKUP_EXACT:
            self.q_format = MULTIPLE_HEADS
        else:
            self.q_format = FLAT

        self.min_project_steps = min_project_steps
        self.max_project_steps = max_project_steps
        self.stop_modes = stop_modes
        self.n_steps = n_steps
        self.lr = lr

        if optimizer == 'adam':
            self.qnet_optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        elif optimizer == 'gd':
            self.qnet_optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        else:
            raise ValueError("Unknown optimizer: %s" % optimizer)
        self.all_states = ptu.tensor(np.arange(env.num_states), dtype=torch.int64)

        with log_utils.timer('ground_truth_q'):
            self.ground_truth_q = q_iteration_cy.softq_iteration(self.env, num_itrs=max(self.env.num_states*2, 1000), 
                                                                discount=self.discount, ent_wt=self.ent_wt)
            self.ground_truth_q_torch = ptu.tensor(self.ground_truth_q)
        self.valid_weights = np.sum(self.ground_truth_q, axis=1)
        self.valid_weights[self.valid_weights != 0] = 1.0
        self.current_q = self.ground_truth_q
        returns = self.eval_policy(render=False, n_rollouts=self.n_eval_trials*5)
        self.expert_returns = returns
        self.current_q = np.zeros_like(self.ground_truth_q)
        returns = self.eval_policy(render=False, n_rollouts=self.n_eval_trials*5)
        self.random_returns = returns
        self.normalize_returns = lambda x: (x - self.random_returns) / (self.expert_returns-self.random_returns)

        self.current_q = ptu.to_numpy(self.network(self.all_states))
        # compute Proj(Q*)
        self.log_proj_qstar = log_proj_qstar
        self.ground_truth_q_proj = np.zeros_like(self.ground_truth_q)
        if log_proj_qstar:
            with log_utils.timer('proj_qstar'):
                self.ground_truth_q_proj = compute_exact_projection(self.ground_truth_q, network, self.all_states, weights=self.valid_weights,
                    lr=lr)
            diff = weighted_q_diff(self.ground_truth_q, self.ground_truth_q_proj, self.valid_weights)
            self.qstar_abs_diff = np.abs(diff)

        self.smooth_target_tau = smooth_target_tau
        self.smooth_previous_target = np.zeros_like(self.ground_truth_q)

    def pre_project(self):
        pass

    def post_project(self):
        pass
    
    def get_sample_states(self, itr=0):
        raise NotImplementedError()

    def evaluate_qvalues(self, sample_s, sample_a, mode=None, network=None):
        if mode is None:
            mode = self.q_format
        if network is None:
            network = self.network
        sample_s, sample_a = ptu.all_tensor([sample_s, sample_a], dtype=torch.int64)
        if mode == MULTIPLE_HEADS:
            q_values = network(sample_s)  # S by A
        else:
            q_values_s = network(sample_s)  # S by A
            q_values = q_values_s.gather(1, sample_a.reshape(-1,1)).squeeze()
        return q_values

    def evaluate_target(self, sample_s, sample_a, sample_ns, sample_r, mode=None):
        if mode is None:
            mode = self.q_format
        if mode == MULTIPLE_HEADS:
            sample_s = ptu.tensor(sample_s, dtype=torch.int64)
            target_q = torch.index_select(self.all_target_q, 0, sample_s)  # S by A
        else:
            v_next = q_iteration.logsumexp(self.current_q[sample_ns], alpha=self.ent_wt)
            target_q =  ptu.tensor(sample_r + self.discount * v_next)
        return target_q
    
    def eval_policy(self, render=False, n_rollouts=None):
        if n_rollouts is None:
            n_rollouts = self.n_eval_trials
        return utils.eval_policy_qfn(self.env, self.current_q, ent_wt=self.ent_wt, n_rollout=n_rollouts, render=render)

    def project(self, network=None, optimizer=None, sampler=None):
        if network is None:
            network = self.network
        if optimizer is None:
            optimizer = self.qnet_optimizer
        if sampler is None:
            sampler = self.get_sample_states

        k=0
        stopped_mode = None
        [stop_mode.reset() for stop_mode in self.stop_modes]
        with log_utils.timer('projection') as timer:
            for k in range(self.max_project_steps):
                with timer.subtimer('compute_samples_weights'):
                    sample_s, sample_a, sample_ns, sample_r, weights = sampler(itr=k)
                sample_s, sample_a = ptu.all_tensor([sample_s, sample_a], dtype=torch.int64)
                weights, = ptu.all_tensor([weights])

                with timer.subtimer('eval_target'):
                    target_q = self.evaluate_target(sample_s, sample_a, sample_ns, sample_r).detach()
                with timer.subtimer('eval_q'):
                    q_values = self.evaluate_qvalues(sample_s, sample_a, network=network)

                if self.q_format == MULTIPLE_HEADS:
                    if len(weights.shape) == 2:
                        critic_loss = torch.mean(weights * (q_values-target_q)**2)
                    else:
                        critic_loss = torch.mean(weights * torch.mean((q_values-target_q)**2, dim=1))
                else:
                    critic_loss = torch.mean(weights * (q_values-target_q)**2)

                with timer.subtimer('backprop'):
                    network.zero_grad()
                    critic_loss.backward()
                    optimizer.step()

                stop_args = dict(critic_loss=ptu.to_numpy(critic_loss), 
                    q_network=network,
                    all_target_q=self.all_target_q,
                    fqi=self,
                    discount=self.discount,
                    ent_wt=self.ent_wt)
                [stop_mode.update(**stop_args) for stop_mode in self.stop_modes]
                if (k >= self.min_project_steps):
                    stopped = False
                    for stop_mode in self.stop_modes:
                        if stop_mode.check():
                            logger.log('Early stopping via %s.' % stop_mode)
                            stopped = True
                            stopped_mode = stop_mode
                            break
                    if stopped:
                        break
        return stopped_mode, critic_loss, k

    def update(self, step=-1):
        start_time = time.time()
        # backup 
        with log_utils.timer('compute_backup'):
            self.all_target_q_np = q_iteration_cy.softq_iteration(self.env, num_itrs=self.n_steps,
                warmstart_q=self.current_q, discount=self.discount, ent_wt=self.ent_wt)
            # smooth
            if self.smooth_target_tau < 1.0:
                self.all_target_q_np = self.smooth_target_tau * self.all_target_q_np + (1-self.smooth_target_tau) * self.current_q
            self.all_target_q = ptu.tensor(self.all_target_q_np)

        # project
        with log_utils.timer('pre_project'):
            self.pre_project()
        
        stopped_mode, critic_loss , k= self.project()

        if isinstance(stopped_mode, stopping.ValidationLoss):
            self.current_q = ptu.to_numpy(stopped_mode.best_validation_qs)
            logger.record_tabular('validation_stop_step', stopped_mode.validation_k)
        else:
            self.current_q = ptu.to_numpy(self.network(self.all_states))
        self.current_q = np.minimum(self.current_q, self.max_q)  # clip when diverging
        self.post_project()
        with log_utils.timer('eval_policy'):
            returns = self.eval_policy()

        logger.record_tabular('project_loss', ptu.to_numpy(critic_loss))
        logger.record_tabular('fit_steps', k)
        if step >=0:
            logger.record_tabular('step', step)

        # Logging
        logger.record_tabular('fit_q_value_mean', np.mean(self.current_q))
        logger.record_tabular('target_q_value_mean', np.mean(self.all_target_q_np))
        logger.record_tabular('returns_expert', self.expert_returns)
        logger.record_tabular('returns_random', self.random_returns)
        logger.record_tabular('returns', returns)
        log_utils.record_tabular_moving('returns', returns, n=50)
        logger.record_tabular('returns_normalized', self.normalize_returns(returns))
        log_utils.record_tabular_moving('returns_normalized', self.normalize_returns(returns), n=50)

        # measure contraction errors
        diff_tq_qstar = weighted_q_diff(self.all_target_q_np, self.ground_truth_q, self.valid_weights)
        abs_diff_tq_qstar = np.abs(diff_tq_qstar)
        log_utils.record_tabular_stats('tq_q*_diff', diff_tq_qstar)
        log_utils.record_tabular_stats('tq_q*_diff_abs', abs_diff_tq_qstar)

        if self.log_proj_qstar:
            diff = weighted_q_diff(self.current_q, self.ground_truth_q_proj, self.valid_weights)
            abs_diff = np.abs(diff)
            log_utils.record_tabular_stats('q*_proj_diff', diff)
            log_utils.record_tabular_stats('q*_proj_diff_abs', abs_diff)
            log_utils.record_tabular_stats('ground_truth_error', self.qstar_abs_diff)

        logger.record_tabular('iteration_time', time.time()-start_time)

        logger.dump_tabular()
