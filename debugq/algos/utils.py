import numpy as np

from rlutil.envs.tabular import q_iteration

def run_rollout(env, q_fn, ent_wt=1.0, render=False):
    s0 = env.reset_state()
    states = []
    actions = []
    next_states = []
    pol = q_iteration.get_policy(q_fn, ent_wt=ent_wt)

    rewards = []
    while True:
        states.append(env.get_state())
        probs = pol[env.get_state()]
        action = np.random.choice(np.arange(0, env.num_actions), p=probs)
        ts = env.step_state(action)
        rewards.append(ts['reward'])
        actions.append(action)
        next_states.append(env.get_state())
        if ts['done']:
            break
        if render:
            env.render()
    return states, actions, next_states, rewards


def eval_policy_qfn(env, q_fn, n_rollout=10, ent_wt=1.0, render=False):
    returns = []
    for i in range(n_rollout):
        _, _, _, rews = run_rollout(env, q_fn, ent_wt=ent_wt,
            render=render and (i==n_rollout-1))
        returns.append(np.sum(rews))
    return np.mean(returns)
