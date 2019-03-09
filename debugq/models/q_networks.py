import numpy as np
import torch

from debugq import pytorch_util as ptu

class TabularNetwork(torch.nn.Module):
  def __init__(self, env):
    super(TabularNetwork, self).__init__()
    self.num_states = env.num_states
    self.network = torch.nn.Sequential(
        torch.nn.Linear(self.num_states, env.num_actions)
    )

  def forward(self, states):
    onehot = ptu.one_hot(states, self.num_states)
    return self.network(onehot)

  def reset_weights(self):
    for layer in self.network:
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  

def stack_observations(env):
    obs = []
    for s in range(env.num_states):
        obs.append(env.observation(s))
    return np.stack(obs)


class LinearNetwork(torch.nn.Module):
  def __init__(self, env):
    super(LinearNetwork, self).__init__()
    self.all_observations = ptu.tensor(stack_observations(env))
    self.dim_input = env.observation_space.shape[-1]
    self.network = torch.nn.Sequential(
        torch.nn.Linear(self.dim_input, env.num_actions)
    )

  def forward(self, states):
    observations = torch.index_select(self.all_observations, 0, states) 
    return self.network(observations)

  def reset_weights(self):
    for layer in self.network:
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()


class FCNetwork(torch.nn.Module):
  def __init__(self, env, layers=[20,20]):
    super(FCNetwork, self).__init__()
    self.all_observations = ptu.tensor(stack_observations(env))
    dim_input = env.observation_space.shape
    dim_output = env.num_actions
    net_layers = []

    dim = dim_input[-1]
    for i, layer_size in enumerate(layers):
      net_layers.append(torch.nn.Linear(dim, layer_size))
      net_layers.append(torch.nn.ReLU())
      dim = layer_size
    net_layers.append(torch.nn.Linear(dim, dim_output))
    self.layers = net_layers
    self.network = torch.nn.Sequential(*net_layers)

  def forward(self, states):
    observations = torch.index_select(self.all_observations, 0, states) 
    return self.network(observations)

  def reset_weights(self):
    for layer in self.network:
      if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
  
