import numpy as np

import tensorflow as tf
#import tensorflow.contrib.eager.python.tfe as tfe
from tensorflow.contrib.eager.python import tfe


class TabularNetwork(tf.keras.Model):
  def __init__(self, env):
    super(TabularNetwork, self).__init__()
    self.num_states = env.num_states
    self.network = tf.keras.Sequential(
        [tf.keras.layers.Dense(name='dense1', units=env.num_actions)]
    )

  def call(self, inputs):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).
      actions: a batch of action.

    Returns:
      Values of observations.
    """
    onehot = tf.one_hot(inputs, self.num_states)
    return self.network(onehot)
  
  def initialize_variables(self):
    self.call(np.array([0,0]))


class LinearNetwork(tf.keras.Model):
  def __init__(self, env):
    super(LinearNetwork, self).__init__()
    self.dim_input = env.observation_space.shape
    self.network = tf.keras.Sequential(
        [tf.keras.layers.Dense(name='dense1', units=env.num_actions)]
    )

  def call(self, inputs):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).
      actions: a batch of action.

    Returns:
      Values of observations.
    """
    return self.network(inputs)
  
  def initialize_variables(self):
    data = np.random.randn(*self.dim_input).astype(np.float32)
    self.call(np.stack([data, data]))


class FCNetwork(tf.keras.Model):
  def __init__(self, env, layers=[20,20], activation='relu'):
    super(FCNetwork, self).__init__()
    self.dim_input = env.observation_space.shape
    
    net_layers = []
    for i, layer_size in enumerate(layers):
      net_layers.append(tf.keras.layers.Dense(name='dense%d'%i, units=layer_size, activation=activation, kernel_initializer='truncated_normal'))
    net_layers.append(tf.keras.layers.Dense(name='dense1', units=env.num_actions))

    self.network = tf.keras.Sequential(net_layers)

  def call(self, inputs):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).
      actions: a batch of action.

    Returns:
      Values of observations.
    """
    return self.network(inputs)
  
  def initialize_variables(self):
    data = np.random.randn(*self.dim_input).astype(np.float32)
    self.call(np.stack([data, data]))
  

class FCPenNetwork(tf.keras.Model):
  def __init__(self, env, layers=[20,20], activation='relu', dim_input=None):
    super(FCPenNetwork, self).__init__()
    self.layer_sizes = layers
    if dim_input is None:
      self.dim_input = env.observation_space.shape
    else:
      self.dim_input = dim_input
    print ('Input Dim: ', self.dim_input)
    
    net_layers = []
    net_layers.append(tf.keras.layers.Dense(name='inp_dense', units=layers[0], activation=activation, input_shape=self.dim_input))
    for i, layer_size in enumerate(layers[1:]):
      net_layers.append(tf.keras.layers.Dense(name='dense%d'%i, units=layer_size, activation=activation))
    net_layers.append(tf.keras.layers.Dense(name='dense_out', units=env.num_actions))
    print ('Net layers: ', net_layers)
    self.network = tf.keras.Sequential(net_layers)
    self.pen_network = tf.keras.Sequential(net_layers[:-1])

  def penultimate(self, inputs):
    """Gets successor features of an input"""
    return self.pen_network(inputs)
    
  def call(self, inputs):
    """Performs a forward pass given the inputs.
    Args:
      inputs: a batch of observations (tfe.Variable).
      actions: a batch of action.
    Returns:
      Values of observations.
    """
    return self.network(inputs)
  
  def initialize_variables(self):
    data = np.random.randn(*self.dim_input).astype(np.float32)
    self.call(np.stack([data, data]))


class SoftmaxNetwork(tf.keras.Model):
  def __init__(self, network):
    super(SoftmaxNetwork, self).__init__()
    self.network = network

  @property
  def variables(self):
    return self.network.variables
  
  def initialize_variables(self):
    self.network.initialize_variables()
  
  def logits(self, inputs):
    return self.network(inputs)

  def call(self, inputs, alpha=1.0):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).
      actions: a batch of action.

    Returns:
      Values of observations.
    """
    logits  = self.logits(inputs)
    if alpha != 1.0:
      logits /= alpha
    return tf.nn.softmax(logits)
  