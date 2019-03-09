import torch
import numpy as np

import debugq.pytorch_util as ptu


class StopMode(object):
    def __init__(self):
        self.stopped = False
        self.reset()

    def reset(self):
        self.stopped = False

    def update(self, **kwargs):
        pass
    
    def stop(self):
        self.stopped = True

    def check(self):
        return self.stopped


class AtolStop(StopMode):
    def __init__(self, atol=1e-8):
        super(AtolStop, self).__init__()
        self.atol = atol
    
    def update(self, critic_loss=float('inf'), **kwargs):
        if critic_loss <= self.atol:
            self.stop()

            
class RtolStop(StopMode):
    def __init__(self, rtol=1e-5):
        super(RtolStop, self).__init__()
        self.rtol = rtol

    def reset(self):
        self.previous_loss = float('inf')
    
    def update(self, critic_loss=float('inf'), **kwargs):
        if self.previous_loss == float('inf'):
            self.previous_loss = critic_loss
            return
        if np.abs(critic_loss - self.previous_loss)/ self.previous_loss <= self.rtol:
            self.stop()
        self.previous_loss = critic_loss


class ValidationLoss(StopMode):
    def __init__(self):
        super(ValidationLoss, self).__init__()

    def reset(self):
        self.validation_loss= float('inf')
        self.validation_k = 0
        self.validation_k_counter = 0
    
    def update(self, q_network=None, fqi=None, all_target_q=None, **kwargs):
        # compute oracle loss
        q_values = q_network(fqi.all_states).detach()
        weights = ptu.tensor(fqi.validation_sa_weights)
        oracle_loss = ptu.to_numpy(torch.sum(weights*((q_values- all_target_q)**2)))
        prev_oracle_loss = self.validation_loss

        self.validation_k_counter += 1
        if oracle_loss < prev_oracle_loss:
            self.best_validation_qs = q_values
            self.validation_loss = oracle_loss
            self.validation_k = self.validation_k_counter-1
