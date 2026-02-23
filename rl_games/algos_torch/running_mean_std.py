from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class RunningMeanStd(nn.Module):
    """Tracks the running mean and variance of input data."""
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, update_in_eval=False, name=None):
        super(RunningMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        self.update_in_eval = update_in_eval
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            elif len(self.insize) == 2:
                self.axis = [0, 2]
            elif len(self.insize) == 1:
                self.axis = [0]
            else:
                # Fallback or error?
                # e.g. raise ValueError(f"Unexpected insize length: {len(self.insize)}")
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count:int):
        label = f'RunningMeanStd({name})' if name else 'RunningMeanStd'
        print(f'{label}: {insize} per_channel={per_channel} update_in_eval={update_in_eval}')
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, denorm:bool=False, mask:Optional[torch.Tensor]=None):
        if self.training or self.update_in_eval:
            if mask is not None:
                mean, var = torch_ext.get_mean_var_with_masks(input, mask)
            else:
                mean = input.mean(self.axis) # along channel axis
                var = input.var(self.axis, unbiased=False)

            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size(0)
            )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            elif len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            elif len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
            else:
                current_mean = self.running_mean
                current_var = self.running_var
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        # get output
        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class RunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, normalize_keys=None):
        """Maintains running statistics for each observation key provided as a dictionary.
        
        Args:
            normalize_keys: ONLY these keys are normalized. If None, all keys are normalized.
        """
        assert(isinstance(insize, dict))
        super(RunningMeanStdObs, self).__init__()
        self.normalize_keys = normalize_keys
        print(f'RunningMeanStdObs: normalize_keys={normalize_keys}, obs_keys={list(insize.keys())}')
        self.running_mean_std = nn.ModuleDict({
            k : RunningMeanStd(v, epsilon, per_channel, norm_only, name=k)
            for k, v in insize.items()
            if normalize_keys is None or k in normalize_keys
        })
    
    def forward(self, input, denorm:bool=False):
        res = {}
        for k, v in input.items():
            if (self.normalize_keys is None or k in self.normalize_keys) and k in self.running_mean_std:
                res[k] = self.running_mean_std[k](v, denorm)
            else:
                res[k] = v
        return res