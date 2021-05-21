from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import RNDModel


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


class RNDAgent:
    def __init__(
        self,
        rnd_model,
        learning_rate=1e-4,
        epoch=3,
        batch_size=128,
        update_proportion=0.25,
        use_cuda=False,
    ):
        self.epoch = epoch
        self.batch_size = batch_size
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.rnd = rnd_model
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=learning_rate)

        self.rnd = self.rnd.to(self.device)

    def forward(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)

        predict_next_feature, target_next_feature = self.rnd(obs)
        output = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return output.data.cpu().numpy()

    def train(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)

        sample_range = np.arange(len(obs))
        forward_mse = nn.MSELoss(reduction='none')

        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(len(obs) // self.batch_size):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(obs[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------

                self.optimizer.zero_grad()
                loss.backward()
                global_grad_norm_(self.rnd.predictor.parameters())
                self.optimizer.step()


class DeepRelNov:
    def __init__(self, nov_rnd, input_size, output_size, use_cuda=False, rel_nov_percentile=90, freq_percentile=90):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.n_l = 7
        self.thresh_adjustment_rate = 0.1

        self.rel_nov_percentile = rel_nov_percentile
        self.rel_nov_thresh = 0
        self.nov_rnd = RNDAgent(nov_rnd, use_cuda=True)

        self.freq_percentile = freq_percentile
        self.freq_thresh = 0
        self.freq_rnd = RNDAgent(RNDModel(input_size, output_size), use_cuda=True)

        self.rel_nov_state_buf = deque(maxlen=100)

    def train(self, trajectory):
        rel_nov_vals = self.get_rel_nov_vals(trajectory)
        self.update_rel_nov_thresh(rel_nov_vals)

        rel_nov_states = self.get_rel_nov_states(rel_nov_vals, trajectory)
        self.rel_nov_state_buf.extend(rel_nov_states)

        self.freq_rnd.train(self.rel_nov_state_buf)

        freq_vals = self.get_freq_vals(rel_nov_states)
        self.update_freq_thresh(freq_vals)

    def update_rel_nov_thresh(self, rel_nov_vals):
        p = np.percentile(rel_nov_vals, self.rel_nov_percentile)
        self.rel_nov_thresh = self.thresh_adjustment_rate * p + (1 - self.thresh_adjustment_rate) * self.rel_nov_thresh

    def update_freq_thresh(self, freq_vals):
        if len(freq_vals) > 0:
            p = np.percentile(freq_vals, self.freq_percentile)
            self.freq_thresh = self.thresh_adjustment_rate * p + (1 - self.thresh_adjustment_rate) * self.freq_thresh

    def get_rel_nov_vals(self, trajectory):
        # Gets the relative novelty values for the states in the trajectory
        trajectory_tensor = torch.FloatTensor(trajectory).to(self.device)
        nov_vals = self.nov_rnd.forward(trajectory_tensor)

        def get_rel_nov(i):
            if i < self.n_l or i >= len(trajectory) - self.n_l:
                return 0

            visits_before = nov_vals[i - self.n_l : i]
            visits_after = nov_vals[i : i + self.n_l]

            return np.sum(visits_after) / np.sum(visits_before)

        return [get_rel_nov(i) for i in range(len(trajectory))]

    def get_rel_nov_states(self, rel_nov_vals, trajectory):
        # Gets states in the trajectory whose relative novelty is greater than
        # the novelty threshold
        return trajectory[rel_nov_vals > self.rel_nov_thresh]

        # rel_nov_states = []
        # for i in range(self.n_l, len(trajectory) - self.n_l):
        #     rel_nov = rel_nov_vals[i]
        #     obs = trajectory[i]
        #     if rel_nov > self.novelty_threshold:
        #         rel_nov_states.append(obs)
        # return rel_nov_states

    def get_freq_vals(self, rel_nov_states):
        states_tensor = torch.FloatTensor(rel_nov_states).to(self.device)
        return self.freq_rnd.forward(states_tensor)

    def get_subgoals(self, trajectory):
        subgoals = []
        rel_nov_vals = self.get_rel_nov_vals(trajectory)
        rel_nov_states = self.get_rel_nov_states(rel_nov_vals, trajectory)
        for obs in rel_nov_states:
            freq = self.freq_rnd.forward(torch.FloatTensor([obs]).to(self.device))
            if freq < self.freq_thresh:
                subgoals.append(obs)
        return subgoals
