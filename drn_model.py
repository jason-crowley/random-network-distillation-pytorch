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
    def __init__(self, novelty_rnd, input_size, output_size, use_cuda=False):
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.n_l = 7
        self.novelty_rnd = RNDAgent(novelty_rnd, use_cuda=True)
        self.novelty_threshold = 1.5
        self.frequency_rnd = RNDAgent(RNDModel(input_size, output_size), use_cuda=True)
        self.frequency_threshold = 60
        self.nov_state_buffer = deque(maxlen=100)

    def train(self, trajectory):
        self.nov_state_buffer.extend(self.get_rel_nov_states(trajectory))
        self.frequency_rnd.train(self.nov_state_buffer)

    def get_rel_nov_states(self, trajectory):
        trajectory_tensor = torch.FloatTensor(trajectory).to(self.device)
        novelty_vals = [self.novelty_rnd.forward(torch.unsqueeze(s, 0)) for s in trajectory_tensor]

        def get_rel_novelty(i):
            visits_before = novelty_vals[i - self.n_l : i]
            visits_after = novelty_vals[i : i + self.n_l]

            return np.sqrt(np.sum(visits_before) / np.sum(visits_after))

        rel_nov_states = []
        for i in range(self.n_l, len(trajectory) - self.n_l):
            rel_nov = get_rel_novelty(i)
            obs = trajectory[i]
            if rel_nov > self.novelty_threshold:
                rel_nov_states.append(obs)
        return rel_nov_states

    def get_subgoals(self, trajectory):
        subgoals = []
        for obs in self.get_rel_nov_states(trajectory):
            freq = self.frequency_rnd.forward(torch.FloatTensor([obs]).to(self.device))
            if freq < self.frequency_threshold:
                subgoals.append(obs)
        return subgoals
