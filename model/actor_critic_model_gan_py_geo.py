# wcao@merl.com

import torch.nn as nn
import torch
from torch.distributions import Categorical
from model.gan_actor_py_geo import GAT_actor
from model.gan_critic_py_geo import GAT_critic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_latent_var, first_heads, output_heads, action_dim, dropout):

        super(ActorCritic, self).__init__()
        # actor
        self.action_layer = GAT_actor(state_dim, n_latent_var, first_heads, output_heads, action_dim, dropout)
        # critic
        self.value_layer = GAT_critic(state_dim, n_latent_var, first_heads, output_heads, action_dim, dropout)


    def forward(self):
        raise NotImplementedError

    def act(self, state, adj, memory):

        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state, adj)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob_sum = torch.sum(log_prob, dim=0)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob_sum)

        return action.detach()

    def evaluate(self, state, adj, action):

        action_probs = self.action_layer(state, adj)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        log_prob_sum = torch.sum(action_logprobs, dim=1)
        dist_entropy = dist.entropy()
        dist_entropy_sum = torch.sum(dist_entropy, dim=1)

        state_value = self.value_layer(state, adj)

        return log_prob_sum, torch.squeeze(state_value), dist_entropy_sum
