# wcao@merl.com

import torch.nn as nn
import torch
from torch.distributions import Categorical
from model.gan_fc_actor import GAT_actor
from model.gan_fc_critic import GAT_critic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_gan_dim, n_latent_var, num_heads, nclass_gan, state_spec_dim, nhid_fc, nclass_f, nhid_f, dropout):

        super(ActorCritic, self).__init__()
        # actor
        self.action_layer = GAT_actor(state_gan_dim, n_latent_var, num_heads, nclass_gan, state_spec_dim, nhid_fc, nclass_f, nhid_f, dropout)
        # critic
        self.value_layer = GAT_critic(state_gan_dim, n_latent_var, num_heads, nclass_gan, state_spec_dim, nhid_fc, nhid_f, dropout)


    def forward(self):
        raise NotImplementedError

    def act(self, state_gan, state_spec, adj, memory):

        state_gan = torch.from_numpy(state_gan).float().to(device)
        state_spec = torch.from_numpy(state_spec).float().to(device)
        action_probs = self.action_layer(state_gan, state_spec, adj)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob_sum = torch.sum(log_prob, dim=0)


        memory.states_gan.append(state_gan)
        memory.states_spec.append(state_spec)
        memory.actions.append(action)
        memory.logprobs.append(log_prob_sum)

        return action.detach()

    def evaluate(self, state_gan, state_spec, adj, action):

        action_probs = self.action_layer(state_gan, state_spec, adj)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        log_prob_sum = torch.sum(action_logprobs, dim=1)
        dist_entropy = dist.entropy()
        dist_entropy_sum = torch.sum(dist_entropy, dim=1)

        state_value = self.value_layer(state_gan, state_spec, adj)

        return log_prob_sum, torch.squeeze(state_value), dist_entropy_sum
