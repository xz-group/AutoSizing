import gym
from model.ppo import PPO
#from buffer.data_memory import Memory
import torch
from rollout.actor_critic_model_fc_rollout import ActorCritic
import numpy as np
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_name = "gym_twostageamp:twostageamp-v0"
env_name_used = "gym_twostageamp:twostageamp-v1"
env = gym.make(env_name)
num_val_specs = 50
traj_len = 50
state_dim = 15
#env.observation_space.shape[0]
action_dim = 3*7
# render = False
# solved_reward = -0.001  # stop training if avg_reward > solved_reward
# log_interval = 50  # print avg reward in the interval
# max_episodes = 50000  # max training episodes
# max_timesteps = 30 # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
# update_timestep = 30*30  # update policy every n timesteps
# lr = 0.001
# betas = (0.9, 0.9)
# gamma = 0.95  # discount factor
# K_epochs = 10  # update policy for K epochs
# eps_clip = 0.3  # clip parameter for PPO
# random_seed = None

#directory = "/homes/wcao/Documents/Berkely_Auto/Auto_RF_v1/"
#filename = "PPO_{}.pth".format(env_name)

directory = "/homes/wcao/Documents/ICML_21_AMP_EGCN_FC/trained_model/FC_policy/"
filename = "PPO_3000.pth"

#trained_model = torch.load(directory + filename)
agent = ActorCritic(state_dim, action_dim, n_latent_var)
agent.load_state_dict(torch.load(directory + filename))

# action = ppo.policy_old.act(state, memory)
# state, reward, done, _ = env.step(action)
# ep_reward += reward

def rollout(agent, env, out="assdf"):

    norm_spec_ref = env.global_g
    spec_num = len(env.specs)
    rollouts = []
    next_states = []
    obs_reached = []
    obs_nreached = []
    action_array = []
    action_arr_comp = []
    rollout_steps = 0
    reached_spec = 0
    while rollout_steps < num_val_specs:
        if out is not None:
            rollout_num = []
        #state = env.reset()[8:16]
        state = env.reset()
        done = False
        reward_total = 0.0
        steps = 0
        while not done and steps < traj_len:
            action = agent.act(state)
            action_array.append(action)
            next_state, reward, done, _ = env.step(action)
            print(action)
            print(reward)
            print(done)
            reward_total += reward
            if out is not None:
                rollout_num.append(reward)
                #next_states.append(next_state[8:16])
                next_states.append(next_state)
            steps += 1
            #state = next_state[8:16]
            state = next_state
            state_spec = next_state
        norm_ideal_spec = state_spec[spec_num:spec_num + spec_num]
        ideal_spec = unlookup(norm_ideal_spec, norm_spec_ref)
        if done == True:
            reached_spec += 1
            obs_reached.append(ideal_spec)
            action_arr_comp.append(action_array)
            action_array = []
            pickle.dump(action_arr_comp, open("action_arr_test", "wb"))
        else:
            obs_nreached.append(ideal_spec)  # save unreached observation
            action_array = []
        if out is not None:
            rollouts.append(rollout_num)
        print("Episode reward", reward_total)
        rollout_steps += 1
        # if out is not None:
        # pickle.dump(rollouts, open(str(out)+'reward', "wb"))
        pickle.dump(obs_reached, open("opamp_obs_reached_test", "wb"))
        pickle.dump(obs_nreached, open("opamp_obs_nreached_test", "wb"))
        print("Specs reached: " + str(reached_spec) + "/" + str(len(obs_nreached)))

    print("Num specs reached: " + str(reached_spec) + "/" + str(num_val_specs))

def unlookup(norm_spec, goal_spec):
    spec = -1*np.multiply((norm_spec+1), goal_spec)/(norm_spec-1)
    return spec


if __name__ == '__main__':
    rollout(agent, env)
