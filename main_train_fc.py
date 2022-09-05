import torch
import gym
import sys
import time
from buffer.data_memory import Memory
from model.ppo import PPO


wtdir = "/homes/wcao/Documents/ICML_21_AMP_EGCN_FC/train_log"
log = open(wtdir + '/train_fc_s_2.log', 'a')
def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    log.flush()

############## Hyperparameters ##############
env_name = "gym_twostageamp:twostageamp-v0"    # creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 3*7
render = False
solved_reward = 5 #0.01  # -0.002 stop training if avg_reward > solved_reward
log_interval = 25  # print avg reward in the interval
max_episodes = 500000  # max training episodes
max_timesteps = 50 # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
n_updata_episode = 30
update_timestep = n_updata_episode * max_timesteps  # update policy every n timesteps
#lr = 0.002
lr = 0.0005
betas = (0.9, 0.9)
gamma = 0.95  # discount factor
K_epochs = 10  # update policy for K epochs
eps_clip = 0.3  # clip parameter for PPO
random_seed = None
#############################################

"""
def getlr(iter): ## get learning rate
    if iter < 2e3:
        return lr
    elif iter < 5e3:
        return 0.5*lr
    elif iter < 1e4:
        return 0.25*lr
    elif iter < 1.5e4:
        return 0.1*lr
    elif iter < 2e4:
        return 0.05*lr
    else:
        return 0.025*lr
"""

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
#print(lr, betas)

# logging variables
running_reward = 0
avg_length = 0
timestep = 0
niter = 0


# training loop
mprint("Starting from Iteration %d" % niter)
for i_episode in range(1, max_episodes + 1):
    #ppo = PPO(state_dim, action_dim, n_latent_var, getlr(i_episode), betas, gamma, K_epochs, eps_clip)
    #ppo = PPO(lr =getlr(i_episode))
    state = env.reset()
    for t in range(max_timesteps):
        timestep += 1

        # Running policy_old:
        action = ppo.policy_old.act(state, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            timestep = 0

        running_reward += reward
        if render:
            env.render()
        if done:
            break

    avg_length += t

    # stop training if avg_reward > solved_reward
    target = log_interval * solved_reward

    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = round(running_reward / log_interval, 3)
        #running_reward = running_reward / log_interval
        mprint("[%05d] lr = %.2e, Episode=[%05d], avg length=[%02d], reward = %.2e" % (niter, lr, i_episode, avg_length, running_reward))
        # getlr(i_episode)
        #print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        torch.save(ppo.policy.state_dict(), './trained_model/PPO_{}.pth'.format(niter))
        niter = niter + 1

        if running_reward > target:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './trained_model/PPO_{}.pth'.format(env_name))
            break
        running_reward = 0
        avg_length = 0

mprint("Done!")
log.close()