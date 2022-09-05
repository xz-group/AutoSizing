import torch
import gym
import sys
import time
from buffer.data_memory_gan import Memory
from model.ppo_gan_py_geo_v1 import PPO
import numpy as np
import networkx as nx



###### generate circuits graph ######
G = nx.Graph(name='twostageamp')
#Create nodes, considering there are 7 unique nodes in twostageamp. Note that the differential structure.
G.add_node(0, name='mp1')
G.add_node(1, name='mn1')
G.add_node(2, name='mp3')
G.add_node(3, name='mn3')
G.add_node(4, name='mn4')
G.add_node(5, name='mn5')
G.add_node(6, name='cc')

#Define the edges and the edges to the graph
edges = [(0, 1), (0, 2), (0, 6), (1, 2), (1, 3), (1, 6),
         (2, 5), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6)]
G.add_edges_from(edges)

#See graph info
print('Graph Info:\n', nx.info(G))

#Inspect the node features
print('\nGraph Nodes: ', G.nodes.data())

#Plot the graph
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()

#Get the Adjacency Matrix (A) as numpy array
A = np.array(nx.attr_matrix(G)[0])
#Add Self Loops
G_self_loops = G.copy()

self_loops = []
for i in range(G.number_of_nodes()):
    self_loops.append((i,i))

G_self_loops.add_edges_from(self_loops)
print('Edges of G with self-loops:\n', G_self_loops.edges)

#Get the Adjacency Matrix (A) of added self-lopps graph
A_hat = np.array(nx.attr_matrix(G_self_loops)[0])
#I = np.identity(A.shape[0])  # create Identity Matrix of A
#A_hat_1 = A + I

A_hat = torch.reshape(torch.tensor(A_hat), (1, 7, 7))
print('Adjacency Matrix of added self-loops G (A_hat):\n', A_hat)


#"""

wtdir = "/homes/wcao/Documents/ICML_21_AMP_EGCN_FC/train_log"
log = open(wtdir + '/train_gan_fc_geo_v1.log', 'a')
def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    log.flush()

############## Hyperparameters ##############
env_name = "gym_twostageamp:twostageamp-v1"    # creating environment
env = gym.make(env_name)
state_dim = 1  #env.observation_space.shape[0]
action_dim = 3  # action space of each device
render = False
solved_reward = -0.002  # stop training if avg_reward > solved_reward
log_interval = 25  # print avg reward in the interval
max_episodes = 50000  # max training episodes
max_timesteps = 50  # max timesteps in one episode
num_heads = 3  # number of heads in graph attention neural network
n_latent_var = 16 * num_heads  # number of variables in hidden layer
n_updata_episode = 30  # number of episodes to update the policy network
update_timestep = n_updata_episode * max_timesteps  # update policy every n timesteps
#lr = 0.002

#output_heads = 1
lr = 0.0005
betas = (0.9, 0.9)
gamma = 0.95  # discount factor
K_epochs = 10  # update policy for K epochs
eps_clip = 0.3  # clip parameter for PPO
dropout = 0.2
#############################################

# 
# def getlr(iter): ## get learning rate
#     if iter < 2e3:
#         return lr
#     elif iter < 5e3:
#         return 0.5*lr
#     elif iter < 1e4:
#         return 0.25*lr
#     elif iter < 1.5e4:
#         return 0.1*lr
#     elif iter < 2e4:
#         return 0.05*lr
#     else:
#         return 0.025*lr
# 

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, num_heads, dropout, lr, betas, gamma, K_epochs, eps_clip)
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
    #state = np.reshape(state, (len(state), 1))
    for t in range(max_timesteps):
        timestep += 1

        # Running policy_old:
        #a = np.reshape(state, (len(state), 1))
        #b = a.dim()
        action = ppo.policy_old.act(np.reshape(state, (len(state), 1)), A_hat, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if timestep % update_timestep == 0:
            ppo.update(A_hat, memory)
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


#"""