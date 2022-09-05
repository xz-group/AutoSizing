import torch
import gym
import sys
import time
from buffer.data_memory_gcn_fc import Memory
from model.ppo_gcn_fc import PPO
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

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
G.add_node(7, name='vdd')
G.add_node(8, name='gnd')

#Define the edges and the edges to the graph
# edges = [(0, 1), (0, 2), (0, 6), (1, 2), (1, 3), (1, 6),
#          (2, 5), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6)]

edges = [(0, 1), (0, 2), (0, 6), (0, 7), (1, 2), (1, 3), (1, 6), (2, 5),
         (2, 6), (2, 7), (3, 4), (3, 5), (3, 8), (4, 5), (4, 8), (5, 6), (5, 8)]

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

print('Adjacency Matrix of added self-loops G (A_hat):\n', A_hat)

#Get the Degree Matrix of the added self-loops graph
#Deg_Mat = G_self_loops.degree()
#print('Degree Matrix of added self-loops G (D): ', Deg_Mat)

#Convert the Degree Matrix to a N x N matrix where N is the number of nodes
#D = np.diag([deg for (n,deg) in list(Deg_Mat)])
D = np.diag(np.sum(A_hat, axis=0))
print('Degree Matrix of added self-loops G as numpy array (D):\n', D)

#Find the inverse of Degree Matrix (D)
#D_inv = np.linalg.inv(D)
#print('Inverse of D:\n', D_inv)

#Symmetrically-normalization
D_half_norm = fractional_matrix_power(D, -0.5)
DAD = D_half_norm.dot(A_hat).dot(D_half_norm)
print('DAD:\n', DAD)


#"""

wtdir = "/homes/wcao/Documents/ICML_21_AMP_EGCN_FC/train_log"
log = open(wtdir + '/train_gcn_fc_s_4.log', 'a')
def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    log.flush()

############## Hyperparameters ##############
env_name = "gym_twostageamp:twostageamp-v0"    # creating environment
env = gym.make(env_name)
state_gcn_dim = 6 #env.observation_space.shape[0]
state_spec_dim = 8
action_dim = 3
render = False
solved_reward = 5  # -0.002 stop training if avg_reward > solved_reward
log_interval = 25  # print avg reward in the interval
max_episodes = 500000  # max training episodes
max_timesteps = 50  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
n_updata_episode = 30
update_timestep = n_updata_episode * max_timesteps  # update policy every n timesteps
#lr = 0.002
lr = 0.0005
betas = (0.9, 0.9)
gamma = 0.95  # discount factor
K_epochs = 10  # update policy for K epochs
eps_clip = 0.3  # clip parameter for PPO
#dropout = 0
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
ppo = PPO(state_gcn_dim, state_spec_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
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

    ### one-hot encoding of gcn node features
    state = env.reset()
    #node_feature_type = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
    #node_feature_type = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [2, 0, 0, 0], [3, 0, 1, 0],
    #                              [4, 0, 1, 0], [5, 0, 1, 0], [6, 1, 0, 0]])

    node_feature_type = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                                  [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])

    device_node_feature_para = np.reshape(state[8:15], (len(state[8:15]), 1))
    ### power node featurs
    power_node_feature_para = np.array([[1.0], [0]])
    ### concatenate power node features
    node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
    state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)

    state_spec = state[0:8]

    #state = np.reshape(state, (len(state), 1))
    for t in range(max_timesteps):
        timestep += 1

        # Running policy_old:
        #action = ppo.policy_old.act(np.reshape(state, (len(state), 1)), DAD, memory)
        action = ppo.policy_old.act(state_gcn, state_spec, DAD, memory)
        states, reward, done, _ = env.step(action)
        #state_gan = np.concatenate((node_feature_type, np.reshape(states[8:15], (len(states[8:15]), 1))), axis=1)
        device_node_feature_para = np.reshape(states[8:15] / 100, (len(states[8:15]), 1)) ### typo
        node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
        state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)
        state_spec = states[0:8]
        #np.reshape(node_feature_para, (len(node_feature_para), 1))

        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if timestep % update_timestep == 0:
            ppo.update(DAD, memory)
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


#"""