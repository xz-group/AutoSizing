import gym
import torch
from rollout.actor_critic_model_gcn_fc_rollout import ActorCritic
import numpy as np
import pickle
import networkx as nx
#import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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







env_name = "gym_twostageamp:twostageamp-v0"
env_name_used = "gym_twostageamp:twostageamp-v1"
env = gym.make(env_name)

state_gcn_dim = 6 #env.observation_space.shape[0]
state_spec_dim = 8
action_dim = 3
num_val_specs = 1
traj_len = 50



n_latent_var = 64  # number of variables in hidden layer


#directory = "/homes/wcao/Documents/Berkely_Auto/Auto_RF_v1/"
#filename = "PPO_{}.pth".format(env_name)

directory = "/homes/wcao/Documents/ICML_21_AMP_EGCN_FC/trained_model/"
filename = "PPO_3434.pth"


agent = ActorCritic(state_gcn_dim, state_spec_dim, n_latent_var, action_dim)
a = torch.load(directory + filename)
agent.load_state_dict(torch.load(directory + filename))

node_feature_type = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                                  [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])

power_node_feature_para = np.array([[1.0], [0]])



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

        device_node_feature_para = np.reshape(state[8:15], (len(state[8:15]), 1))
        ### concatenate power node features
        node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
        state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)
        state_spec = state[0:8]

        done = False
        reward_total = 0.0
        steps = 0
        while not done and steps < traj_len:
            #action = agent.act(state)
            action = agent.act(state_gcn, state_spec, DAD)
            action_array.append(action)
            #next_state, reward, done, _ = env.step(action)
            states, reward, done, _ = env.step(action)

            device_node_feature_para = np.reshape(state[8:15] / 100, (len(state[8:15]), 1))
            node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
            state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)
            state_spec = states[0:8]




            print(action)
            print(reward)
            print(done)
            reward_total += reward
            if out is not None:
                rollout_num.append(reward)
                next_states.append(states[8:16])
            steps += 1
            #state = next_state[8:16]
            #state = next_state
            #state_spec = states



        norm_ideal_spec = state_spec[spec_num:spec_num + spec_num]
        ideal_spec = unlookup(norm_ideal_spec, norm_spec_ref)
        if done == True:
            reached_spec += 1
            obs_reached.append(ideal_spec)
            action_arr_comp.append(action_array)
            action_array = []
            #pickle.dump(action_arr_comp, open("action_arr_test", "wb"))
        else:
            obs_nreached.append(ideal_spec)  # save unreached observation
            action_array = []
        if out is not None:
            rollouts.append(rollout_num)
        print("Episode reward", reward_total)
        rollout_steps += 1
        # if out is not None:
        # pickle.dump(rollouts, open(str(out)+'reward', "wb"))
        #pickle.dump(obs_reached, open("opamp_obs_reached_test", "wb"))
        #pickle.dump(obs_nreached, open("opamp_obs_nreached_test", "wb"))
        print("Specs reached: " + str(reached_spec) + "/" + str(len(obs_nreached)))

    print("Num specs reached: " + str(reached_spec) + "/" + str(num_val_specs))

def unlookup(norm_spec, goal_spec):
    spec = -1*np.multiply((norm_spec+1), goal_spec)/(norm_spec-1)
    return spec


if __name__ == '__main__':
    rollout(agent, env)
