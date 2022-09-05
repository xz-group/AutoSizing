"""
A new ckt environment based on a new structure of MDP
"""
### WD: this is the custom ckt environment (user defined) file for RL agent to be trained. It should be deeply understood.
### WD: if this file is an env, does it mean that all states, actions, and rewards are updated in this file?


import gym
from gym import spaces

import random
import psutil

from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import IPython
import itertools
#from eval_engines.util.core import *
### WD: introduce forece from eval_engines/util. This one seems not to be used

import pickle

#debug = True

from eval_engines.ngspice.TwoStageClass import *
### WD: introduce the TwoStageClass from eval_engines

#way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping
    
### WD: please the comment in test_gen_specs.ipynb



class TwoStageAmp_2(gym.Env):
    
### WD: The following are the Env methods you should know:

### 1. reset(self): Reset the environment's state. Returns observation.
### 2. step(self, action): Step the environment by one timestep. Returns observation, reward, done, info.
### 3. render(self, mode='human'): Render one frame of the environment. The default mode will do something human friendly, such as pop up a window.    

### WD: see the link to understand how to configure environment in ray. Custom env classes passed directly to the trainer must take a single env_config parameter in their constructor:  
    
    
### WD: """Custom Environment that follows gym interface"""    
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    #obtains yaml file
    path = os.getcwd()
    CIR_YAML = path+'/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml'

    #def __init__(self, env_config):
    def __init__(self):
        #super(TwoStageAmp, self).__init__()
        self.multi_goal = False
        self.generalize = True
        num_valid = 50
        self.specs_save = False
        self.valid = False
        print("Env is successfully initialized!")
        self.env_steps = 0
        with open(TwoStageAmp_2.CIR_YAML, 'r') as f:
### WD: open two_stage_opamp.yaml file
### WD: /homes/wcao/Documents/Berkely_Auto/AutoCkt-annotated/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml

            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        if self.generalize == False:
            specs = yaml_data['target_specs']
        else:
### WD: when self.generalize == true. This means that, we have already generated target design specifications before runing the training. Then, we can directly load the pre-generated design specifications.

            load_specs_path = TwoStageAmp_2.path+"/gen_specs/ngspice_specs_gen_two_stage_opamp"
### WD: TwoStageAmp.path is pointing to /homes/wcao/Documents/Berkely_Auto/AutoRF
### WD: this part is to load the (sampled) sepcifications for plolicy training.

            with open(load_specs_path, 'rb') as f:
                specs = pickle.load(f)
### WD: we load sepcs from the file.

                
### WD: this step is to call for using specifications for training. The specifications are defined in /autockt/gen_specs/ngspice_specs_gen_two_stage_opamp
### WD: Please check gen_specs.py. In function gen_data, it uses pickle to save the specifications.
            
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
### WD: I do not understand this step.
        
        if self.specs_save:
            with open("specs_"+str(num_valid)+str(random.randint(1,100000)), 'wb') as f:
                pickle.dump(self.specs, f)
### WD: default of self.specs_save is false
        
        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1 
        self.num_os = len(list(self.specs.values())[0])
        
        # param array
        params = yaml_data['params']
### WD: can check two_stage_opamp.yaml file.


        self.params = []
        self.params_id = list(params.keys())
### WD: params.keys() seems to be mp1, mn1, mp3, mn3, mn4, mn5, cc:   
  

        for value in params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)
### WD: example: mp1:  !!python/tuple [1, 100, 1] 
          
        
        #initialize sim environment
        self.sim_env = TwoStageClass(yaml_path=TwoStageAmp_2.CIR_YAML, num_process=1, path=TwoStageAmp_2.path)
### WD: Does this one recall NgSPICE to simulate the circuits based actions (the parameters of devices)?
### WD: Here, the TwoStagge Class is from eval_engines.ngspice.TwoStageClass.

### 
        
        
        self.action_meaning = [-1,0,2]
### WD: -1--reduce size; 0--keep the size; 2-- increase the size;
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))]*len(self.params_id))
        #self.action_space = spaces.Discrete(len(self.action_meaning)**len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array([TwoStageAmp_2.PERF_LOW]*2*len(self.specs_id)),
            high=np.array([TwoStageAmp_2.PERF_HIGH]*2*len(self.specs_id)))
### WD: does this define the action space?
### WD: this defines the boundary of space.
### WD: Discrete spaces are used when we have a discrete action/observation space to be defined in the environment. So spaces.Discrete(2) means that we have a discrete variable which can take one of the two possible values.        
        
### WD: self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
### Box means that you are dealing with real valued quantities. The first array np.array([-1,0,0] are the lowest accepted values, and the second np.array([+1,+1,+1]) are the highest accepted values. In this case (using the comment) we see that we have 3 available actions:
# 1. Steering: Real valued in [-1, 1]
# 2. Gas: Real valued in [0, 1]
# 3. Brake: Real valued in [0, 1]
   
### WD. Please see the link here: https://stackoverflow.com/questions/44404281/openai-gym-understanding-action-space-notation-spaces-box

### WD, the link related to space.Tuple(): https://stackoverflow.com/questions/58964267/how-to-create-an-openai-gym-observation-space-with-multiple-features
    
    
        #initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        #Get the g* (overall design spec) you want to reach
        self.global_g = []
        for spec in list(self.specs.values()):
                self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)
        self.global_g = np.array(yaml_data['normalize'])
### WD: please see the definition in two_stage_opamp.yaml file
        
        #objective number (used for validation)
        self.obj_idx = 0

    def reset(self):
        #if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os-1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0,self.num_os-1)
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star 
            else:
                idx = random.randint(0,self.num_os-1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)
        #print("num total:"+str(self.num_os))

        #applicable only when you have multiple goals, normalizes everything to some global_g
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        #initialize current parameters
        self.cur_params_idx = np.array([33, 33, 33, 33, 33, 14, 20])
        self.cur_specs = self.update(self.cur_params_idx)
        #cur_spec_norm = self.lookup(self.cur_specs, self.global_g)
        #reward = self.reward(self.cur_specs, self.specs_ideal)
### How does this one define? self.reward is a function? Please seee the reward function define below.
        
        #observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate([self.specs_ideal_norm, self.cur_params_idx])
        return self.ob
### WD: This is very important! The observations are the same as reported in that paper!


 
    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """

        #Take action that RL agent returns to change current params
        action = list(np.reshape(np.array(action),(np.array(action).shape[0],)))
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])

#        self.cur_params_idx = self.cur_params_idx + np.array(self.action_arr[int(action)])
        self.cur_params_idx = np.clip(self.cur_params_idx, [0]*len(self.params_id), [(len(param_vec)-1) for param_vec in self.params])
        #Get current specs and normalize
        self.cur_specs = self.update(self.cur_params_idx)
        #cur_spec_norm  = self.lookup(self.cur_specs, self.global_g)
        reward = self.reward(self.cur_specs, self.specs_ideal)
        done = False

        #incentivize reaching goal state
        if (reward >= 10):
            done = True
        #     print('-'*10)
        #     print('params = ', self.cur_params_idx)
        #     print('specs:', self.cur_specs)
        #     print('ideal specs:', self.specs_ideal)
        #     print('re:', reward)
        #     print('-'*10)

        self.ob = np.concatenate([self.specs_ideal_norm, self.cur_params_idx])
        self.env_steps = self.env_steps + 1

        #print('cur ob:' + str(self.cur_specs))
        #print('ideal spec:' + str(self.specs_ideal))
        #print(reward)
        return self.ob, reward, done, {}
### WD: how to understand the action values? In each step, is it an increament or decrement.
    
    

    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec-goal_spec)/(goal_spec+spec)
        return norm_spec
### WD: not sure if this is mapped with function defined in the paper.
### WD: this seems to be matched with the reward
    
    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        rel_specs = self.lookup(spec, goal_spec)
        pos_val = [] 
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if(self.specs_id[i] == 'ibias_max'):
                rel_spec = rel_spec*-1.0#/10.0
### WD: do not understand this step!
            if rel_spec < 0:
                reward += rel_spec
                pos_val.append(0)
            else:
                pos_val.append(1)

        return reward if reward < -0.05  else 10
        #return reward if reward < -0.02 else 10


### WD: deeply understand how the reward is defined here?   
 

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        #impose constraint tail1 = in
        #params_idx[0] = params_idx[3]
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id,params)))]
        
        #run param vals and simulate
        cur_specs = OrderedDict(sorted(self.sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k:k[0]))
        cur_specs = np.array(list(cur_specs.values()))
### WD: simulate the returned device paprameters to get the current specifications? 
### WD: create_design_and_simulate is function defined in eval_engines.ngspice.ngspice_wrapper.py

        

        return cur_specs

#def main():
#  env_config = {"generalize":True, "valid":True}
#  env = TwoStageAmp(env_config)
#  env.reset()
#  env.step([2,2,2,2,2,2,2])

#  IPython.embed()
### WD: The embed() function of IPython module makes it possible to embed IPython in your Python codes’ namespace. Thereby you can leverage IPython features like object introspection and tab completion, in default Python environment.

#if __name__ == "__main__":
#  main()
