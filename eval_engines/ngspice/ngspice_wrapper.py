import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import IPython
debug = False
#debug = True

### WD: This code is to call Ngspice into the training loop.

class NgSpiceWrapper(object):

    BASE_TMP_DIR = os.path.abspath("/tmp/ckt_da")
### WD: check that the simulation data are stored in this path.

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        design_netlist = yaml_data['dsn_netlist']
### WD: dsn_netlist is defined in yaml file.---- dsn_netlist: "eval_engines/ngspice/ngspice_inputs/netlist/two_stage_opamp.cir"
### WD: the path of __init__ is defined in ngspice_vanilla_opamp.py file 
### WD: TwoStageAmp.path: /homes/wcao/Documents/Berkely_Auto/AutoCkt-annotated




        design_netlist = path+'/'+ design_netlist
### WD: need to figure out the path (absolute path) here. Figured out, see above
 
        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
### WD: this is the initial base_design_name used. It is the first one used in the training.

        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)
### WD: This is a folder in /tmp/ckt_da, and its name is called designs_two_stage_opamp.

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)
### WD: os.makedirs() method in Python is used to create a directory recursively. That means while making leaf directory if any intermediate-level directory is missing, os.makedirs() method will create them all.
### WD: https://www.geeksforgeeks.org/python-os-makedirs-method/

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
### WD: read the netlist in the original netlist file.
        raw_file.close()

    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):
        design_folder = os.path.join(self.gen_dir, new_fname)+str(random.randint(0,10000))
### WD: the folder is in the /tmp/ckt_da
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')
### WD: this fpath is similar to the the .cir file defined in /tmp/ckt_da/designs_two_stage_opamp/two_stage_opamp_9_9_90_15_8_69_3.3e-124146

        lines = copy.deepcopy(self.tmp_lines)
### WD: Here, rewrite the netlist in the original netlist file into the .cir file defined in /tmp/ckt_da/designs_two_stage_opamp/two_stage_opamp_9_9_90_15_8_69_3.3e-124146
        for line_num, line in enumerate(lines):
            if '.include' in line:
                regex = re.compile("\.include\s*\"(.*?)\"")
                found = regex.search(line)
                if found:
                    # current_fpath = os.path.realpath(__file__)
                    # parent_path = os.path.abspath(os.path.join(current_fpath, os.pardir))
                    # parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
                    # path_to_model = os.path.join(parent_path, 'spice_models/45nm_bulk.txt')
                    # lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)
                    pass # do not change the model path
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            if 'wrdata' in line:
                regex = re.compile("wrdata\s*(\w+\.\w+)\s*")
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1))
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement)

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath
### WD: the folder is in the /tmp/ckt_da



    
    def simulate(self, fpath):
        info = 0 # this means no error occurred
        command = "ngspice -b %s >/dev/null 2>&1" %fpath
        #command = "hpeesofsim -r /homes/wcao/Documents/ADS_run_cmd/sim_result/test.raw /homes/wcao/Documents/ADS_run_cmd/netlist/inv_tes.log"
### WD: This is where to call NgSPICE.
### WD: %fpath is the cir file sent into NgSpice for simulation or the simulation outs from ngspice?

        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
           # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info
### WD: this is to call NgSpice for simulation. Totally understood.



### WD: is this step to create design and simulate.
    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath)
### WD: I think the path define the file path which stores the netlist.
### WD: simulate the the design from create_design function
### WD: Where is this state coming from?
        
        specs = self.translate_result(design_folder)
        return state, specs, info


    def run(self, states, design_names=None, verbose=False):
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result
