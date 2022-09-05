import os
import re

def update_file(fname, path_to_model):
    print("changing "+ fname)
    with open(fname, 'r') as f:
        lines = f.readlines()
### WD: fname seems to be read for all lines;
### WD: https://cmdlinetips.com/2016/01/opening-a-file-in-python-using-with-statement/
### WD: 'r' for read, 'w' for write.
        

    for line_num, line in enumerate(lines):
        if '.include' in line:
            regex = re.compile("\.include\s*\"(.*?45nm\_bulk\.txt)\"")
            found = regex.search(line)
            if found:
                lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)
### WD: this part is to change the path of tech file for circuits in circuit model.
### WD: re.compile(pattern, repl, string):

### WD: We can combine a regular expression pattern into pattern objects, which can be used for pattern matching. It also helps to search a pattern again without rewriting it.

### WD: regex.search() function will search the regular expression pattern and return the first occurrence.https://www.guru99.com/python-regular-expressions-complete-tutorial.html
                
                

    with open(fname, 'w') as f:
        f.writelines(lines)
        f.close()

if __name__ == '__main__':
    cur_fpath = os.path.realpath(__file__)
### WD: os.path.realpath(path): Return the canonical path of the specified filename, eliminating any symbolic links encountered in the path (if they are supported by the operating system).
### WD: __file__ is a variable that contains the path to the module that is currently being imported. Python creates a __file__ variable for itself when it is about to import a module. 
### https://www.geeksforgeeks.org/__file__-a-special-variable-in-python/



### WD: In Python, you can get the location (path) of the running script file .py with __file__. __file__ is useful for reading other files based on the location of the running file.
### WD: https://note.nkmk.me/en/python-script-file-path/
    
    
    
    parent_path = os.path.abspath(os.path.join(cur_fpath, os.pardir))
### WD: os.pardir is ‘..' for UNIX based OS and ‘::‘ for Mac OS.  
### WD: parent_path after this step is executed is: /.../eval_engines/ngspice/ngspice_inputs:


    netlist_path = os.path.join(parent_path, 'netlist')
### WD: os.path.join: Join one or more path components intelligently.
    
    spice_model = os.path.join(parent_path, 'spice_models/45nm_bulk.txt')
    
### WD: the above lines define various pathes of different models or files.

    for root, dirs, files in os.walk(netlist_path):
        for f in files:
            if f.endswith(".cir"):
                update_file(fname=os.path.join(root, f), path_to_model=spice_model)
                
### OS.walk() generate the file names in a directory tree by walking the tree either top-down or bottom-up. For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).

#root : Prints out directories only from what you specified.
#dirs : Prints out sub-directories from root.
#files : Prints out all files from root and directories.
# https://www.geeksforgeeks.org/os-walk-python/               
    
    
                
### WD: os.path: This module implements some useful functions on pathnames. 



### WD: Here, the function update_file is called, print the changing /Users/weidongcao/Box/Research_Washu/Intern/2020-MERL/DATE_2020_Berkeley/AutoCkt-master/AutoCkt-master/eval_engines/ngspice/ngspice_inputs/netlist/two_stage_opamp.cir