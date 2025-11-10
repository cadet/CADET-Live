# -*- coding: utf-8 -*-

import h5py
import random


def load_h5_file(filename):
    #Open the H5 file in read mode
    
    with h5py.File(filename, 'r') as file:
       print("File read successfully")
       print(file)
       data = h5py.File.in_memory()
       for ds in file.keys():
           file.copy(file[ds], data["/"])
       return(data)

def save_h5_file(filename, data):
    with h5py.File(filename, 'w') as file:
        for ds in data.keys():
            data.copy(data[ds], file["/"])
        


#sim_file = load_h5_file("./data.h5")
#print('Keys: %s' % sim_file.keys())
#print(sim_file)
#        print('Keys: %s' % file.keys())
#    a_group_key = list(file.keys())[0]
#    
#    # Getting the data
#    data = list(file["input"]["model"])
#    print(data)
#    
#    more_date = file["input"]["model"]["solver"]["MAX_RESTARTS"]
#    uff = more_date[()]
#    print(uff)
#    more_date[()] = random.randint(1, 100)
#    uff2 = more_date[()]
#    print(uff2)