import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import load_data

def test_300w_load():
    t_x, t_y, v_x, v_y = load_data.load_300w_train()
    print(t_x.shape)

def test_aflw_load():
    t_x, t_y, v_x, v_y = load_data.load_aflw_train()
    print(t_y.shape)

if __name__=='__main__':
    test_aflw_load()