import numpy as np
import torch

def save_map(spec_data, filename='map_data.npz'):
    np.savez(filename, **spec_data)
    print("Data saved in "+filename)

def load_map(filename='map_data.npz'):
    file_data = np.load(filename)
    data = {}
    
    for key in file_data.files:
        data[key] = file_data[key]

    file_data.close()
    return data