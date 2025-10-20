import numpy as np
import torch

def save_spectation(spec_data, filename='spectation_data.npz'):
    np.savez(filename, **spec_data)
    print("Данные сохранены в "+filename)

def load_spectation(filename='spectation_data.npz'):
    spec_data = np.load(filename)
    expert_data = {}
    
    for key in spec_data.files:
        expert_data[key] = spec_data[key]

    spec_data.close()
    return expert_data