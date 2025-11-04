import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import NN
import data_loader
from frame_stacker import *

torch.backends.cudnn.benchmark = True

def to_dataloader(expert_data, batch_size):
    observations = torch.FloatTensor(expert_data['observations'])
    actions = torch.LongTensor(expert_data['actions'])

    dataset = torch.utils.data.TensorDataset(observations, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def train_imitation_learning(model, expert_data, epochs=50, batch_size=256, learning_rate=0.01, subtask_label="none"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    torch.set_float32_matmul_precision('medium')

    dataloader = to_dataloader(expert_data, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            batch_obs, batch_actions = [x.to(device) for x in batch]
            logits = model(batch_obs)
            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch_actions).sum().item()
            total += batch_actions.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
        
    model.save("actor_model.npz")

    return model

if __name__ == "__main__":

    epochs=200
    batch_size=512
    learning_rate=0.01
    subtask_label = "shooting"
    frame_stack = 1

    frame_stacker = FrameStacker(frame_stack=frame_stack)

    stacked_obs = []
    expert_data = data_loader.load_map("data_"+subtask_label+".npz")
    new_obs = None
    for obs in expert_data["observations"]:
        if new_obs is None:
            frame_stacker.reset(obs)
        else:
            frame_stacker.append(obs)
        
        stacked_obs.append(frame_stacker.get_stacked_frames())
        new_obs = obs

    obs_encoded = np.reshape(stacked_obs, (len(stacked_obs), frame_stacker.frame_stack, 64, 64))
    expert_data = {
        'num_actions': expert_data['num_actions'],
        'observations': obs_encoded,
        'actions': expert_data['actions']
    }

    model = NN.PolicyNN(
        input_channels=1*frame_stacker.frame_stack,
        num_actions=expert_data['num_actions']
    )
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    
    trained_model = train_imitation_learning(model, expert_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, subtask_label=subtask_label)