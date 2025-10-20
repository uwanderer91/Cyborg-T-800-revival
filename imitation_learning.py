import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import NN
import data_loader

def train_imitation_learning(model, expert_data, epochs=50, lr=1e-3):
    observations = torch.FloatTensor(expert_data['observations'])
    actions = torch.LongTensor(expert_data['actions'])
    
    dataset = torch.utils.data.TensorDataset(observations, actions)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_obs, batch_actions in dataloader:
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
        
    model.save()

    return model

if __name__ == "__main__":
    model = NN.NN(
        input_channels=1,
        num_actions=7
    )

    print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    expert_data = data_loader.load_spectation()
    obs_encoded = np.reshape(expert_data["observations"], (len(expert_data["observations"]), 1, 64, 64))
    expert_data = {
        'observations': obs_encoded,
        'actions': expert_data['actions']
    }
    
    trained_model = train_imitation_learning(model, expert_data, epochs=100)