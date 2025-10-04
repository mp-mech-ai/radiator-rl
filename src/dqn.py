import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    def __init__(self, state_dim, hidden_size, num_layers, action_dim):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_dim)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # print(x.shape, h0.shape, c0.shape)  # Debug: print the shape of the input tensor, hn and cn
            
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Prendre la sortie du dernier pas de temps
        return out, (hn, cn)
