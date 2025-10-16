import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    def __init__(self, state_dim, hidden_size, num_layers, action_dim, dropout=0.1):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM with dropout for regularization
        self.lstm = nn.LSTM(
            input_size=state_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers>1 else 0
            )
        # Layer normalization helps with training stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(hidden_size, action_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)

        # Initialize hidden size if not provided
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take output from the last time step
        last_output = out[:, -1, :]

        # Apply layer normalization
        normalized = self.layer_norm(last_output)

        # Dropout
        dropped = self.dropout(normalized)

        # Linear forward pass
        out = self.fc(dropped)

        return out, (hn, cn)
