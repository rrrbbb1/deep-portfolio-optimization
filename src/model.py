import torch
import torch.nn.functional as F

class POptModel(torch.nn.Module):
    def __init__(self, n_asset: int, ts_dim: int, decision_step: int = 20, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        
        self.n_asset = n_asset
        self.ts_dim = ts_dim
        
        self.decision_step = decision_step

        self.norm = torch.nn.GroupNorm(ts_dim, ts_dim * n_asset)
        self.lstm = torch.nn.LSTM(
            input_size = ts_dim * n_asset,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_asset)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_norm = self.norm(x)
        x_norm = x_norm.permute(0, 2, 1)

        h_t, _ = self.lstm(x_norm)
        #print('h_t.shape: ', h_t.shape)

        h_decision = h_t[: , self.decision_step:-1, :]
        #print('h_decision.shape: ', h_decision.shape)

        scores = self.head(h_decision)
        #print('scores.shape: ', scores.shape)

        w = F.softmax(scores, dim=-1)
        #print('w.shape: ', w.shape)
        
        return w 