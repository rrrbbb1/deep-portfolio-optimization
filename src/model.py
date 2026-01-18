import torch
import torch.nn.functional as F

class POptModel(torch.nn.Module):
    def __init__(self, n_asset: int, ts_dim: int, decision_step: int = 20, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        
        self.n_asset = n_asset
        self.ts_dim = ts_dim
        
        self.decision_step = decision_step
        self.lstm = torch.nn.LSTM(
            input_size = ts_dim * n_asset,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_asset)
        )
    
    def forward(self, x):
        #print("(model) x.shape: ", x.shape)

        h_t, _ = self.lstm(x)
        #print('(model) h_t.shape: ', h_t.shape)

        h_decision = h_t[:, self.decision_step:-1, :]
        #print('(model) h_decision.shape: ', h_decision.shape)

        scores = self.head(h_decision)
        #print('(model) scores.shape: ', scores.shape)
        #print('(model) scores: ', scores[0])
        w = F.softmax(scores, dim=-1)
        #print('(model) w.shape: ', w.shape)
        #print('(model) w: ', w[0])

        return w