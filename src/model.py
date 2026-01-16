import torch
import torch.nn.functional as F

class POptModel(torch.nn.Module):
    def __init__(self, n_asset: int, decision_step: int = 20, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        
        self.n_asset = n_asset
        self.decision_step = decision_step

        self.price_norm = torch.nn.RMSNorm(n_asset)
        self.return_norm = torch.nn.RMSNorm(n_asset)

        self.lstm = torch.nn.LSTM(
            input_size = 2 * n_asset,
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
        prices = x[:, :, :self.n_asset]
        returns = x[:, :, self.n_asset:]

        norm_price = self.price_norm(prices)
        norm_ret = self.return_norm(returns)

        norm_x = torch.cat((norm_price, norm_ret), dim=-1)
        h_t, _ = self.lstm(norm_x)
        #print('h_t.shape: ', h_t.shape)

        h_decision = h_t[: , self.decision_step:-1, :]
        #print('h_decision.shape: ', h_decision.shape)

        scores = self.head(h_decision)
        #print('scores.shape: ', scores.shape)

        w = F.softmax(scores, dim=-1)
        #print('w.shape: ', w.shape)
        
        return w 