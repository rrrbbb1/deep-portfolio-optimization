import torch

class SharpeLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, w, next_r):
        
        port_returns = torch.sum(w * next_r, dim=-1)
        #print('port_returns.shape: ', port_returns.shape)

        mean_t = port_returns.mean(dim=1) 
        var_t = port_returns.var(dim=1, unbiased=False)
        sharpe_t = mean_t / torch.sqrt(var_t + self.eps)

        return -sharpe_t.mean()

class WeightPenalty(torch.nn.Module):
    def __init__(self, param: float = 0.1):
        super().__init__()
        self.param = param
    
    def forward(self, w):
        delta_w = w[:, 1:, :] - w[:, :-1, :]
        penalty_k = torch.mean(torch.abs(delta_w), dim=1)
        penalty = torch.mean(penalty_k, dim=1)

        return penalty.mean()