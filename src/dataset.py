import torch
import numpy as np
from torch.utils.data import Dataset

class PortfolioDataset(Dataset):
    def __init__(self, returns_df: pd.DataFrame, lookback: int = 60, decision_step: int = 10, n_asset: int = 10, n_samples: int = 100_000):
        self.returns = returns_df.values.astype(np.float32)
        self.T, self.N = self.returns.shape

        self.lookback = lookback
        self.decision_step = decision_step
        self.window_length = lookback + decision_step

        self.k = n_asset
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        t = np.random.randint(self.window_length, self.T)
        asset_idx = np.random.choice(self.N, self.k, replace=False)
        window = self.returns[t - self.lookback:t, asset_idx]
        
        return {'input_r': torch.tensor(window), 'asset_idx': torch.tensor(asset_idx)}