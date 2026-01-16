import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PortfolioDataset(Dataset):
    def __init__(
            self, prices_df: pd.DataFrame, returns_df: pd.DataFrame, norm_returns_df: pd.DataFrame,
            n_asset: int = 10, lookback: int = 60, n_samples: int = 100_000
        ):

        self.prices = prices_df.values.astype(np.float32)
        self.returns = returns_df.values.astype(np.float32)
        self.inputs = norm_returns_df.values.astype(np.float32)

        assert self.returns.shape == self.prices.shape
        assert self.returns.shape == self.inputs.shape

        self.T, self.N = self.returns.shape

        self.lookback = lookback
        self.k = n_asset
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        t = np.random.randint(self.lookback, self.T)
        asset_idx = np.random.choice(self.N, self.k, replace=False)

        inputs = self.inputs[t - self.lookback:t, asset_idx]
        prices = self.prices[t - self.lookback:t, asset_idx]

        window = np.concatenate((prices, inputs), axis=1)

        returns = self.returns[t - self.lookback:t, asset_idx]
        
        return {
            'input': torch.tensor(window),
            'returns': torch.tensor(returns),
            'asset_idx': torch.tensor(asset_idx)
            }