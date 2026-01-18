import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
            self, df_map,
            lookback: int = 60
        ):

        self.returns = df_map['returns'].values.astype(np.float32)
        self.prices = df_map['prices'].values.astype(np.float32)

        self.norm_returns = df_map['norm_returns'].values.astype(np.float32)
        self.means = df_map['means'].values.astype(np.float32)
        self.stds = df_map['stds'].values.astype(np.float32)

        self.T, self.N = self.returns.shape

        self.lookback = lookback
        self.k = self.N

    def __len__(self):
        return self.T - self.lookback


class DetFullDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ts_dim = 5
    
    def __getitem__(self, idx):
        t = idx + self.lookback

        prices = self.prices[t - self.lookback:t, :]
        returns = self.returns[t - self.lookback:t, :]
        norm_returns = self.norm_returns[t - self.lookback:t, :]
        means = self.means[t - self.lookback:t, :]
        stds = self.stds[t - self.lookback:t, :]

        inputs = np.concatenate((prices, returns, norm_returns, means, stds), axis=1)
        returns = self.returns[t - self.lookback:t, :]
        
        return {
            'input': torch.from_numpy(inputs),
            'returns': torch.from_numpy(returns)
            }

class DetSignalDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ts_dim = 3
    
    def __getitem__(self, idx):
        t = idx + self.lookback

        norm_returns = self.norm_returns[t - self.lookback:t, :]
        means = self.means[t - self.lookback:t, :]
        stds = self.stds[t - self.lookback:t, :]

        inputs = np.concatenate((norm_returns, means, stds), axis=1)
        returns = self.returns[t - self.lookback:t, :]
        
        return {
            'input': torch.from_numpy(inputs),
            'returns': torch.from_numpy(returns)
            }

class DetRawDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ts_dim = 2
    
    def __getitem__(self, idx):
        t = idx + self.lookback

        prices = self.prices[t - self.lookback:t, :]
        returns = self.returns[t - self.lookback:t, :]

        inputs = np.concatenate((prices, returns), axis=1)
        returns = self.returns[t - self.lookback:t, :]
        
        return {
            'input': torch.from_numpy(inputs),
            'returns': torch.from_numpy(returns)
            }