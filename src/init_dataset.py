from torch.utils.data import DataLoader
from src.datasets.stochastic import StoFullDataset, StoSignalDataset, StoRawDataset
from src.datasets.deterministic import DetFullDataset, DetSignalDataset, DetRawDataset

def init_dataset(df_map, feature_mod: str, asset_list: list[str] = None, split_ratio = 0.8, batch_size: int = 256):
    train_df_map = {}
    test_df_map = {}

    for _, df in df_map.items():
        df_shape = df.shape
        break

    T = df.shape[0]
    N = df.shape[1]
    split_index = int(split_ratio * T)

    for key, df in df_map.items():
        assert df.shape == df_shape

        train_df_map[key] = df[:split_index]
        test_df_map[key] = df[split_index:]

    if asset_list is None:
        if feature_mod == 'all':
            print('Initializing with all features defined in the preprocessing pipeline (stochastic sampling)')
            train_dataset = StoFullDataset(
                df_map=train_df_map,
                n_asset=10,
                lookback=100,
                n_samples=1_000_000
            )
            test_dataset = StoFullDataset(
                df_map=test_df_map,
                n_asset=10,
                lookback=100,
                n_samples=50_000
            )
        
        elif feature_mod == 'raw':
            print('Initializing with raw features defined in the preprocessing pipeline (stochastic sampling)')
            train_dataset = StoRawDataset(
                df_map=train_df_map,
                n_asset=10,
                lookback=100,
                n_samples=1_000_000
            )
            test_dataset = StoRawDataset(
                df_map=test_df_map,
                n_asset=10,
                lookback=100,
                n_samples=50_000
            )
        
        elif feature_mod == 'signal':
            print('Initializing with signal features defined in the preprocessing pipeline (stochastic sampling)')
            train_dataset = StoSignalDataset(
                df_map=train_df_map,
                n_asset=10,
                lookback=100,
                n_samples=1_000_000
            )
            test_dataset = StoSignalDataset(
                df_map=test_df_map,
                n_asset=10,
                lookback=100,
                n_samples=50_000
            )
        


    elif asset_list is not None:
        for key, df in train_df_map.items():
            train_df_map[key] = df[asset_list]
        
        for key, df in test_df_map.items():
            test_df_map[key] = df[asset_list]

        if feature_mod == 'all':
            print('Initializing with all features defined in the preprocessing pipeline (deterministic sampling)')
            train_dataset = DetFullDataset(train_df_map)
            test_dataset = DetFullDataset(test_df_map)
        
        elif feature_mod == 'raw':
            print('Initializing with raw features defined in the preprocessing pipeline (deterministic sampling)')
            train_dataset = DetRawDataset(train_df_map)
            test_dataset = DetRawDataset(test_df_map)
        
        elif feature_mod == 'signal':
            print('Initializing with signal features defined in the preprocessing pipeline (deterministic sampling)')
            train_dataset = DetSignalDataset(train_df_map)
            test_dataset = DetSignalDataset(test_df_map)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
