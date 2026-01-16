from torch.utils.data import DataLoader
from src.datasets.stochastic import StoFullDataset, StoSignalDataset, StoRawDataset
from src.datasets.deterministic import DetFullDataset, DetSignalDataset, DetRawDataset

def init_dataset(df_map, feature_mod: str, asset_list: list[str] = None, split_ratio=0.8, batch_size: int = 256):
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
            train_dataset = StoFullDataset(train_df_map)
            test_dataset = StoFullDataset(test_df_map)
        
        elif feature_mod == 'raw':
            train_dataset = StoRawDataset(train_df_map)
            test_dataset = StoRawDataset(test_df_map)
        
        elif feature_mod == 'signal':
            train_dataset = StoSignalDataset(train_df_map)
            test_dataset = StoSignalDataset(test_df_map)


    elif asset_list is not None:
        if feature_mod == 'all':
            train_dataset = DetFullDataset(train_df_map)
            test_dataset = DetFullDataset(test_df_map)
        
        elif feature_mod == 'raw':
            train_dataset = DetRawDataset(train_df_map)
            test_dataset = DetRawDataset(test_df_map)
        
        elif feature_mod == 'signal':
            train_dataset = DetSignalDataset(train_df_map)
            test_dataset = DetSignalDataset(test_df_map)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
