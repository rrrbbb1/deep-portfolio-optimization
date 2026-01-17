import pandas as pd
import numpy as np

def load_df(data_path):
    stocks_df = pd.read_parquet(data_path)

    tickers = stocks_df.groupby('Ticker')['Close'].count().sort_values(ascending=False)
    tickers = tickers[tickers == max(tickers)].index.to_list()
    stocks_df = stocks_df[stocks_df['Ticker'].isin(tickers)]
    close_df = stocks_df.pivot(index='Date', columns='Ticker', values='Close')

    returns_df = close_df.pct_change()[1:]
    prices_df = close_df[1:]

    means_df = returns_df.rolling(60).mean()[59:]
    stds_df = returns_df.rolling(60).std()[59:]

    norm_returns_df = (returns_df[59:] - means_df) / stds_df

    returns_df = returns_df[59:]
    prices_df = prices_df[59:]

    df_map = {}

    df_map['prices'] = prices_df
    df_map['returns'] = returns_df

    df_map['norm_returns'] = norm_returns_df
    df_map['means'] = means_df
    df_map['stds'] = stds_df

    return df_map

def get_rnd_asset_list(data_path, k=10, seed=None):
    stocks_df = pd.read_parquet(data_path)

    tickers = stocks_df.groupby('Ticker')['Close'].count()
    tickers = tickers[tickers == tickers.max()].index.to_numpy()

    rng = np.random.default_rng(seed)
    return rng.choice(tickers, size=k, replace=False).tolist()