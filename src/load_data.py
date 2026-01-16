import pandas as pd

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

    return prices_df, returns_df, norm_returns_df