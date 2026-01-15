import pandas as pd

def load_df(data_path):
    stocks_df = pd.read_parquet(data_path)

    tickers = stocks_df.groupby('Ticker')['Close'].count().sort_values(ascending=False)
    tickers = tickers[tickers == max(tickers)].index.to_list()
    stocks_df = stocks_df[stocks_df['Ticker'].isin(tickers)]
    close_df = stocks_df.pivot(index='Date', columns='Ticker', values='Close')

    returns_df = close_df.pct_change()[1:]

    return returns_df