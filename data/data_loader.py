import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm


def fetch_stock_data(ticker_symbol: str, period: str = "1d") -> pd.DataFrame:
    # 1. Define the ticker and the data directory
    output_dir = Path("data")
    output_path = output_dir / "mock_prices.csv"

    # Ensure the directory exists
    output_dir.mkdir(exist_ok=True)

    print(f"Fetching historical data for {ticker_symbol}...")
    try:
        # 2. Fetch 5 years of historical data
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)

        if df.empty:
            raise ValueError(f"No data returned for {ticker_symbol}")
        else:
            # 3. Save it to a CSV locally
            df.to_csv(output_path)

            # 4. Print some summary stats for verification
            print("Data downloaded and saved successfully.")
            print(f"Rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            print(f"Date Range: {df.index.min()} to {df.index.max()}")
            print(f"Data saved to: {output_path.resolve()}")
            return df

    except Exception as e:
        print(f"An error occurred: {e}")

