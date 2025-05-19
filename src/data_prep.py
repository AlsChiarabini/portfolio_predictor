import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

def crea_dataset_clf(tickers, start="2021-01-01", end="2024-01-01"):
    dataset = []
    benchmark = yf.Ticker("SPY").history(start=start, end=end)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)

            if len(hist) < 252:
                continue

            info = stock.info
            hist = hist.dropna()

            for i in range(len(hist) - 126):
                t0 = hist.index[i]
                t1 = hist.index[i + 126]

                # Salta se SPY non ha dati per lo stesso intervallo
                if t1 not in benchmark.index or t0 not in benchmark.index:
                    continue

                future_close = hist.loc[t1]["Close"]
                sp500_return = (benchmark.loc[t1]["Close"] / benchmark.loc[t0]["Close"]) - 1
                stock_return = (future_close / hist.loc[t0]["Close"]) - 1

                momentum = (hist.loc[t0]["Close"] / hist.iloc[max(0, i-126)]["Close"]) - 1 if i >= 126 else np.nan
                volatility = hist["Close"].iloc[max(0, i-126):i].pct_change().std() * np.sqrt(252)

                row = {
                    "ticker": ticker,
                    "date": t0,
                    "PE": info.get("trailingPE", np.nan),
                    "PB": info.get("priceToBook", np.nan),
                    "ROE": info.get("returnOnEquity", np.nan),
                    "ROA": info.get("returnOnAssets", np.nan),
                    "DebtToEquity": info.get("debtToEquity", np.nan),
                    "Beta": info.get("beta", np.nan),
                    "MarketCap": info.get("marketCap", np.nan),
                    "DividendYield": info.get("dividendYield", np.nan),
                    "Volatility": volatility,
                    "Momentum_6m": momentum,
                    "Return_6m": stock_return,
                    "Return_SP500_6m": sp500_return,
                    "Label": int(stock_return > sp500_return),
                    #Da qua, aggiungo 3 features tramite feature engineering, una qua e 2 successivamente
                    "Mom_vs_Vol": momentum / volatility if volatility != 0 else np.nan,
                }

                dataset.append(row)

        except Exception as e:
            print(f"Errore con {ticker}: {e}")
            continue

    df = pd.DataFrame(dataset)
    df.dropna(inplace=True)

    return df
