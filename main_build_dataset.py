import os
import pandas as pd
from io import StringIO
import requests
from src.data_prep import crea_dataset_clf

DATA_PATH = "data/processed/features_with_target.csv"

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(StringIO(requests.get(url).text))
    sp500 = tables[0]
    tickers = [t.replace(".", "-") for t in sp500["Symbol"].tolist()]
    return tickers

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        print(f"✅ Dataset già esistente in '{DATA_PATH}'. Salto creazione.")
    else:
        print("📥 Scarico tickers S&P500...")
        tickers = get_sp500_tickers()

        print("🔄 Creo il dataset...")
        df = crea_dataset_clf(tickers)

        print("📈 Aggiungo Rank features...")
        # Calcolo rank per ogni data
        df["Rank_Momentum_6m"] = df.groupby("date")["Momentum_6m"].rank(method="average")
        df["Rank_Volatility"] = df.groupby("date")["Volatility"].rank(method="average")

        print("✅ Rank features aggiunte e file aggiornato.")

        print(f"💾 Salvo dataset in '{DATA_PATH}'")
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

        print("✅ Done.")
