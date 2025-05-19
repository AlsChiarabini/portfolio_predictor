import pandas as pd
import os
from src.model import train_and_evaluate
from src.rolling_data import rolling_validation

models = ["rf", "xgb", "logreg"]

if __name__ == "__main__":
    print("📂 Carico dataset...")
    df = pd.read_csv("data/processed/features_with_target.csv")

    model = input(f"Che modello usare tra {models}? ").strip().lower()
    if model not in models:
        raise ValueError(f"Modello non valido, usa quelli validi --> {models}")
    
    print(f"🚀 Alleno modello --> {model}...")
    pipeline = train_and_evaluate(df, model_type=model)

    print(f"📊 Genero ranking top 20 per anno...")
    df_pred = pd.read_csv("outputs/predizioni.csv")
    df_full = pd.read_csv("data/processed/features_with_target.csv")

    # Aggiungi info mancanti
    df_pred["ticker"] = df_full.loc[df_pred.index, "ticker"].values
    df_pred["date"] = pd.to_datetime(df_full.loc[df_pred.index, "date"], errors="coerce", utc=True)
    df_pred = df_pred[df_pred["date"].notna()]

    # Filtro: solo predizioni corrette positive
    successi_veri = df_pred[(df_pred["pred"] == 1) & (df_pred["true"] == 1)]

    # Genera top 20 per ogni anno
    for year in successi_veri["date"].dt.year.unique():
        subset = successi_veri[successi_veri["date"].dt.year == year]
        top20 = subset.sort_values("prob", ascending=False).head(20)
        top20.to_csv(f"outputs/top20_successi_{year}_{model}.csv", index=False)

    print("✅ File top20_successi_<anno>_<model>.csv salvati.")

    scelta = input("Vuoi fare anche un'analisi di rolling validation? (y/n) ")
    if scelta == 'y':
        print(f"🚀 Rolling validation sul modello --> {model}...")
        results = rolling_validation(df, model_type=model)
        print(results)

        os.makedirs("outputs", exist_ok=True)
        results.to_csv(f"outputs/rolling_metrics_{model}.csv", index=False)
    elif scelta == 'n':
        pass
    else:
        raise ValueError(f"Scelta non valida, scegliere y o n")

    
