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

    
