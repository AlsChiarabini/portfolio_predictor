import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["rf", "logreg", "xgb"]
METRICS = ["f1_class1", "auc", "precision_class1", "recall_class1"]

def load_results(model):
    path = f"outputs/rolling_metrics_{model}.csv"
    df = pd.read_csv(path)
    df["model"] = model.upper()
    return df

def plot_comparison(dfs, metric):
    plt.figure(figsize=(10, 6))
    for df in dfs:
        plt.plot(df["year"], df[metric], marker='o', label=df["model"].iloc[0])
    plt.xlabel("Anno")
    plt.ylabel(metric)
    plt.title(f"Confronto modelli – {metric}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/comparison_{metric}.png")
    plt.show()

if __name__ == "__main__":
    dfs = []
    for model in MODELS:
        try:
            df = load_results(model)
            dfs.append(df)
        except FileNotFoundError:
            print(f"⚠️ File mancante per modello: {model.upper()}. Salta.")
    if dfs:
        metric = input(f"Scegli una metrica di comparazione, scegliendo tra: {METRICS}. ")
        if metric not in METRICS:
            raise ValueError(f"Metrica non valida, usa quelle valide --> {METRICS}")
        plot_comparison(dfs, metric)
    else:
        print("❌ Nessun risultato disponibile.")
