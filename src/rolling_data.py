import pandas as pd
from src.model import build_pipeline
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def rolling_validation(df, model_type):
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce") # evito utc e forzo conversione
    df = df[df["date"].notna()].sort_values("date") # Droppo righe vuote

    anni_test = sorted(df["date"].dt.year.unique())
    anni_test = [y for y in anni_test if y > min(anni_test)]  # salta primo anno

    results = []

    for test_year in anni_test:
        print(f"\n🧪 Test su anno {test_year}")

        train_data = df[df["date"].dt.year < test_year]
        test_data = df[df["date"].dt.year == test_year]

        X_train = train_data.drop(columns=["Label", "ticker", "date"])
        y_train = train_data["Label"]

        X_test = test_data.drop(columns=["Label", "ticker", "date"])
        y_test = test_data["Label"]

        pipeline = build_pipeline(model_type)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"ROC AUC: {auc:.4f}")

        results.append({
            "year": test_year,
            "auc": auc,
            "f1_class1": report["1"]["f1-score"],
            "precision_class1": report["1"]["precision"],
            "recall_class1": report["1"]["recall"]
        })

    return pd.DataFrame(results)

