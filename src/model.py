import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

def build_pipeline(model_type="rf"):
    """
    Costruisce una pipeline sklearn con normalizzazione e classificatore.
    Supporta: 'rf', 'logreg', 'xgb'.
    """
    if model_type == "rf":
        clf = RandomForestClassifier(random_state=42)
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif model_type == "xgb":
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError("Model type not supported")

    pipeline = Pipeline([
        ("scaler", QuantileTransformer(output_distribution="normal", random_state=42)),
        ("clf", clf)
    ])
    return pipeline

def plot_feature_importance(pipeline, X_columns, model_name="rf"):
    """
    Plotta le feature importance per RF/XGB o i coefficienti per LogReg.
    Salva il grafico in outputs/feature_importance_<model>.png
    """
    model = pipeline.named_steps["clf"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]  # LogReg
        importances = abs(importances)  # prende il modulo dei coefficienti
    else:
        print(f"⚠️ Il modello {model_name} non supporta interpretabilità.")
        return

    fi_df = pd.DataFrame({
        "Feature": X_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    plt.barh(fi_df["Feature"], fi_df["Importance"], color="mediumseagreen")
    plt.xlabel("Importanza (assoluta)")
    plt.title(f"Top 10 feature – {model_name.upper()}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"outputs/feature_importance_{model_name}.png")
    plt.show()

def train_and_evaluate(df, model_type):
    """
    Allena e valuta il modello selezionato, salva le predizioni e plottizza le feature importance se supportata.
    """
    X = df.drop(columns=["Label", "ticker", "date", "Return_6m", "Return_SP500_6m"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    pipeline = build_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("🧪 ROC AUC:", roc_auc_score(y_test, y_prob))
    print("🧮 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))                             

    df_out = pd.DataFrame({
        "true": y_test.values,
        "pred": y_pred,
        "prob": y_prob
    }, index=y_test.index)
    df_out.to_csv("outputs/predizioni.csv", index=False)

    plot_feature_importance(pipeline, X.columns, model_type)

    return pipeline