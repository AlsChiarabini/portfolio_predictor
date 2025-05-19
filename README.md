# ML Portfolio Predictor

Un progetto di classificazione per prevedere se un titolo dell'S&P500 batterà il mercato nei prossimi 6 mesi, usando tecniche di Machine Learning supervisionato.

Il modello ha performance molto elevate (ROC AUC > 0.98), ma va notato che il dataset copre solo 3 anni, quindi la validazione rolling è limitata. I risultati sono indicativi della stabilità del modello, ma richiederebbero dati su orizzonti temporali più lunghi per una valutazione robusta. Ulteriori test con modelli più semplici (es. Logistic Regression) saranno inclusi come benchmark.

Dopo l’aggiunta di tre feature derivate (Momentum_vs_Volatility, Rank_Momentum_6m, Rank_Volatility), tutti i modelli testati hanno mostrato un miglioramento consistente:

    AUC media: da ~0.61 a ~0.64

    F1-score: da 0.56–0.57 a 0.59–0.60

Questo conferma che la componente informativa relativa (ranking tra titoli) e il rapporto rischio/rendimento migliorano la capacità del modello di distinguere titoli che batteranno l’S&P500.

Per ogni anno nel periodo di test, vengono salvate le 20 aziende con maggiore probabilità di outperform secondo ciascun modello. Questo consente di osservare la coerenza e l’evoluzione delle previsioni nel tempo, evitando distorsioni legate al test set fisso.

---

## 📌 Obiettivi

- Predire la probabilità che un'azione sovraperformi l'S&P500
- Testare modelli come Random Forest, Logistic Regression, XGBoost
- Valutare le performance in rolling validation temporale
- Esportare le predizioni per analisi in Power BI

---

## 🧠 Dataset

- Dati scaricati da Yahoo Finance (`yfinance`)
- Periodo: 2021–2024
- Label: `1` se l'azione ha avuto un rendimento maggiore dell'S&P500 nei successivi 6 mesi, altrimenti `0`

---

## ⚙️ Struttura progetto

ml_portfolio_predictor/
│
├── data/ # Dati grezzi e processati
├── notebooks/ # Analisi esplorativa e test modello
├── outputs/ # CSV con predizioni, grafici, Power BI
├── src/ # Moduli Python riusabili
│ ├── data_prep.py
│ ├── model.py
│ └── utils.py
├── main_build_dataset.py # Crea il dataset
├── main_train.py # Allena modello base
├── main_rolling_validation.py # Rolling validation per anno
├── requirements.txt
└── README.md


---

## 🚀 Come eseguire

### 1. Crea ambiente virtuale (consigliato)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt

2. Crea il dataset

python main_build_dataset.py

3. Allena il modello base

python main_train.py

4. Esegui rolling validation

python main_rolling_validation.py

📊 Output

    outputs/predizioni.csv → predizioni su test set

    outputs/rolling_metrics.csv → performance per anno

    outputs/rolling_metrics.png → grafico metrica temporale

🧠 Modelli supportati

    rf: Random Forest

    logreg: Logistic Regression

    xgb: XGBoost

🧩 Estensioni previste

    Dashboard interattiva con Power BI

    API predizione singola (FastAPI)

    Versione regression-only

    Deploy su Streamlit

👤 Autore

Alessandro Chiarabini
LinkedIn | GitHub 