# Customer Churn Prediction

Machine-learning system that predicts which telecom customers are likely to leave (`1` = churn, `0` = stay),
built on the IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

## Features (functional requirements)
| ID | Requirement | Where |
|----|-------------|-------|
| FR-01 | Upload dataset | Streamlit page 1, `preprocessing.load_data` |
| FR-02 | Clean missing values | `preprocessing.clean_data` |
| FR-03 | Exploratory data analysis | `src/eda.py`, notebook, Streamlit page 2 |
| FR-04 | Feature engineering | `preprocessing.engineer_features` |
| FR-05 | Train multiple models | `src/train.py` (LogReg, Decision Tree, Random Forest, Gradient Boosting) |
| FR-06 | Compare models | `reports/model_comparison.csv`, Streamlit pages 4 & 6 |
| FR-07 | Predict churn | `src/predict.py`, Streamlit page 5 |
| FR-08 | Probabilities | `predict_customer` / probability bar in the app |
| FR-09 | Visual reports | `reports/figures/*.png`, `reports/model_report.md` |
| FR-10 | Save model | `models/churn_model.pkl` |

## Quick start
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# place the Kaggle CSV at data/customer_data.csv
python -m src.train          # trains, evaluates, saves model + reports
python -m src.predict        # demo prediction
streamlit run app/app.py     # dashboard
python -m pytest -q tests    # unit tests
```

## Methodology
* **Leak-free pipeline**: feature engineering -> imputation/scaling/one-hot -> model in a single sklearn `Pipeline`;
  the saved model accepts raw customer columns.
* **Reproducible**: fixed `random_state=42`, stratified 80/20 split, 5-fold stratified cross-validation.
* **Business-driven selection**: churners are only ~27% of customers, so accuracy is misleading. The decision threshold is
  tuned on out-of-fold training predictions to maximise F1.5 (recall weighted above precision, because losing a customer
  costs more than a wasted retention offer). The model is **chosen on cross-validation only**; the test set is used once.
* Change the cost trade-off with `BETA` in `src/config.py`.

## Project structure
```
data/customer_data.csv        src/config.py, preprocessing.py, eda.py, train.py, evaluate.py, predict.py
models/churn_model.pkl        app/app.py          reports/model_report.md, figures/
notebooks/eda.ipynb           tests/test_pipeline.py
```
