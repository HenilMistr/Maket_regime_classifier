# 🧠 Market Regime Classifier & Strategy Backtester

This project is a multi-stage financial analysis pipeline designed to classify market regimes and develop data-driven trading strategies. It includes technical indicator generation, machine learning classification, regime-aware strategy evaluation, and hyperparameter optimization using Optuna.

---

## 📈 Project Goals
Classify Market Regimes (Bull, Bear, Sideways) using technical indicators and trend logic.

Predict Price Movement with machine learning classifiers (Random Forest, XGBoost).

Backtest and Evaluate Strategies based on classification signals.

Optimize Strategy Parameters per regime using Optuna.

---

## 🧩 Project Structure
Level 1 – Trend-Based Regime Classification
Uses moving averages and volatility bands to label market regimes.

Visualizes classified regimes on price charts.

Level 2 – Strategy Evaluation
Backtests simple and indicator-based (RSI, MACD) strategies.

Calculates financial metrics: Sharpe Ratio, CAGR, Max Drawdown.

Compares performance across strategies and regimes.

Level 3 – Machine Learning Classification
Prepares technical features (Returns, MA, RSI, MACD, Momentum, Volatility).

Trains and evaluates:

RandomForestClassifier

XGBoostClassifier

Uses classification_report, confusion matrix, and accuracy metrics.

Predicts next-day price movement as Up, Down, or Flat.

Regime-Based Strategy Optimization (Bonus)
Uses Optuna to find optimal strategy parameters (e.g., threshold, indicators).

Optimizes based on Sharpe Ratio, CAGR, or other metrics.

Returns a best-fit strategy per regime type.

---

## 📊 Features

- ✅ Technical indicator calculation (MA, RSI, MACD, Momentum, Volatility)
- ✅ Regime classification (Bull, Bear, Sideways)
- ✅ Labeling short-term market direction (Up, Down, Flat)
- ✅ Machine learning models: Random Forest, XGBoost
- ✅ Regime-aware trading logic using RSI and MACD
- ✅ Backtesting with Sharpe Ratio, CAGR, Max Drawdown
- ✅ Regime strategy optimizer using Optuna

---

## 📈 Strategy Logic

Different regimes trigger different strategy signals:
- **Bull Market**: Buy when RSI > 50 or MACD histogram is positive
- **Bear Market**: Sell when RSI < 50 or MACD histogram is negative
- **Sideways Market**: No position

Backtest performance is compared to a baseline strategy.

## 📊 Key Metrics Reported
Sharpe Ratio

CAGR (Compound Annual Growth Rate)

Max Drawdown

Classification Accuracy

Confusion Matrix

---

## 📁 Output
Regime-labeled CSV: regime_classified_data.csv

Visual strategy performance comparisons
![image](https://github.com/user-attachments/assets/45e87526-a6cb-4eb2-b196-4302da2ede33)


Model predictions and evaluation reports

Optimized parameters per regime

---

## 🚀 Future Improvements
Add LSTM-based time-series model for regime prediction.

Include transaction cost modeling in backtests.

Deploy as a dashboard using Streamlit or Dash.

Add SHAP explainability for ML models.

---
# 🧠 Author
Henil Mistry
📧 Henil.mistry09@gmail.com
🌐 https://henilmistry.com



