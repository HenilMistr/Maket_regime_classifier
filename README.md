# 🧠 Market Regime Classifier & Strategy Backtester

This project is a multi-stage financial analysis pipeline designed to classify market regimes and develop data-driven trading strategies. It includes technical indicator generation, machine learning classification, regime-aware strategy evaluation, and hyperparameter optimization using Optuna.

## 🚀 Project Overview

This repo implements a 3-level system:

- **Level 1**: Feature engineering and basic regime classification (Bull, Bear, Sideways) using technical indicators (MA, Volatility, RSI, MACD).
- **Level 2**: Machine learning classification (Random Forest, XGBoost) for short-term price direction using regime-labeled data and technical features.
- **Level 3**: Regime-specific strategy evaluation using RSI/MACD signals and regime optimization using Optuna.

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

## 🧠 Machine Learning

### Models Used
- **Random Forest**
- **XGBoost**

### Features
- Lagged returns
- Momentum
- Volatility
- RSI
- MACD
- Regime label

---

## 📈 Strategy Logic

Different regimes trigger different strategy signals:
- **Bull Market**: Buy when RSI > 50 or MACD histogram is positive
- **Bear Market**: Sell when RSI < 50 or MACD histogram is negative
- **Sideways Market**: No position

Backtest performance is compared to a baseline strategy.


