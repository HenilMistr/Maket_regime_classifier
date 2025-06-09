import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

ticker = '^GSPC'

today = datetime.datetime.today().strftime('%Y-%m-%d')

# Download the data
df = yf.download(ticker, start="2024-01-01", end=today,group_by='column') #group is optional
df.columns = df.columns.get_level_values(0) #optional


# Show the first few rows
print(df.head())

# Plot the closing price
df['Close'].plot(title = f"{ticker} Closing Prices", figsize=(12, 6))
# plt.show()

# This calculates the Features
df['Return'] = df['Close'].pct_change()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['Volatility'] = df['Return'].rolling(window=21).std()
df['Momentum'] = df['Close'] - df['Close'].shift(10)


df = df.dropna()
print(df[['MA20', 'MA50', 'Close']].isnull().sum())
df = df.dropna(subset=['MA20', 'MA50', 'Close'])

#part of level 2
df['Future_5D_Return'] = df['Close'].shift(-5) / df['Close'] - 1
df['Future_10D_Return'] = df['Close'].shift(-10) / df['Close'] - 1

# Optional classification labels (for supervised classification)
def classify_return(r):
    if r > 0.01:
        return 'Up'
    elif r < -0.01:
        return 'Down'
    else:
        return 'Flat'

df['Future_5D_Label'] = df['Future_5D_Return'].apply(classify_return)
df['Future_10D_Label'] = df['Future_10D_Return'].apply(classify_return)
#end


# Show and plot features
print(df[['Close', 'Future_5D_Return', 'Future_10D_Return', 'Future_5D_Label', 'Future_10D_Label']].tail(15))

# Plot closing price with moving averages
df[['Close', 'MA50', 'MA20']].plot(figsize=(12, 6), title = f"{ticker} with Moving Averages")
# plt.show()

# Create a regime label
def classify_regime(row):
    # print(row[['MA20','MA50','Close']]) #Debugo
    ma_diff = row['MA20'] - row['MA50']
    threshold = 0.01 * row['Close']  # 2% of MA_200
    
    if ma_diff > threshold:
        return "Bull"
    elif ma_diff < -threshold:
        return "Bear"
    else:
        return "Sideways"

df['Regime'] = df.apply(classify_regime, axis=1)

# Print a few rows to check for dubing
# print(df[['Close', 'MA50', 'MA20', 'Regime']].tail())

# sns.set(style="darkgrid") # this makes the plot dark

# Plot with color by regime
plt.figure(figsize=(14, 6))

sns.lineplot(data=df, x=df.index, y='Close', hue='Regime', palette={'Bull': 'green', 'Bear': 'red', 'Sideways': 'gray'})
# Loop through each regime
for regime, color in zip():
    subset = df[df['Regime'] == regime]
    plt.plot(subset.index, subset['Close'], label=regime, color=color)

plt.title(f'{ticker} Closing Prices with Regime Classification')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.savefig(f'{ticker}_regime_chart.png')
# plt.show()
print("chart saved as regime_chart.png")

# print(df.columns) #debug statement


# Create feature and target datasets
df['MA_Diff'] = df['MA20'] - df['MA50']

features = df[['Return', 'Volatility', 'Momentum', 'MA_Diff']]
target = df['Regime']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

#. Feature scaling (important for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

#. Make predictions
y_pred = clf.predict(X_test_scaled)

# Evaluate
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# Strategy return: Invest only in Bull regime
df['Strategy_Return'] = df['Return'] * (df['Regime'] == 'Bull').astype(int)

# Cumulative returns
df['Market_Cumulative'] = (1 + df['Return']).cumprod()
df['Strategy_Cumulative'] = (1 + df['Strategy_Return']).cumprod()

# Plot performance
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Market_Cumulative'], label='Market', color='blue')
plt.plot(df.index, df['Strategy_Cumulative'], label='Regime-Based Strategy', color='green')
plt.title('Market vs. Regime-Based Strategy Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.tight_layout()
plt.savefig(f"{ticker}_strategy_vs_market.png")
# plt.show()

# RSI Calculation
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = compute_rsi(df['Close'])

# MACD Calculation
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Default no signal
df['Signal'] = 0

# Long entry condition in Bull market
df.loc[
    (df['Regime'] == 'Bull') & 
    (df['RSI'] < 70) & 
    (df['MACD'] > df['MACD_Signal']),
    'Signal'
] = 1

# Shift signal to avoid lookahead bias
df['Position'] = df['Signal'].shift(1)

# Calculate strategy return
df['Strategy_Return'] = df['Position'] * df['Return']

# Re-calculate cumulative returns
df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
df['Cumulative_Market'] = (1 + df['Return']).cumprod()



plt.figure(figsize=(14, 6))
plt.plot(df['Cumulative_Market'], label='Market Return', color='black')
plt.plot(df['Cumulative_Strategy'], label='Strategy Return', color='blue')
plt.title('Market vs Strategy Performance (RSI + MACD Logic)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{ticker}_strategy_performance.png")
# plt.show()


# Daily returns
daily_returns = df['Strategy_Return']
market_returns = df['Return']

# 252 trading days in a year
trading_days = 252

# CAGR
strategy_cagr = (df['Cumulative_Strategy'].iloc[-1])**(1/((df.index[-1] - df.index[0]).days / 365)) - 1
market_cagr = (df['Cumulative_Market'].iloc[-1])**(1/((df.index[-1] - df.index[0]).days / 365)) - 1

# Volatility calcuation
strategy_volatility = np.std(daily_returns) * np.sqrt(trading_days)
market_volatility = np.std(market_returns) * np.sqrt(trading_days)

# Sharpe Ratio
sharpe_ratio = strategy_cagr / strategy_volatility

# Max Drawdown
roll_max = df['Cumulative_Strategy'].cummax()
drawdown = df['Cumulative_Strategy']/roll_max - 1.0
max_drawdown = drawdown.min()

# Print metrics
print("Strategy Evaluation:")
print(f"CAGR: {strategy_cagr:.2%}")
print(f"Volatility: {strategy_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

df.to_csv(f'{ticker}_regime_classified_data.csv')
print("data saved to regime_classified_data.csd")

#level 2 stuff

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb


# Select features and target
features = ['Return', 'MA20', 'MA50', 'Volatility', 'Momentum']
target = 'Future_5D_Label'

# Drop rows with missing values in features or target
df_model = df.dropna(subset=features + [target])

label_mapping = {'Down': 0, 'Flat':1, 'Up':2 }
X = df_model[features]
y = df_model[target].map(label_mapping)

# Map string labels to integers
label_mapping = {'Down': 0, 'Flat': 1, 'Up': 2}
y = df_model[target].map(label_mapping)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#level 3

import pandas as pd
import numpy as np
import optuna

# Load your data (ensure 'Close' column exists)
df = pd.read_csv(f"{ticker}_regime_classified_data.csv")

# Calculate returns for volatility and strategy later
df['Return'] = df['Close'].pct_change()

def objective(trial):
    # Hyperparameters
    ma_short = trial.suggest_int("ma_short", 10, 30)
    ma_long = trial.suggest_int("ma_long", ma_short + 20, 200)
    threshold_pct = trial.suggest_float("threshold", 0.001, 0.03)

    if ma_short >= ma_long:
        return -np.inf

    data = df.copy()

    # Calculate technical indicators
    data['MA_short'] = data['Close'].rolling(window=ma_short).mean()
    data['MA_long'] = data['Close'].rolling(window=ma_long).mean()
    data['Volatility'] = data['Return'].rolling(window=21).std()
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data.dropna(inplace=True)

    # Classify market regime
    def classify(row):
        ma_diff = row['MA_short'] - row['MA_long']
        threshold = threshold_pct * row['Close']
        if ma_diff > threshold:
            return "Bull"
        elif ma_diff < -threshold:
            return "Bear"
        else:
            return "Sideways"

    data['Regime'] = data.apply(classify, axis=1)

    # Simulate a basic strategy: invest in Bull regime only
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = 0.0
    data.loc[data['Regime'].shift(1) == 'Bull', 'Strategy_Return'] = data['Market_Return']
    data.dropna(inplace=True)

    # Sharpe Ratio (annualized)
    strategy_std = data['Strategy_Return'].std()
    if strategy_std and not np.isnan(strategy_std):
        sharpe = (data['Strategy_Return'].mean() / strategy_std) * np.sqrt(252)
    else:
        sharpe = -np.inf

    return sharpe

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Output best hyperparameters
print("Best hyperparameters:", study.best_params)

print("Best trial:")
trial = study.best_trial

print(f"value: {trial.value}")
print(f"params: {trial.params}")