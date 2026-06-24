import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Fetch data
df = yf.Ticker("AAPL").history(period="2y")[['Close']].copy()

# Step 2: Create features
df['MA_5']  = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_20'] = df['Close'].rolling(20).mean()
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Step 3: Split
features = ['Close', 'MA_5', 'MA_10', 'MA_20', 'Lag_1', 'Lag_2', 'Lag_3']
X, y = df[features], df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print(f"MAE : ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")

# Step 6: Predict next day
next_day = model.predict(X.iloc[[-1]])[0]
print(f"\nLast close     : ${df['Close'].iloc[-1]:.2f}")
print(f"Predicted next : ${next_day:.2f}")

# Step 7: Plot
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='Actual', color='steelblue')
plt.plot(y_pred, label='Predicted', color='orange', linestyle='--')
plt.title("AAPL: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()
