# DL-homwork
## LSTM Stock Predictor Using Fear and Greed Index
### Data Preparation
```
import numpy as np
import pandas as pd
import hvplot.pandas
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
```
### Data load
```
df = pd.read_csv('btc_sentiment.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
df = df.drop(columns="fng_classification")
df.head()
```
pic bitcoin fng value
```
df2 = pd.read_csv('btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
df2 = df2.sort_index()
df2.tail()
```
pic of close bitcoin
```
df = df.join(df2, how="inner")
df.tail()
```
pic join fng and close

### X and Y 
```
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)
```

### Predict Closing Prices using a 10 day window of previous fng values
```
window_size = 10

# Column index 0 is the 'fng_value' column
# Column index 1 is the `Close` column
feature_column = 0
target_column = 1
X, y = window_data(df, window_size, feature_column, target_column)
```

### Use 70% of the data for training and the remaineder for testing
```
split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]
```

### Use the MinMaxScaler to scale data between 0 and 1.
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
```

### Reshape the features for the model
```
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print (f"X_train sample values:\n{X_train[:5]} \n")
print (f"X_test sample values:\n{X_test[:5]}")
```
pic of X_train sample

## Build and Train the LSTM RNN

### initial import
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```

### Build the LSTM model
```
model = Sequential()

number_units = 5
dropout_fraction = 0.2

# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1))
# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")
# Summarize the model
model.summary()
```
pic of summary

### Train the model
```
model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=1, verbose=1)
```
pic of train model

### model performance
```
model.evaluate(X_test, y_test)
```
pic of evaluate of model performance

```
predicted = model.predict(X_test)
```
pic of predict model

### Create a DataFrame of Real and Predicted values
```
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
stocks.head()
```
pic of real and predicted values
```
stocks.plot()
```
pic of stocks plots

## LSTM Stock Predictor Using Closing Prices
### Data Preparation
```
import numpy as np
import pandas as pd
import hvplot.pandas
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
```
### load data
```
df = pd.read_csv('btc_sentiment.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
df = df.drop(columns="fng_classification")
df.head()
```
pic of fng
```
df2 = pd.read_csv('btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
df2 = df2.sort_index()
df2.tail()
```
pic of close
```
df = df.join(df2, how="inner")
df.tail()
```
pic of join

### X and Y
```
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)
```
### Predict Closing Prices using a 10 day window of previous closing prices
```
window_size = 10

# Column index 0 is the 'fng_value' column
# Column index 1 is the `Close` column
feature_column = 1
target_column = 1
X, y = window_data(df, window_size, feature_column, target_column)
```

### Use 70% of the data for training and the remaineder for testing
```
split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]
```
### Use the MinMaxScaler to scale data between 0 and 1.
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
```

### Reshape the features for the model
```
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print (f"X_train sample values:\n{X_train[:5]} \n")
print (f"X_test sample values:\n{X_test[:5]}")
```
pic of reshape

## Build and Train the LSTM RNN
### initial import
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```
### Build the LSTM model. 
```
model = Sequential()

number_units = 5
dropout_fraction = 0.2

# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()
```
pic of summary

### Train the model
```
model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=1, verbose=1)
```
pic of train the model

## Model Performance
### Evaluate the model
```
model.evaluate(X_test, y_test)
```
pic of eva the model
###  Make some predictions
```
predicted = model.predict(X_test)
```
pic of prediction
###  Recover the original prices instead of the scaled version
```
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
```
### Create a DataFrame of Real and Predicted values
```
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
stocks.head()
```
pic of stock
```
stocks.plot()
```
pic of plot
