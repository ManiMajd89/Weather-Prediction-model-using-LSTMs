# UTRA Robotics Team Project
# Perception Subteam, December 2024

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as sk

# Load the data
df = pd.read_csv('FullWeatherData_Toronto.csv')

# Extract input columns and output columns
input_cols = df.columns[:-4]
output_cols = []

for c in df.columns:
    if c.isdigit():
        output_cols.append(c)

# Convert the dataframe to a numpy array
timeseries = df[input_cols].values.astype('float32')

print(timeseries.shape)

# Train-test split for time series
train_size = int(len(timeseries) * 0.6)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

# Function to create dataset for multi-time step prediction
def create_dataset(dataset, lookback, n_steps, total_features, output_features):
    X, y = [], []
    total_features_list = list(total_features)
    output_indices = [total_features_list.index(feature) for feature in output_features]
    for i in range(len(dataset) - lookback - n_steps + 1):
        X.append(dataset[i:i + lookback, :])
        y.append(dataset[i + lookback:i + lookback + n_steps, output_indices])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

# Parameters for dataset creation
lookback = 10
n_steps = 6  # Number of future time steps to predict

# Create training and test datasets
X_train, y_train = create_dataset(train, lookback=lookback, n_steps=n_steps, total_features=input_cols, output_features=output_cols)
X_test, y_test = create_dataset(test, lookback=lookback, n_steps=n_steps, total_features=input_cols, output_features=output_cols)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Define the LSTM model
class AirModel(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz, n_steps, dropout_prob=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_sz, hidden_size=hidden_sz, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(hidden_sz, output_sz * n_steps)
        self.output_sz = output_sz
        self.n_steps = n_steps

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Use the output of the last time step
        x = self.linear(x)
        x = torch.relu(x)
        x = x.view(-1, self.n_steps, self.output_sz)  # Ensure correct output shape
        return x

# Model parameters
input_size = len(input_cols)
output_size = len(output_cols)
model = AirModel(input_size, hidden_sz=50, output_sz=output_size, n_steps=n_steps, dropout_prob=0.1)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=32)

print("output size:", output_size)
print("Train samples", train_size)

# Training loop
n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_loss = loss_fn(y_pred_train, y_train)
            train_rmse = torch.sqrt(train_loss)
            train_mae = torch.mean(torch.abs(y_pred_train - y_train))
            train_r2 = sk.r2_score(y_train.numpy().flatten(), y_pred_train.numpy().flatten())

            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_rmse = torch.sqrt(test_loss)
            test_mae = torch.mean(torch.abs(y_pred_test - y_test))
            test_r2 = sk.r2_score(y_test.numpy().flatten(), y_pred_test.numpy().flatten())

            print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, MAE {train_mae:.4f}, R^2 {train_r2:.4f} | Test RMSE {test_rmse:.4f}, MAE {test_mae:.4f}, R^2 {test_r2:.4f}")

# Plotting
with torch.no_grad():
    # Extract predictions
    y_pred_train = model(X_train).numpy()
    y_pred_test = model(X_test).numpy()

    # Plot both train and test predictions on the same plot for each output variable
    for i, col in enumerate(output_cols):
        plt.figure(figsize=(30, 6))
        plt.plot(timeseries[:, i], label='Actual')  # Plot the actual data for the current output variable

        # Shift train predictions for plotting
        train_plot = np.ones_like(timeseries[:, i]) * np.nan
        for j in range(y_pred_train.shape[0]):
            train_plot[lookback + j:lookback + j + n_steps] = y_pred_train[j, :, i]
        plt.plot(train_plot, c='r', label='Train Predictions')

        # Shift test predictions for plotting
        test_plot = np.ones_like(timeseries[:, i]) * np.nan
        for j in range(y_pred_test.shape[0]):
            test_plot[train_size + lookback + j:train_size + lookback + j + n_steps] = y_pred_test[j, :, i]
        plt.plot(test_plot, c='g', label='Test Predictions')

        plt.title(col)
        plt.legend()
        plt.tight_layout()
        plt.show()

