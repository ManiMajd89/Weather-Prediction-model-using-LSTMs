# Weather-Prediction-Model-using-LSTMs
This code implements a multi-time-step weather prediction model using LSTMs (Long Short-Term Memory Networks). It preprocesses weather data, creates time-series datasets, trains an LSTM-based neural network to predict future weather conditions.


## Features
- **Data Preprocessing**:
  - Loads and processes weather data from a CSV file.
  - Splits data into training and testing sets for time series modeling.
  - Converts data into sequences for multi-step prediction.

- **Model**:
  - Implements a multi-layer LSTM neural network with dropout and fully connected layers.
  - Supports multi-output predictions over future time steps.

- **Evaluation**:
  - Uses RMSE, MAE, and R² metrics to evaluate the model on training and testing data.
  - Plots actual vs. predicted values for clear visualization of results.

- **Visualization**:
  - Displays prediction performance for each output variable using Matplotlib.

## Dataset
The project uses a weather dataset (`FullWeatherData_Toronto.csv`) containing multiple input features (e.g., temperature, humidity) and numerical outputs for prediction.

## Requirements
- Python 3.7+
- Required libraries:
  - `numpy`
  - `pandas`
  - `torch`
  - `scikit-learn`
  - `matplotlib`

Install dependencies using:
```bash
pip install numpy
pip install pandas
pip install torch
pip install scikit-learn
pip install matplotlib
```

## Usage
### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/time-series-weather-prediction.git
   cd time-series-weather-prediction
   ```
2. Place the `FullWeatherData_Toronto.csv` file in the project directory.
3. Run the script:
   ```bash
   python weather_prediction.py
   ```

### Model Parameters
- `lookback`: Number of past time steps used as input.
- `n_steps`: Number of future time steps to predict.
- `hidden_sz`: Number of hidden units in the LSTM layers.
- `dropout_prob`: Dropout probability for regularization.
- `n_epochs`: Number of training epochs.
- `batch_size`: Batch size for training.

## Outputs
1. **Model Performance**:
   - Logs metrics (RMSE, MAE, R²) for training and testing data at regular intervals.
2. **Visualizations**:
   - Plots actual and predicted values for each output variable in the dataset.

## Example Results
- Example performance logs during training:
  ```
  Epoch 0: Train RMSE 5.4321, MAE 3.2145, R² 0.8721 | Test RMSE 6.1234, MAE 3.9876, R² 0.8567
  ```


## File Structure
```
.
├── weather_prediction.py     # Main script for training and testing
├── FullWeatherData_Toronto.csv # Dataset file
├── README.md                 # Project documentation
```
