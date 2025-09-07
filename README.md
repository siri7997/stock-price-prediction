# stock-price-prediction
📈 LSTM Stock Price Prediction

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical closing data. The example in this repository focuses on Apple Inc. (AAPL), but it can be easily adapted to other stock tickers.

🚀 Features

Fetches historical stock data using yfinance

Preprocesses data with MinMaxScaler

Splits into training and testing datasets (65% / 35%)

Builds a stacked LSTM model using TensorFlow/Keras

Evaluates performance with metrics:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

MAPE (Mean Absolute Percentage Error)

Visualizes:

Actual vs Predicted prices (Train & Test sets)

Forecasted prices for the next 30 days

Extended stock price trajectory with future predictions

📂 Project Structure
├── stock_lstm.py      # Main script with LSTM model
├── README.md          # Project documentation

🛠️ Requirements

Install dependencies before running the script:

pip install yfinance pandas numpy matplotlib scikit-learn tensorflow

▶️ Usage

Run the script directly:

python stock_lstm.py


The script will:

Download Apple (AAPL) stock data (2015–2024)

Train the LSTM model

Print evaluation metrics

Show plots:

Historical vs predicted prices

Next 30-day forecast

📊 Example Output

Training vs Test Prediction Plot
Shows how well the LSTM fits historical data.

30-Day Forecast Plot
Displays predicted stock prices for the next 30 days.

🔧 Customization

You can easily change the stock ticker by modifying this line in the script:

df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')


Replace 'AAPL' with any other stock ticker (e.g., 'MSFT', 'GOOGL', 'TSLA').

⚠️ Disclaimer

This project is for educational and research purposes only.
It should not be used as financial advice or for making real trading decisions.
