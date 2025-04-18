# 📈 LSTM-Based Stock Price Prediction for Mastercard

This project applies a Long Short-Term Memory (LSTM) neural network to forecast Mastercard Inc.'s closing stock prices using historical data. It includes data preprocessing, model training, evaluation, and visualization of predictions against actual values.

---

## 📌 Overview

- **Goal:** Forecast Mastercard's stock closing prices using deep learning.
- **Method:** Sequential LSTM model with simplified feature engineering.
- **Tools:** Python, TensorFlow/Keras, Pandas, Matplotlib, Scikit-learn.

---

## 📂 Project Structure

lstm-stock-prediction/ ├── data/ │ └── Mastercard_stock_history.csv # Historical stock data (excluded from GitHub) ├── src/ │ └── lstm_mastercard.py # Main training and prediction script ├── results/ │ └── plots/ # Output prediction vs actual plots ├── notebook/ │ └── LSTM_Stock_Prediction.ipynb # (Optional) Notebook version of the script ├── README.md # Project documentation ├── requirements.txt # List of dependencies └── .gitignore

yaml
Copy
Edit

---

## 🧠 Model Architecture

- 4 stacked LSTM layers with 50 units each
- Dropout layers (20%) after each LSTM to prevent overfitting
- Final Dense layer for output
- Compiled with:
  - **Optimizer:** Adam
  - **Loss:** Mean Squared Error (MSE)
  - **Metrics:** MSE, MAE, MAPE

---

## 🔄 Data Processing

- Filtered data from 2016 to 2023.
- Normalized `Close` prices between 0 and 1 using `MinMaxScaler`.
- Created sequences of 80 time steps for training.
- Split into training and testing datasets.

---

## 📊 Results

- The model captures general price trends accurately.
- Visualizations show predicted values closely tracking actual prices.
- Model performance metrics:
  - **MSE**, **MAE**, **MAPE** tracked over training epochs.

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/lstm-stock-prediction.git
cd lstm-stock-prediction
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the model
bash
Copy
Edit
python src/lstm_mastercard.py
Ensure data/Mastercard_stock_history.csv is present.

📦 Dependencies
See requirements.txt. Includes:

pandas

numpy

matplotlib

scikit-learn

tensorflow

keras

👩‍💻 Author
Roxana Niksefat
M.Sc. in Applied Modeling & Quantitative Methods
GitHub: @roxananik

📃 License
This project is licensed under the MIT License.