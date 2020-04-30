[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/omkarudawant/Stock-Price-Prediction-with-LSTM/master)

# Predicting S&P500 index using LSTM networks

## Files/Directories included in the main directory:
 
 - research/LSTM_Prediction.ipynb - Jupyter notebook containing instructions for training the LSTM network and forecasting the prices.
 
 - src/prediction.py - Python script for forecasting prices 
 
 - src/train_lstm_model.py - Python script for training/updating the LSTM model

## Instructions:

 - Install all the dependencies from requirements.txt

 - For updating or training the LSTM model with newer prices, use a csv file containing Closing prices
 
 - For training/updating use the command, 'python train_lstm_model.py' and type the name of csv file (or path of csv file)
 
 - For forecasting use the command, 'python prediction.py' and enter the number of days you want to forecast (It is recommended to enter a value less than 15 days and always traing the network on a 15 days interval)
