Predicting S&P500 index using LSTM networks

Files/Directories included in the main directory:
	1. LSTM_models - Trained LSTM model files are saved in this directory
	2. LSTM_Prediction.ipynb - Jupyter notebook containing instructions for training the LSTM network and forecasting the prices.
	3. prediction.py - Python script for forecasting prices 
	4. train_lstm_model.py - Python script for training/updating the LSTM model

Instructions:
	1. Install Anaconda3.x.x (Reference - https://www.anaconda.com/distribution/)
	2. Install all the dependencies from requirements.txt file by using the command 'pip install -r requirements.txt' without the single quotes in your terminal
	3. For accessing .ipynb file type 'jupyter notebook' without the quotes in your terminal (If you are using linux as a root user, 'jupyter notebook --allow-root')
	4. For updating or training the LSTM model with newer prices, use a csv file containing Closing prices
	5. For training/updating use the command, 'python train_lstm_model.py' and type the name of csv file (or path of csv file)
	6. For forecasting use the command, 'python prediction.py' and enter the number of days you want to forecast (It is recommended to enter a value less than 15 days and always traing the network on a 15 days interval as described in (5))

For any other queries contact me at, https://fiver.com/omkarudawant 
