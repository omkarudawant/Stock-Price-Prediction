from warnings import filterwarnings
filterwarnings('ignore')
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import r2_score, mean_squared_error
import os
import pandas as pd
import numpy as np

lstm_models_folder = 'LSTM_models/'
if not os.path.exists(lstm_models_folder):
    os.mkdir(lstm_models_folder)

# Mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# split a univariate sequence into samples
def split_sequence(sequence, n_steps, train_size):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    # Train_Test split
    split_X = int(len(X) * train_size)
    split_y = int(len(y) * train_size)
    X_train, y_train = np.array(X[:split_X]), np.array(y[:split_y]),

    split_x_val = int(len(X[int(split_X):]) * 0.5)
    split_y_val = int(len(y[int(split_y):]) * 0.5)
    print(split_x_val, split_y_val)

    X_test, y_test = np.array(X[split_X:(split_X + split_x_val)]), np.array(
        y[split_y:(split_y + split_y_val)])

    X_val, y_val = np.array(X[(split_X + split_x_val):]), np.array(
        y[(split_y + split_y_val):])

    return X_train, X_test, y_train, y_test, X_val, y_val


def train_model(X_train, y_train, X_val, y_val, n_steps, n_features):
	# define model
	model_vanilla = Sequential()
	model_vanilla.add(
	    LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
	model_vanilla.add(Dense(1))
	model_vanilla.compile(optimizer='adam', loss='mse')
	print('Model architecture details: \n')
	print(model_vanilla.summary())

	# fit model
	print('Starting training...\n')
	history_vanilla = model_vanilla.fit(X_train,
	                                    y_train,
	                                    epochs=150,
	                                    verbose=0,
	                                    validation_data=(X_val, y_val))
	print('Training done !')
	return model_vanilla


df = pd.read_csv(input('Enter path/name of csv file: '), 
                 index_col=[0],
                 parse_dates=True)

# define input sequence
raw_seq = list(df.values)

# choose a number of time steps
n_steps = int(input('Enter number of days to look behind: '))

# split into samples
X_train, X_test, y_train, y_test, X_val, y_val = split_sequence(raw_seq,
                                                                n_steps,
                                                                train_size=0.9)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

model_vanilla = train_model(X_train, y_train, X_val, y_val, n_steps, n_features)

# demonstrate prediction
yhat = model_vanilla.predict(X_test, verbose=0)

# Evaluation metrics
r2 = round(r2_score(y_test, yhat), 3)
mape = round(mean_absolute_percentage_error(y_test, yhat), 3)

# Saving the model
model_name = 'vanilla_lstm-' + f'r2({r2})-mape({mape}).h5'
model_vanilla.save(lstm_models_folder + model_name)
print(f'Model saved in {lstm_models_folder} as {model_name}')

# Printing the Evaluation metrics
print(f'\nr2(0 to 1, higher the better): {r2}\nMape(0 to 100, lower the better): {mape}')
