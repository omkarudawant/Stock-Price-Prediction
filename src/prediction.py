from warnings import filterwarnings
filterwarnings('ignore')
from keras.models import load_model
import pandas as pd
import numpy as np

def get_input_sequesnce(raw_seq):
    # define input sequence
    X, y = list(), list()
    n_steps = 10
    for i in range(len(raw_seq)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(raw_seq) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = raw_seq[i:end_ix], raw_seq[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def predict_prices(model, ref_data, days):
    model = load_model(model)
    df_pred = pd.read_csv(ref_data, index_col=[0], parse_dates=True)
    predictions = list()
    
    for i in range(days):
        input_values = df_pred.values[-11:]
        X, y = get_input_sequesnce(input_values)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        df_pred = df_pred.append([{'Close':model.predict(X)[0, 0]}], ignore_index=True)
        predictions.append(model.predict(X)[0, 0])
        
    print(predictions)
    pred = {'Close_predictions': predictions}
    t = pd.to_datetime(df_pred.index[-1])
    indexes = pd.date_range(str(t.year) + '-' + str(t.month) + '-' + str(t.day), periods=days)
    pred = pd.DataFrame(pred, index=indexes)
    pred.to_csv('Predictions.csv')


predict_prices(model='LSTM_models/vanilla_lstm-r2(0.85)-mape(0.87).h5',
               ref_data='SP500_Sheet.csv',
               days=int(input('Enter days in future to predict: ')))
