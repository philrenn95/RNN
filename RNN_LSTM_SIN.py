#Philipp Renner
#Creates a sin wave to be forecasted by an RNN and an LSTM
###########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

class SinWaveForecaster:
    def __init__(self, data):
        self.data = data
        self.train = None
        self.test = None
        self.scaler = MinMaxScaler()
        self.scaled_train = None
        self.scaled_test = None
        self.generator = None
        self.model = None

    def preprocess_data(self, test_percent=0.1):
        x = np.linspace(0, 50, 501)
        y = np.sin(x)
        df = pd.DataFrame(data=y, index=x, columns=['Sine'])

        test_point = np.round(len(df) * test_percent)
        test_ind = int(len(df) - test_point)
        self.train = df.iloc[:test_ind]
        self.test = df.iloc[test_ind:]

        self.scaler.fit(self.train)
        self.scaled_train = self.scaler.transform(self.train)
        self.scaled_test = self.scaler.transform(self.test)

    def create_generator(self, length, batch_size):
        self.generator = TimeseriesGenerator(
            self.scaled_train, self.scaled_train, length=length, batch_size=batch_size
        )

    def train_rnn(self, length=2):
        n_features = 1

        self.model = Sequential()
        self.model.add(SimpleRNN(50, input_shape=(length, n_features)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit_generator(self.generator, epochs=5)

    def generate_rnn_predictions(self):
        length = 50
        first_eval_batch = self.scaled_train[-length:]
        current_batch = first_eval_batch.reshape((1, length, 1))
        test_predictions = []

        for i in range(len(self.test)):
            current_pred = self.model.predict(current_batch)[0]
            test_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        true_predictions = self.scaler.inverse_transform(test_predictions)
        self.test['RNN Predictions'] = true_predictions

    def train_lstm(self, length=49):
        generator = TimeseriesGenerator(
            self.scaled_train, self.scaled_train, length=length, batch_size=1
        )

        validation_generator = TimeseriesGenerator(
            self.scaled_test, self.scaled_test, length=length, batch_size=1
        )

        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(length, 1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit_generator(
            generator, epochs=20, validation_data=validation_generator,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
        )

    def generate_lstm_predictions(self):
        length = 49
        first_eval_batch = self.scaled_train[-length:]
        current_batch = first_eval_batch.reshape((1, length, 1))
        test_predictions = []

        for i in range(len(self.test)):
            current_pred = self.model.predict(current_batch)[0]
            test_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        true_predictions = self.scaler.inverse_transform(test_predictions)
        self.test['LSTM Predictions'] = true_predictions

    def evaluate_full_data(self):
        full_scaler = MinMaxScaler()
        scaled_full_data = full_scaler.fit_transform(self.data)

        length = 50
        generator = TimeseriesGenerator(
            scaled_full_data, scaled_full_data, length=length, batch_size=1
        )

        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(length, 1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit_generator(generator, epochs=6)

        forecast = []
        first_eval_batch = scaled_full_data[-length:]
        current_batch = first_eval_batch.reshape((1, length, 1))

        for i in range(len(self.test)):
            current_pred = self.model.predict(current_batch)[0]
            forecast.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        forecast = full_scaler.inverse_transform(forecast)
        forecast_index = np.arange(50.1, 55.1, step=0.1)
        plt.plot(self.data.index, self.data['Sine'])
        plt.plot(forecast_index, forecast)
        plt.show()


# Main script
if __name__ == "__main__":
    # Create an instance of the SinWaveForecaster class
    forecaster = SinWaveForecaster(data=None)

    # Preprocess the data
    forecaster.preprocess_data()

    # Create generator for RNN
    forecaster.create_generator(length=2, batch_size=1)

    # Train RNN
    forecaster.train_rnn()

    # Generate RNN predictions
    forecaster.generate_rnn_predictions()

    # Create generator for LSTM
    forecaster.create_generator(length=49, batch_size=1)

    # Train LSTM
    forecaster.train_lstm()

    # Generate LSTM predictions
    forecaster.generate_lstm_predictions()

    # Evaluate full data
    forecaster.evaluate_full_data()
