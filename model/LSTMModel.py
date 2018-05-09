from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


class LSTMModel:
    def __init__(self, time_steps, num_features):
        self.model = Sequential()
        self.time_steps = time_steps
        self.num_features = num_features

    def define_model(self, mem_units, num_lstm_stack):
        ret_seq = True if num_lstm_stack > 0 else False
        self.model.add(LSTM(mem_units, return_sequences=ret_seq, input_shape=(self.time_steps, self.num_features)))
        while ret_seq:
            if num_lstm_stack == 1:
                ret_seq = False
            self.model.add(LSTM(mem_units, return_sequences=ret_seq))
            num_lstm_stack -= 1
        self.model.add(Dense(self.num_features, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model.summary()

    def fit_model(self, batch_gen, X, y, n):
        self.model.fit(X, y, epochs=1, verbose=2)

    def get_model(self):
        return self.model
