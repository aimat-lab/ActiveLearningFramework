import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing
from tensorflow.keras.optimizers import RMSprop

from Interfaces import PassiveLearner


class SimpleRegression_Housing(PassiveLearner):

    def __init__(self):
        model = Sequential()
        model.add(Dense(64, kernel_initializer='normal', activation='relu', input_shape=(13,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=RMSprop(), metrics=['mean_absolute_error'])

        self.model = model

    def initial_training(self, x_train, y_train, batch_size, epochs):
        x_train_scaled = preprocessing.scale(x_train)
        self.scaler = preprocessing.StandardScaler().fit(x_train)
        self.model.fit(x_train_scaled, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=20)])

    def predict(self, x):
        x_array = np.array([x, ])
        return self.model.predict(self.scaler.transform(x_array)).flatten()[0]

    def train_batch(self, x_train, y_train, batch_size, epochs):
        self.model.fit(self.scaler.transform(x_train), y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                       validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
