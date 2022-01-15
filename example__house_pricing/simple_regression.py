import logging

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn import preprocessing
from tensorflow.keras.optimizers import RMSprop

from additional_component_interfaces import PassiveLearner


class SimpleRegressionHousing(PassiveLearner):
    """
    Boston housing regression with 3 models (to get a committee)
    """

    def save_model(self):
        self.model_a.save('assets/saved_models/simple_regression_housing__model_a')
        del self.model_a
        self.model_b.save('assets/saved_models/simple_regression_housing__model_b')
        del self.model_b
        self.model_c.save('assets/saved_models/simple_regression_housing__model_c')
        del self.model_c

    def load_model(self):
        self.model_a = load_model('assets/saved_models/simple_regression_housing__model_a')
        self.model_b = load_model('assets/saved_models/simple_regression_housing__model_b')
        self.model_c = load_model('assets/saved_models/simple_regression_housing__model_c')

    def __init__(self):
        # TODO: different models? e.g. different optimizer?
        model_a = Sequential()
        model_a.add(Dense(64, kernel_initializer='normal', activation='relu', input_shape=(13,)))
        model_a.add(Dense(64, activation='relu'))
        model_a.add(Dense(1))
        model_a.compile(loss='mse', optimizer=RMSprop(), metrics=['mean_absolute_error'])

        self.model_a = model_a

        model_b = Sequential()
        model_b.add(Dense(64, kernel_initializer='normal', activation='relu', input_shape=(13,)))
        model_b.add(Dense(64, activation='relu'))
        model_b.add(Dense(1))
        model_b.compile(loss='mse', optimizer=RMSprop(), metrics=['mean_absolute_error'])

        self.model_b = model_b

        model_c = Sequential()
        model_c.add(Dense(64, kernel_initializer='normal', activation='relu', input_shape=(13,)))
        model_c.add(Dense(64, activation='relu'))
        model_c.add(Dense(1))
        model_c.compile(loss='mse', optimizer=RMSprop(), metrics=['mean_absolute_error'])

        self.model_c = model_c

        self.x_train, self.y_train = np.array([]), np.array([])
        self.scaler = None

    def initial_training(self, x_train, y_train, **kwargs):
        x_train_scaled = preprocessing.scale(x_train)
        self.scaler = preprocessing.StandardScaler().fit(x_train)
        batch_size = kwargs.get("batch_size", 5)
        epochs = kwargs.get("epochs", 10)

        # TODO: different models? (small alternations)
        self.model_a.fit(x_train_scaled, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
        self.model_b.fit(x_train_scaled, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
        self.model_c.fit(x_train_scaled, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])

    def predict(self, x):
        x_array = np.array([x, ])

        prediction_a = self.model_a.predict(self.scaler.transform(x_array)).flatten()[0]
        prediction_b = self.model_b.predict(self.scaler.transform(x_array)).flatten()[0]
        prediction_c = self.model_c.predict(self.scaler.transform(x_array)).flatten()[0]

        # TODO maybe calculation of variance in informativeness analyser? => otherwise step is moved into candidate updater
        # TODO uncertainty is currently variance => not normalized,
        return np.mean(np.array([prediction_a, prediction_b, prediction_c]), axis=0), np.var(np.array([prediction_a, prediction_b, prediction_c]), axis=0)

    def train(self, x, y):
        if len(self.x_train) == 0:
            self.x_train = np.array([x])
        else:
            self.x_train = np.append(self.x_train, [x], axis=0)
        self.y_train = np.append(self.y_train, y)

        if len(self.x_train) == 16:
            self.train_batch(self.x_train, self.y_train, 4, 5)
            self.x_train, self.y_train = np.array([]), np.array([])

    def train_batch(self, x_train, y_train, batch_size, epochs):
        self.model_a.fit(self.scaler.transform(x_train), y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
        self.model_b.fit(self.scaler.transform(x_train), y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
        self.model_c.fit(self.scaler.transform(x_train), y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
