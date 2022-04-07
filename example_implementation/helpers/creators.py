from keras.layers import Dense
from keras.models import Sequential


def create_model():
    model = Sequential()

    model.add(Dense(128, input_shape=(8,), activation='relu', name='dense_1'))
    model.add(Dense(64, activation='relu', name='dense_2'))
    model.add(Dense(1, activation='linear', name='dense_output'))

    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    return model
