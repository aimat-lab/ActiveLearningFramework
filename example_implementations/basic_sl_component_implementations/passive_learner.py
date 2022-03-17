import logging
import os
from typing import Tuple, Optional, Sequence

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback

from basic_sl_component_interfaces import PassiveLearner
from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel
from example_implementations.pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from example_implementations.pyNNsMD.utils.loss import ScaledMeanAbsoluteError
from helpers import X, Y, AddInfo_Y

model_location = "assets/saved_models/pbs/"
weight_file_name = {
    "a": '1__weights_a.h5',
    "b": '1__weights_b.h5',
    "c": '1__weights_c.h5'
}


class CallbackStopIfLossLow(Callback):

    def __init__(self, min_epoch, thr):
        super().__init__()
        self.thr, self.min_epoch = thr, min_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('loss') <= self.thr and epoch >= self.min_epoch:
            self.model.stop_training = True


def _create_model(scaler):
    model = EnergyGradientModel(atoms=12, states=2, invd_index=True)
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    mae_energy = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
    mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
    mae_energy.set_scale(scaler.energy_std)
    mae_force.set_scale(scaler.gradient_std)
    model.compile(optimizer=optimizer,
                  loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
                  metrics=[[mae_energy], [mae_force]])
    return model


class ButenePassiveLearner(PassiveLearner):

    def __init__(self, x_test, eng_test, grads_test):
        self.scaler = EnergyGradientStandardScaler()

        self.model_a: EnergyGradientModel = _create_model(self.scaler)
        self.model_b: EnergyGradientModel = _create_model(self.scaler)
        self.model_c: EnergyGradientModel = _create_model(self.scaler)

        self.x_train, self.y_train = np.array([]), np.array([])
        self.x_test, self.eng_test, self.grads_test = x_test, eng_test, grads_test
        self.loss_history = []

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y]) -> None:
        eng, grads = y_train[:, 0:2], np.array(y_train[:, 2:]).reshape((len(y_train), 2, 12, 3))

        x_scaled, y_scaled = self.scaler.fit_transform(x=np.array(x_train).reshape((len(x_train), 12, 3)), y=[eng, grads])

        self.model_a.precomputed_features = True
        self.model_b.precomputed_features = True
        self.model_c.precomputed_features = True

        feat_x, feat_grad = self.model_a.precompute_feature_in_chunks(x_scaled, batch_size=4)
        self.model_a.set_const_normalization_from_features(feat_x)
        self.model_b.set_const_normalization_from_features(feat_x)
        self.model_c.set_const_normalization_from_features(feat_x)

        self.model_a.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=4, epochs=2000, verbose=2, callbacks=[CallbackStopIfLossLow(thr=0.5, min_epoch=100)])
        self.model_b.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=4, epochs=2000, verbose=2, callbacks=[CallbackStopIfLossLow(thr=0.5, min_epoch=100)])
        self.model_c.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=4, epochs=2000, verbose=2, callbacks=[CallbackStopIfLossLow(thr=0.5, min_epoch=100)])

        self.model_a.precomputed_features = False
        self.model_b.precomputed_features = False
        self.model_c.precomputed_features = False

        self.x_train = x_train
        self.y_train = y_train

    def load_model(self) -> None:
        filename = os.path.abspath(os.path.abspath(model_location))
        os.makedirs(filename, exist_ok=True)

        self.model_a: EnergyGradientModel = _create_model(self.scaler)
        self.model_a.load_weights(os.path.join(filename, weight_file_name["a"]))

        self.model_b: EnergyGradientModel = _create_model(self.scaler)
        self.model_b.load_weights(os.path.join(filename, weight_file_name["b"]))

        self.model_c: EnergyGradientModel = _create_model(self.scaler)
        self.model_c.load_weights(os.path.join(filename, weight_file_name["c"]))

    def close_model(self) -> None:
        del self.model_a
        del self.model_b
        del self.model_c

    def save_model(self) -> None:
        # Folder to store model in
        filename = os.path.abspath(os.path.abspath(model_location))
        os.makedirs(filename, exist_ok=True)

        self.model_a.save_weights(os.path.join(filename, weight_file_name["a"]))
        del self.model_a

        self.model_b.save_weights(os.path.join(filename, weight_file_name["b"]))
        del self.model_b

        self.model_c.save_weights(os.path.join(filename, weight_file_name["c"]))
        del self.model_c

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        x_scaled, _ = self.scaler.transform(x=x.reshape((12, 3)), y=[np.zeros(2), np.zeros([2, 12, 3])])
        prediction_scaled_a = self.model_a.predict(x_scaled)
        prediction_scaled_b = self.model_b.predict(x_scaled)
        prediction_scaled_c = self.model_c.predict(x_scaled)

        _, prediction_a = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_a)
        _, prediction_b = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_b)
        _, prediction_c = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_c)

        y_a, y_b, y_c = np.append(prediction_a[0][0], prediction_a[1][0].flatten()), np.append(prediction_b[0][0], prediction_b[1][0].flatten()), np.append(prediction_c[0][0], prediction_c[1][0].flatten())
        return np.mean(np.array([y_a, y_b, y_c]), axis=0), np.var(np.array([y_a, y_b, y_c]), axis=0)

    def predict_set(self, xs: Sequence[X]) -> Tuple[Sequence[Y], Sequence[AddInfo_Y]]:
        x_scaled, _ = self.scaler.fit_transform(x=np.array(xs).reshape((len(xs), 12, 3)), y=[np.zeros([len(xs), 2]), np.zeros([len(xs), 2, 12, 3])])

        prediction_scaled_a = self.model_a.predict(x_scaled)
        prediction_scaled_b = self.model_b.predict(x_scaled)
        prediction_scaled_c = self.model_c.predict(x_scaled)

        _, prediction_a = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_a)
        _, prediction_b = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_b)
        _, prediction_c = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_c)

        y_a = np.array([np.append(prediction_a[0][i], prediction_a[1][i].flatten()) for i in range(len(prediction_a[0]))])
        y_b = np.array([np.append(prediction_b[0][i], prediction_b[1][i].flatten()) for i in range(len(prediction_b[0]))])
        y_c = np.array([np.append(prediction_c[0][i], prediction_c[1][i].flatten()) for i in range(len(prediction_c[0]))])

        return np.mean(np.array([y_a, y_b, y_c]), axis=0), np.var(np.array([y_a, y_b, y_c]), axis=0)

    def train(self, x: X, y: Y) -> None:
        if len(self.x_train) == 0:
            self.x_train = np.array([x])
            self.y_train = np.array([y])
        else:
            self.x_train = np.append(self.x_train, [x], axis=0)
            self.y_train = np.append(self.y_train, [y], axis=0)

        batch_size = 16
        repeat_times = 16
        if len(self.x_train) % batch_size == 0:
            # self.train_batch_early_stopping(self.x_train[-batch_size:], self.y_train[-batch_size:], batch_size, 0.5, 100, 1000)
            self.train_batch_early_stopping(self.x_train, self.y_train, batch_size, 0.5, 100, 2000)

        if len(self.x_train) == batch_size*repeat_times:
            self.x_train, self.y_train = self.x_train[-(batch_size*repeat_times):], self.y_train[-(batch_size*repeat_times):]

    def train_batch_early_stopping(self, xs: Sequence[X], ys: Sequence[Y], batch_size: int, thr, min_epoch, max_epoch):
        eng, grads = ys[:, 0:2], np.array(ys[:, 2:]).reshape((len(ys), 2, 12, 3))

        x_scaled, y_scaled = self.scaler.fit_transform(x=np.array(xs).reshape((len(xs), 12, 3)), y=[eng, grads])

        self.model_a.precomputed_features = True
        feat_x, feat_grad = self.model_a.precompute_feature_in_chunks(x=x_scaled, batch_size=batch_size)
        self.model_a.set_const_normalization_from_features(feat_x)
        self.model_a.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=batch_size, epochs=max_epoch, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epoch)])
        self.model_a.precomputed_features = False

        self.model_b.precomputed_features = True
        feat_x, feat_grad = self.model_b.precompute_feature_in_chunks(x=x_scaled, batch_size=batch_size)
        self.model_b.set_const_normalization_from_features(feat_x)
        self.model_b.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=batch_size, epochs=max_epoch, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epoch)])
        self.model_b.precomputed_features = False

        self.model_c.precomputed_features = True
        feat_x, feat_grad = self.model_c.precompute_feature_in_chunks(x=x_scaled, batch_size=batch_size)
        self.model_c.set_const_normalization_from_features(feat_x)
        self.model_c.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=batch_size, epochs=max_epoch, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epoch)])
        self.model_c.precomputed_features = False

    def sl_model_satisfies_evaluation(self) -> bool:
        x_scaled, y_scaled = self.scaler.fit_transform(x=self.x_test, y=[self.eng_test, self.grads_test])

        prediction_scaled_a = self.model_a.predict(x_scaled)
        _, prediction_a = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_a)
        prediction_scaled_b = self.model_b.predict(x_scaled)
        _, prediction_b = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_b)
        prediction_scaled_c = self.model_b.predict(x_scaled)
        _, prediction_c = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_c)

        current_loss = np.mean(np.abs(self.eng_test - np.mean(np.array([prediction_a[0], prediction_b[0], prediction_c[0]]), axis=0))) * 1 + np.mean(np.abs(self.grads_test - np.mean(np.array([prediction_a[1], prediction_b[1], prediction_c[1]])))) * 5

        if (len(self.loss_history) == 0) or not (current_loss == self.loss_history[-1]):  # ignores the case, that an updated model performs the same as the previous model (very unlikely)
            self.loss_history.append(current_loss)
        else:
            # if now new information gets provided, no need for evaluation
            return False

        energy_loss = np.mean(np.abs(self.eng_test - np.mean(np.array([prediction_a[0], prediction_b[0], prediction_c[0]]), axis=0)))

        logging.info(f"PERFORMANCE EVALUATION, LOSS HISTORY: {self.loss_history}")
        logging.info(f"PERFORMANCE EVALUATION, energy_loss: {energy_loss}")

        return energy_loss < 0.5

