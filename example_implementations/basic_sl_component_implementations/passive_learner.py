import logging
from typing import Tuple, Optional, Sequence

import numpy as np
import tensorflow as tf
from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel
from example_implementations.pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from example_implementations.pyNNsMD.utils.loss import ScaledMeanAbsoluteError

from basic_sl_component_interfaces import PassiveLearner
from helpers import X, Y, AddInfo_Y

model_location = {
    "a": 'assets/saved_models/a_butene_energy_force',
    "b": 'assets/saved_models/b_butene_energy_force'
}


class ButenePassiveLearner(PassiveLearner):

    def __init__(self, x_test, eng_test, grads_test):
        self.scaler = EnergyGradientStandardScaler()

        def create_model():
            model = EnergyGradientModel(atoms=12, states=2, invd_index=True)
            optimizer = tf.keras.optimizers.Adam(lr=1e-3)
            mae_energy = ScaledMeanAbsoluteError(scaling_shape=self.scaler.energy_std.shape)
            mae_force = ScaledMeanAbsoluteError(scaling_shape=self.scaler.gradient_std.shape)
            mae_energy.set_scale(self.scaler.energy_std)
            mae_force.set_scale(self.scaler.gradient_std)
            model.compile(optimizer=optimizer,
                          loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
                          metrics=[[mae_energy], [mae_force]])
            return model

        self.model_a = create_model()
        self.model_b = create_model()

        self.x_train, self.y_train = np.array([]), np.array([])
        self.x_test, self.eng_test, self.grads_test = x_test, eng_test, grads_test
        self.loss_history = []

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y]) -> None:
        eng, grads = y_train[:, 0:2], np.array(y_train[:, 2:]).reshape((len(y_train), 2, 12, 3))

        x_scaled, y_scaled = self.scaler.fit_transform(x=np.array(x_train).reshape((len(x_train), 12, 3)), y=[eng, grads])

        self.model_a.precomputed_features = True
        self.model_b.precomputed_features = True

        feat_x, feat_grad = self.model_a.precompute_feature_in_chunks(x_scaled, batch_size=5)
        self.model_a.set_const_normalization_from_features(feat_x)
        self.model_b.set_const_normalization_from_features(feat_x)

        self.model_a.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=5, epochs=2, verbose=2)
        self.model_b.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=5, epochs=2, verbose=2)

        self.model_a.precomputed_features = False
        self.model_b.precomputed_features = False

    def load_model(self) -> None:
        self.model_a = tf.keras.models.load_model(model_location["a"], custom_objects={"EnergyGradientModel": EnergyGradientModel}, compile=False)
        self.model_b = tf.keras.models.load_model(model_location["b"], custom_objects={"EnergyGradientModel": EnergyGradientModel}, compile=False)
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        mae_energy = ScaledMeanAbsoluteError(scaling_shape=self.scaler.energy_std.shape)
        mae_force = ScaledMeanAbsoluteError(scaling_shape=self.scaler.gradient_std.shape)
        mae_energy.set_scale(self.scaler.energy_std)
        mae_force.set_scale(self.scaler.gradient_std)
        self.model_a.compile(optimizer=optimizer,
                             loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
                             metrics=[[mae_energy], [mae_force]])
        self.model_b.compile(optimizer=optimizer,
                             loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
                             metrics=[[mae_energy], [mae_force]])

    def close_model(self) -> None:
        del self.model_a
        del self.model_b

    def save_model(self) -> None:
        self.model_a.save(model_location["a"])
        del self.model_a
        self.model_b.save(model_location["b"])
        del self.model_b

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        x_scaled, _ = self.scaler.transform(x=x.reshape((12, 3)), y=[np.zeros(2), np.zeros([2, 12, 3])])
        prediction_scaled_a = self.model_a.predict(x_scaled)
        prediction_scaled_b = self.model_b.predict(x_scaled)

        _, prediction_a = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_a)
        _, prediction_b = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_b)

        y_a, y_b = np.append(prediction_a[0][0], prediction_a[1][0].flatten()), np.append(prediction_b[0][0], prediction_b[1][0].flatten())
        return np.mean(np.array([y_a, y_b]), axis=0), np.var(np.array([y_a, y_b]), axis=0)

    def predict_set(self, xs: Sequence[X]) -> Tuple[Sequence[Y], Sequence[AddInfo_Y]]:
        x_scaled, _ = self.scaler.fit_transform(x=np.array(xs).reshape((len(xs), 12, 3)), y=[np.zeros([len(xs), 2]), np.zeros([len(xs), 2, 12, 3])])

        prediction_scaled_a = self.model_a.predict(x_scaled)
        prediction_scaled_b = self.model_b.predict(x_scaled)

        _, prediction_a = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_a)
        _, prediction_b = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_b)

        y_a = np.array([np.append(prediction_a[0][i], prediction_a[1][i].flatten()) for i in range(len(prediction_a[0]))])
        y_b = np.array([np.append(prediction_b[0][i], prediction_b[1][i].flatten()) for i in range(len(prediction_b[0]))])

        return np.mean(np.array([y_a, y_b]), axis=0), np.var(np.array([y_a, y_b]), axis=0)

    def train(self, x: X, y: Y) -> None:
        if len(self.x_train) == 0:
            self.x_train = np.array([x])
            self.y_train = np.array([y])
        else:
            self.x_train = np.append(self.x_train, [x], axis=0)
            self.y_train = np.append(self.y_train, [y], axis=0)

        self.train_batch(self.x_train, self.y_train)
        if len(self.x_train) > 5:
            self.x_train, self.y_train = self.x_train[1:], self.y_train[1:]

    def train_batch(self, xs: Sequence[X], ys: Sequence[Y]):
        eng, grads = ys[:, 0:2], np.array(ys[:, 2:]).reshape((len(ys), 2, 12, 3))

        x_scaled, y_scaled = self.scaler.fit_transform(x=np.array(xs).reshape((len(xs), 12, 3)), y=[eng, grads])

        self.model_a.precomputed_features = True
        feat_x, feat_grad = self.model_a.precompute_feature_in_chunks(x=x_scaled, batch_size=2)
        self.model_a.set_const_normalization_from_features(feat_x)
        self.model_a.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=5, epochs=1, verbose=2)
        self.model_a.precomputed_features = False

        self.model_b.precomputed_features = True
        feat_x, feat_grad = self.model_b.precompute_feature_in_chunks(x=x_scaled, batch_size=2)
        self.model_b.set_const_normalization_from_features(feat_x)
        self.model_b.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=5, epochs=1, verbose=2)
        self.model_b.precomputed_features = False

    def sl_model_satisfies_evaluation(self) -> bool:
        x_scaled, y_scaled = self.scaler.fit_transform(x=self.x_test, y=[self.eng_test, self.grads_test])

        prediction_scaled_a = self.model_a.predict(x_scaled)
        _, prediction_a = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_a)
        prediction_scaled_b = self.model_b.predict(x_scaled)
        _, prediction_b = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled_b)

        current_loss = np.mean(np.abs(self.eng_test - np.mean(np.array([prediction_a[0], prediction_b[0]]), axis=0))) * 1 + np.mean(np.abs(self.grads_test - np.mean(np.array([prediction_a[1], prediction_b[1]])))) * 5

        if (len(self.loss_history) == 0) or not (current_loss == self.loss_history[-1]):  # ignores the case, that an updated model performs the same as the previous model (very unlikely)
            self.loss_history.append(current_loss)
        else:
            # if now new information gets provided, no need for evaluation
            return False

        if len(self.loss_history) < 12:
            return False
        else:
            logging.info(f"PERFORMANCE EVALUATION, LOSS HISTORY: {self.loss_history}")
            logging.info(f"Variance: {np.var(np.array(self.loss_history[-12:-9]))}, {np.var(np.array(self.loss_history[-9:-6]))}, {np.var(np.array(self.loss_history[-6:-3]))}, {np.var(np.array(self.loss_history[-3:]))}")
            return np.var(np.array(self.loss_history[-12:-9])) > np.var(np.array(self.loss_history[-9:-6])) * 1.2 > np.var(np.array(self.loss_history[-6:-3])) * 2.5 > np.var(np.array(self.loss_history[-3:])) * 3.8
