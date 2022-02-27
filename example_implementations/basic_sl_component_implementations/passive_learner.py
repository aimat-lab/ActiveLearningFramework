from typing import Tuple, Optional, Sequence

import numpy as np
import tensorflow as tf
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError

from basic_sl_component_interfaces import PassiveLearner
from helpers import X, Y, AddInfo_Y

model_location = {
    "a": 'assets/saved_models/butene_energy_force'
}


class ButenePassiveLearner(PassiveLearner):

    def __init__(self):
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

        self.model = create_model()

        self.x_train, self.y_train = np.array([]), np.array([])

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y], **kwargs) -> None:
        eng, grads = y_train[:, 0:2], np.array(y_train[:, 2:]).reshape((len(y_train), 2, 12, 3))

        x_scaled, y_scaled = self.scaler.fit_transform(x=np.array(x_train).reshape((len(x_train), 12, 3)), y=[eng, grads])
        self.model.precomputed_features = True
        feat_x, feat_grad = self.model.precompute_feature_in_chunks(x_scaled, batch_size=4)
        self.model.set_const_normalization_from_features(feat_x)

        self.model.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=4, epochs=10, verbose=2)

        self.model.precomputed_features = False

    def load_model(self) -> None:
        self.model = tf.keras.models.load_model(model_location["a"], compile=False)
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        mae_energy = ScaledMeanAbsoluteError(scaling_shape=self.scaler.energy_std.shape)
        mae_force = ScaledMeanAbsoluteError(scaling_shape=self.scaler.gradient_std.shape)
        mae_energy.set_scale(self.scaler.energy_std)
        mae_force.set_scale(self.scaler.gradient_std)
        self.model.compile(optimizer=optimizer,
                           loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
                           metrics=[[mae_energy], [mae_force]])

    def close_model(self) -> None:
        del self.model

    def save_model(self) -> None:
        self.model.save(model_location["a"])
        del self.model

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        x_scaled, _ = self.scaler.transform(x=x.reshape((12, 3)), y=[np.zeros(2), np.zeros([2, 12, 3])])
        prediction_scaled = self.model.predict(x_scaled)

        x, prediction = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled)
        return np.append(prediction[0][0].flatten(), prediction[1][0].flatten()), (0,)

    def predict_set(self, xs: Sequence[X]) -> Tuple[Sequence[Y], Sequence[AddInfo_Y]]:
        x_scaled, _ = self.scaler.fit_transform(x=np.array(xs).reshape((len(xs), 12, 3)), y=[np.zeros([len(xs), 2]), np.zeros([len(xs), 2, 12, 3])])
        prediction_scaled = self.model.predict(x_scaled)

        x, prediction = self.scaler.inverse_transform(x=x_scaled, y=prediction_scaled)
        return np.array([np.append(prediction[0][i], prediction[1][i].flatten()) for i in range(len(prediction[0]))]), np.zeros(len(prediction[0]))

    def train(self, x: X, y: Y) -> None:
        if len(self.x_train) == 0:
            self.x_train = np.array([x])
            self.y_train = np.array([y])
        else:
            self.x_train = np.append(self.x_train, [x], axis=0)
            self.y_train = np.append(self.y_train, [y], axis=0)

        if len(self.x_train) == 2:
            self.train_batch(self.x_train, self.y_train)
            self.x_train, self.y_train = np.array([]), np.array([])

    def train_batch(self, xs: Sequence[X], ys: Sequence[Y]):
        eng, grads = ys[:, 0:2], np.array(ys[:, 2:]).reshape((len(ys), 2, 12, 3))

        x_scaled, y_scaled = self.scaler.fit_transform(x=np.array(xs).reshape((len(xs), 12, 3)), y=[eng, grads])
        self.model.precomputed_features = True
        feat_x, feat_grad = self.model.precompute_feature_in_chunks(x_scaled, batch_size=2)

        self.model.set_const_normalization_from_features(feat_x)
        self.model.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=2, epochs=4, verbose=2)

        self.model.precomputed_features = False

    def sl_model_satisfies_evaluation(self) -> bool:
        return False
