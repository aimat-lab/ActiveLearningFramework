import logging

import tensorflow as tf
from keras.callbacks import Callback
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError

from example_implementations.helpers.mapper import map_shape_output_to_flat
from example_implementations.evaluation.metrics import print_evaluation
from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel

logging.basicConfig(format='\n%(name)s, %(levelname)s: %(message)s', level=logging.INFO)
log = logging.getLogger("LOGGER sl model:  ")


class CallbackStopIfLossLow(Callback):

    def __init__(self, min_epoch, thr):
        super().__init__()
        self.thr, self.min_epoch = thr, min_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('loss') <= self.thr and epoch - 1 >= self.min_epoch:
            self.model.stop_training = True


def _create_model_and_scaler():
    # Generate model
    model = EnergyGradientModel(atoms=12, states=2, invd_index=True)

    # Scale in- and output
    # Important: x, energy and gradients can not be scaled completely independent!!
    scaler = EnergyGradientStandardScaler()

    # compile model with optimizer
    # And use scaled metric to revert the standardization of the output for metric during fit updates (optional).
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    mae_energy = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
    mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
    mae_energy.set_scale(scaler.energy_std)
    mae_force.set_scale(scaler.gradient_std)
    model.compile(optimizer=optimizer, loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5], metrics=[[mae_energy], [mae_force]])

    return model, scaler


class SLModel:

    def __init__(self, x, x_test, y, y_test, title_prefix = ""):
        self._x, self._x_test = x, x_test
        self._eng, self._eng_test, self._grads, self._grads_test = y[0], y_test[0], y[1], y_test[1]
        self._title_prefix = title_prefix

        self._model, self._scaler = _create_model_and_scaler()

    def train(self, max_epochs, min_epochs, thr):
        x_scaled, y_scaled = self._scaler.fit_transform(x=self._x, y=[self._eng, self._grads])
        # Precompute features plus derivative
        # Features are normalized automatically
        self._model.precomputed_features = True
        feat_x, feat_grad = self._model.precompute_feature_in_chunks(x_scaled, batch_size=32)
        self._model.set_const_normalization_from_features(feat_x)

        # fit with precomputed features and normalized energies, gradients
        self._model.fit(x=[feat_x, feat_grad], y=y_scaled, batch_size=32, epochs=max_epochs, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epochs)])

        self._model.precomputed_features = False

    def evaluate(self):
        # EVALUATION TEST SET
        title_test_set = self._title_prefix + "test set"

        self._model.precomputed_features = False
        x_scaled_test, y_scaled_test = self._scaler.fit_transform(x=self._x_test, y=[self._eng_test, self._grads_test])
        y_pred_test = self._model.predict(x_scaled_test)
        x_pred_test, y_pred_test = self._scaler.inverse_transform(x=x_scaled_test, y=y_pred_test)

        print_evaluation(title_test_set, map_shape_output_to_flat(y_pred_test), map_shape_output_to_flat([self._eng_test, self._grads_test]))

        # EVALUATION TRAINING SET
        title_training_set = self._title_prefix + "training set"

        self._model.precomputed_features = False
        x_scaled, y_scaled = self._scaler.fit_transform(x=self._x, y=[self._eng, self._grads])
        y_pred = self._model.predict(x_scaled)
        x_pred, y_pred = self._scaler.inverse_transform(x=x_scaled, y=y_pred)

        print_evaluation(title_training_set, map_shape_output_to_flat(y_pred), map_shape_output_to_flat([self._eng, self._grads]))

