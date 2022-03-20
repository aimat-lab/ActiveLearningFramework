import os

import numpy as np

from example_implementations.helpers.metrics import calc_final_evaluation
from example_implementations.helpers import properties
from example_implementations.helpers.callbacks import CallbackDocumentation, CallbackStopIfLossLow
from example_implementations.helpers.mapper import map_shape_output_to_flat
from example_implementations.helpers.model_creator import create_model_and_scaler


class SLModel:

    def __init__(self, x, x_test, y, y_test, entity):
        self._x, self._x_test = x, x_test
        self._eng, self._eng_test, self._grads, self._grads_test = y[0], y_test[0], y[1], y_test[1]
        self._entity = entity

        self._model, self._scaler = create_model_and_scaler()

        # create store for loss history, final predictions
        filename = os.path.abspath(os.path.abspath(properties.results_location["loss_over_epochs"]))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, entity + properties.loss_history_suffix), np.asarray([]))
        filename = os.path.abspath(os.path.abspath(properties.results_location["prediction_image"]))
        os.makedirs(filename, exist_ok=True)

    def train(self, max_epochs, min_epochs, thr):
        x_scaled, y_scaled = self._scaler.fit_transform(x=self._x, y=[self._eng, self._grads])
        # Precompute features plus derivative
        # Features are normalized automatically
        self._model.precomputed_features = True
        feat_x, feat_grad = self._model.precompute_feature_in_chunks(x_scaled, batch_size=32)
        self._model.set_const_normalization_from_features(feat_x)

        # fit with precomputed features and normalized energies, gradients
        self._model.fit(x=[feat_x, feat_grad], y=y_scaled, batch_size=32, epochs=max_epochs, verbose=2,
                        callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epochs), CallbackDocumentation(entity=self._entity)])

        self._model.precomputed_features = False

    def evaluate(self):
        # EVALUATION TEST SET
        title_test_set = str(self._entity).upper() + " test set"

        self._model.precomputed_features = False
        x_scaled_test, y_scaled_test = self._scaler.fit_transform(x=self._x_test, y=[self._eng_test, self._grads_test])
        y_pred_test = self._model.predict(x_scaled_test)
        x_pred_test, y_pred_test = self._scaler.inverse_transform(x=x_scaled_test, y=y_pred_test)

        _, mae_test, r2_test = calc_final_evaluation(map_shape_output_to_flat(y_pred_test),
                                                     map_shape_output_to_flat([self._eng_test, self._grads_test]),
                                                     title_test_set, self._entity + "_test" + properties.prediction_image_suffix)

        # EVALUATION TRAINING SET
        title_training_set = str(self._entity).upper() + " training set"

        self._model.precomputed_features = False
        x_scaled, y_scaled = self._scaler.fit_transform(x=self._x, y=[self._eng, self._grads])
        y_pred = self._model.predict(x_scaled)
        x_pred, y_pred = self._scaler.inverse_transform(x=x_scaled, y=y_pred)

        len_train, mae_train, r2_train = calc_final_evaluation(map_shape_output_to_flat(y_pred),
                                                               map_shape_output_to_flat([self._eng, self._grads]),
                                                               title_training_set, self._entity + "_train" + properties.prediction_image_suffix)

        return len_train, mae_test, r2_test, mae_train, r2_train
