import os

import numpy as np

from example_implementations.helpers.mapper import map_shape_output_to_flat
from new_example_implementation.helpers import properties
from new_example_implementation.helpers.callbacks import CallbackDocumentation, CallbackStopIfLossLow
from new_example_implementation.helpers.creator_methods import create_scaler, create_model
from new_example_implementation.helpers.mapper import map_flat_input_to_shape, map_flat_output_to_shape
from new_example_implementation.helpers.metrics import calc_final_evaluation


class SLModel:

    def __init__(self, x_test, y_test, entity=properties.entities["up"]):
        self._entity = entity

        self._scaler = create_scaler()
        self._model = create_model(self._scaler)

        self._x_test, self._y_test = x_test, y_test
        self._x, self._y = np.array([]), np.array([])

        # create store for loss history, final predictions
        filename = os.path.abspath(os.path.abspath(properties.results_location["loss_over_epochs"]))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, entity + properties.loss_history_suffix), np.asarray([]))
        filename = os.path.abspath(os.path.abspath(properties.results_location["prediction_image"]))
        os.makedirs(filename, exist_ok=True)

    def train(self, x, y, batch_size, max_epochs, min_epochs, thr):
        x_scaled, y_scaled = self._scaler.fit_transform(x=map_flat_input_to_shape(x), y=map_flat_output_to_shape(y))

        self._model.precomputed_features = True
        feat_x, feat_grad = self._model.precompute_feature_in_chunks(x_scaled, batch_size=properties.al_training_params["initial_set_size"])
        self._model.set_const_normalization_from_features(feat_x)

        self._model.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]],
                        batch_size=batch_size, epochs=max_epochs, verbose=2,
                        callbacks=[CallbackStopIfLossLow(min_epoch=min_epochs, thr=thr), CallbackDocumentation(entity=self._entity)])

        self._model.precomputed_features = False

        self._x, self._y = x, y

    def evaluate(self, x_test, y_test, x, y):
        # EVALUATION TEST SET
        title_test_set = str(self._entity).upper() + " test set"

        self._model.precomputed_features = False
        x_scaled_test, y_scaled_test = self._scaler.fit_transform(x=map_flat_input_to_shape(x_test), y=map_flat_output_to_shape(y_test))
        y_pred_test = self._model.predict(x_scaled_test)
        x_pred_test, y_pred_test = self._scaler.inverse_transform(x=x_scaled_test, y=y_pred_test)

        _, mae_test, r2_test = calc_final_evaluation(map_shape_output_to_flat(y_pred_test), y_test,
                                                     title_test_set, self._entity + "_test" + properties.prediction_image_suffix)

        # EVALUATION TRAINING SET
        title_training_set = str(self._entity).upper() + " training set"

        self._model.precomputed_features = False
        x_scaled, y_scaled = self._scaler.fit_transform(x=map_flat_input_to_shape(x), y=map_flat_output_to_shape(y))
        y_pred = self._model.predict(x_scaled)
        x_pred, y_pred = self._scaler.inverse_transform(x=x_scaled, y=y_pred)

        len_train, mae_train, r2_train = calc_final_evaluation(map_shape_output_to_flat(y_pred), y,
                                                               title_training_set, self._entity + "_train" + properties.prediction_image_suffix)

        return len_train, mae_test, r2_test, mae_train, r2_train
