import logging
import os
from typing import Sequence, Optional, Tuple

import numpy as np
from pyNNsMD.models.mlp_eg import EnergyGradientModel

from basic_sl_component_interfaces import PassiveLearner
from helpers import X, Y, AddInfo_Y
from new_example_implementation.helpers import properties
from new_example_implementation.helpers.callbacks import CallbackStopIfLossLow, CallbackDocumentation
from new_example_implementation.helpers.creator_methods import create_scaler, create_model
from new_example_implementation.helpers.mapper import map_flat_input_to_shape, map_flat_output_to_shape, map_shape_output_to_flat
from new_example_implementation.helpers.metrics import metrics_set


class ButenePassiveLearner(PassiveLearner):

    def __init__(self, x_test, y_test, eval_entity):
        self._entity = eval_entity

        self._scaler = create_scaler()
        self._models: Sequence[EnergyGradientModel] = [create_model(self._scaler) for _ in range(properties.al_training_params["amount_internal_models"])]

        self._x_train, self._y_train = np.array([]), np.array([])
        self._x_test, self._y_test = x_test, y_test

        self._mae_train_history, self._r2_train_history, self._mae_test_history, self._r2_test_history = [], [], [], []

        filename = os.path.abspath(os.path.abspath(properties.results_location["active_metrics_over_iterations"]))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, self._entity + "_train" + properties.mae_history_suffix), np.asarray([]))
        np.save(os.path.join(filename, self._entity + "_test" + properties.mae_history_suffix), np.asarray([]))
        np.save(os.path.join(filename, self._entity + "_train" + properties.r2_history_suffix), np.asarray([]))
        np.save(os.path.join(filename, self._entity + "_test" + properties.r2_history_suffix), np.asarray([]))

        filename = os.path.abspath(os.path.abspath(properties.results_location["loss_over_epochs"]))
        os.makedirs(filename, exist_ok=True)
        for i in range(properties.al_training_params["amount_internal_models"]):
            np.save(os.path.join(filename, eval_entity + "_" + str(i) + properties.loss_history_suffix), np.asarray([]))

    def _batch_training(self, x, y, batch_size, max_epochs, min_epochs, thr):
        x_scaled, y_scaled = self._scaler.fit_transform(x=map_flat_input_to_shape(x), y=map_flat_output_to_shape(y))

        for i, model in enumerate(self._models):
            model.precomputed_features = True
            feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=properties.al_training_params["initial_set_size"])
            model.set_const_normalization_from_features(feat_x)

            model.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]],
                      batch_size=batch_size, epochs=max_epochs, verbose=2,
                      callbacks=[CallbackStopIfLossLow(min_epoch=min_epochs, thr=thr), CallbackDocumentation(entity=self._entity + "_" + str(i))])

            model.precomputed_features = False

    def _evaluate_metrics(self):
        x_test, y_test = self._x_test, self._y_test
        x_training, y_training = self._x_train, self._y_train

        pred_test = self.predict_set(x_test)[0]
        pred_training = self.predict_set(x_training)[0]

        mae_test, r2_test = metrics_set(y_test, pred_test)
        mae_training, r2_training = metrics_set(y_training, pred_training)

        self._mae_test_history.append(mae_test)
        self._r2_test_history.append(r2_test)
        self._mae_train_history.append(mae_training)
        self._r2_train_history.append(r2_training)

        logging.info(f"mae train: {self._mae_train_history}")
        logging.info(f"r2 train: {self._r2_train_history}")
        logging.info(f"mae test: {self._mae_test_history}")
        logging.info(f"r2 test: {self._r2_test_history}")

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y]) -> None:
        self._x_train, self._y_train = x_train, y_train
        self._batch_training(x=x_train, y=y_train, batch_size=properties.al_training_params["initial_set_size"], max_epochs=properties.al_training_params["initial_max_epochs"], min_epochs=properties.al_training_params["initial_min_epochs"], thr=properties.al_training_params["initial_thr"])
        self._evaluate_metrics()

    def load_model(self) -> None:
        filename = os.path.abspath(os.path.abspath(properties.models_storage_location))
        os.makedirs(filename, exist_ok=True)

        for i in range(len(self._models)):
            self._models[i]: EnergyGradientModel = create_model(self._scaler)
            self._models[i].load_weights(os.path.join(filename, self._entity + "__" + str(i) + properties.models_storage_suffix))

    def close_model(self) -> None:
        self._models = [None, None]

    def save_model(self) -> None:
        filename = os.path.abspath(os.path.abspath(properties.models_storage_location))
        os.makedirs(filename, exist_ok=True)

        for i, model in enumerate(self._models):
            model.save_weights(os.path.join(filename, self._entity + "__" + str(i) + properties.models_storage_suffix))

        self._models = [None, None]

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        x_scaled, _ = self._scaler.transform(x=map_flat_input_to_shape(np.array([x])), y=map_flat_output_to_shape(np.array([np.zeros(2 + (2 * 12 * 3))])))

        ys = []
        for model in self._models:
            prediction_scaled = model.predict(x_scaled)
            _, prediction = self._scaler.inverse_transform(x=x_scaled, y=prediction_scaled)
            ys.append(map_shape_output_to_flat(prediction)[0])

        return np.mean(np.array(ys), axis=0), np.var(np.array(ys), axis=0)

    def predict_set(self, xs: Sequence[X]) -> Tuple[Sequence[Y], Sequence[AddInfo_Y]]:
        x_scaled, _ = self._scaler.transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(np.zeros([len(xs), 2 + (2 * 12 * 3)])))

        ys = []
        for model in self._models:
            prediction_scaled = model.predict(x_scaled)
            _, prediction = self._scaler.inverse_transform(x=x_scaled, y=prediction_scaled)
            ys.append(map_shape_output_to_flat(prediction))

        return np.mean(np.array(ys), axis=0), np.var(np.array(ys), axis=0)

    def train(self, x: X, y: Y) -> None:
        self._x_train = np.append(self._x_train, [x], axis=0)
        self._y_train = np.append(self._y_train, [y], axis=0)

        batch_size = properties.al_training_params["batch_size"]
        if len(self._x_train) % batch_size == 0:
            # self.train_batch_early_stopping(self.x_train[-batch_size:], self.y_train[-batch_size:], batch_size, 0.5, 100, 1000)
            self._batch_training(self._x_train, self._y_train, batch_size, max_epochs=properties.al_training_params["max_epochs"], min_epochs=properties.al_training_params["min_epochs"], thr=properties.al_training_params["thr"])
            self._evaluate_metrics()

        filename = os.path.abspath(os.path.abspath(properties.al_training_data_storage_location))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, properties.al_training_data_storage_x), np.asarray(self._x_train))
        np.save(os.path.join(filename, properties.al_training_data_storage_y), np.asarray(self._y_train))

    def sl_model_satisfies_evaluation(self) -> bool:
        return len(self._mae_test_history) > properties.min_al_n and self._mae_test_history[-1] < properties.al_mae_thr
