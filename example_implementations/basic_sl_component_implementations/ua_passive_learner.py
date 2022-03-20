import logging
import os
from typing import Tuple, Optional, Sequence

import numpy as np

from basic_sl_component_interfaces import PassiveLearner
from example_implementations.helpers.metrics import metrics_set
from example_implementations.helpers import properties
from example_implementations.helpers.callbacks import CallbackStopIfLossLow, CallbackDocumentation
from example_implementations.helpers.mapper import map_shape_output_to_flat, map_flat_input_to_shape, map_flat_output_to_shape
from example_implementations.helpers.model_creator import create_model_and_scaler
from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel
from helpers import X, Y, AddInfo_Y

_internal_models = ["a", "b", "c"]


class UAButenePassiveLearner(PassiveLearner):

    def __init__(self, x_test, y_test):
        self.models, self.scaler = [], []
        for _ in _internal_models:
            new_model, new_scaler = create_model_and_scaler()
            self.models.append(new_model), self.scaler.append(new_scaler)

        self._x_train, self._y_train = np.array([]), np.array([])
        self._x_test, self._y_test = x_test, y_test
        self._mae_train_history, self._r2_train_history, self._mae_test_history, self._r2_test_history = [], [], [], []

        filename = os.path.abspath(os.path.abspath(properties.results_location["active_metrics_over_iterations"]))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, properties.entities["ua"] + "_train" + properties.mae_history_suffix), np.asarray([]))
        np.save(os.path.join(filename, properties.entities["ua"] + "_test" + properties.mae_history_suffix), np.asarray([]))
        np.save(os.path.join(filename, properties.entities["ua"] + "_train" + properties.r2_history_suffix), np.asarray([]))
        np.save(os.path.join(filename, properties.entities["ua"] + "_test" + properties.r2_history_suffix), np.asarray([]))

        filename = os.path.abspath(os.path.abspath(properties.results_location["loss_over_epochs"]))
        os.makedirs(filename, exist_ok=True)
        for i in range(len(_internal_models)):
            np.save(os.path.join(filename, properties.entities["ua"] + "_" + _internal_models[i] + properties.loss_history_suffix), np.asarray([]))

    def save_results(self):
        filename = os.path.abspath(os.path.abspath(properties.results_location["active_metrics_over_iterations"]))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, properties.entities["ua"] + "_train" + properties.mae_history_suffix), np.asarray(self._mae_train_history))
        np.save(os.path.join(filename, properties.entities["ua"] + "_test" + properties.mae_history_suffix), np.asarray(self._mae_test_history))
        np.save(os.path.join(filename, properties.entities["ua"] + "_train" + properties.r2_history_suffix), np.asarray(self._r2_train_history))
        np.save(os.path.join(filename, properties.entities["ua"] + "_test" + properties.r2_history_suffix), np.asarray(self._r2_test_history))

        filename = os.path.abspath(os.path.abspath(properties.al_training_data_storage_location))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, properties.al_training_data_storage_x), np.asarray(self._x_train))
        np.save(os.path.join(filename, properties.al_training_data_storage_y), np.asarray(self._y_train))

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y]) -> None:
        self._x_train = x_train
        self._y_train = y_train

        for i in range(len(_internal_models)):
            self.models[i].precomputed_features = True
            x_scaled, y_scaled = self.scaler[i].fit_transform(x=map_flat_input_to_shape(x_train), y=map_flat_output_to_shape(y_train))
            feat_x, feat_grad = self.models[i].precompute_feature_in_chunks(x_scaled, batch_size=4)
            self.models[i].set_const_normalization_from_features(feat_x)
            self.models[i].fit(x=[feat_x, feat_grad], y=y_scaled, batch_size=4, epochs=properties.al_training_params["max_epochs"], verbose=2, callbacks=[CallbackStopIfLossLow(thr=1, min_epoch=properties.al_training_params["min_epochs"]), CallbackDocumentation(entity=properties.entities["ua"] + "_" + _internal_models[i])])
            self.models[i].precomputed_features = False

    def load_model(self) -> None:
        filename = os.path.abspath(os.path.abspath(properties.models_storage_location))
        os.makedirs(filename, exist_ok=True)

        for i in range(len(_internal_models)):
            self.models[i]: EnergyGradientModel = create_model_and_scaler()[0]
            self.models[i].load_weights(os.path.join(filename, _internal_models[i] + properties.models_storage_suffix))

    def close_model(self) -> None:
        self.models = [None, None, None]

    def save_model(self) -> None:
        # Folder to store model in
        filename = os.path.abspath(os.path.abspath(properties.models_storage_location))
        os.makedirs(filename, exist_ok=True)

        for i in range(len(_internal_models)):
            self.models[i].save_weights(os.path.join(filename, _internal_models[i] + properties.models_storage_suffix))

        self.models = [None, None, None]

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        ys = []

        for i in range(len(_internal_models)):
            x_scaled, _ = self.scaler[i].transform(x=map_flat_input_to_shape(np.array([x])), y=map_flat_output_to_shape(np.array([np.zeros(2 + (2 * 12 * 3))])))
            prediction_scaled = self.models[i].predict(x_scaled)
            _, prediction = self.scaler[i].inverse_transform(x=x_scaled, y=prediction_scaled)
            ys.append(map_shape_output_to_flat(prediction)[0])

        return np.mean(np.array(ys), axis=0), np.var(np.array(ys), axis=0)

    def predict_set(self, xs: Sequence[X]) -> Tuple[Sequence[Y], Sequence[AddInfo_Y]]:
        ys = []

        for i in range(len(_internal_models)):
            x_scaled, _ = self.scaler[i].transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(np.zeros([len(xs), 2 + (2 * 12 * 3)])))
            prediction_scaled = self.models[i].predict(x_scaled)
            _, prediction = self.scaler[i].inverse_transform(x=x_scaled, y=prediction_scaled)
            ys.append(map_shape_output_to_flat(prediction))

        return np.mean(np.array(ys), axis=0), np.var(np.array(ys), axis=0)

    def train(self, x: X, y: Y) -> None:
        if len(self._x_train) == 0:
            self._x_train = np.array([x])
            self._y_train = np.array([y])
        else:
            self._x_train = np.append(self._x_train, [x], axis=0)
            self._y_train = np.append(self._y_train, [y], axis=0)

        batch_size = 8
        print(f"TRAINING SIZE of passive learner (sl model): x_size = {len(self._x_train)}, y_size = {len(self._y_train)}")
        if len(self._x_train) % batch_size == 0:
            self.train_batch_early_stopping(self._x_train, self._y_train, batch_size, properties.al_training_params["thr"], properties.al_training_params["min_epochs"], properties.al_training_params["max_epochs"])

        self.save_results()

    def train_batch_early_stopping(self, xs: Sequence[X], ys: Sequence[Y], batch_size: int, thr, min_epoch, max_epoch):
        for i in range(len(_internal_models)):
            self.models[i].precomputed_features = True
            x_scaled, y_scaled = self.scaler[i].fit_transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(ys))
            feat_x, feat_grad = self.models[i].precompute_feature_in_chunks(x=x_scaled, batch_size=batch_size)
            self.models[i].set_const_normalization_from_features(feat_x)
            self.models[i].fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=batch_size, epochs=max_epoch, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epoch)])
            self.models[i].precomputed_features = False

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

    def sl_model_satisfies_evaluation(self) -> bool:
        return len(self._mae_test_history) > properties.min_al_n and self._mae_test_history[-1] < properties.al_mae_thr
