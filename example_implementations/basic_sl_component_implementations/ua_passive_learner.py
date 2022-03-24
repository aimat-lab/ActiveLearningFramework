import logging
import os
from typing import Tuple, Optional, Sequence

import numpy as np

from basic_sl_component_interfaces import PassiveLearner
from example_implementations.basic_sl_component_implementations import ButenePassiveLearner
from example_implementations.helpers.metrics import metrics_set
from example_implementations.helpers import properties
from example_implementations.helpers.callbacks import CallbackStopIfLossLow, CallbackDocumentation
from example_implementations.helpers.mapper import map_shape_output_to_flat, map_flat_input_to_shape, map_flat_output_to_shape
from example_implementations.helpers.model_creator import create_model_and_scaler
from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel
from helpers import X, Y, AddInfo_Y

_internal_models = ["a", "b"]


class UAButenePassiveLearner(ButenePassiveLearner):

    # noinspection PyMissingConstructor
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
