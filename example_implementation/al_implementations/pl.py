import logging
import os.path
from multiprocessing import Lock
from typing import Sequence, Optional, Tuple

import numpy as np
from pyNNsMD.models.mlp_eg import EnergyGradientModel

from basic_sl_component_interfaces import PassiveLearner
from example_implementation.helpers import properties
from example_implementation.helpers.callbacks import CallbackStopIfLossLow, CallbackDocumentation
from example_implementation.helpers.creators import create_scaler, create_model
from example_implementation.helpers.mapper import map_flat_input_to_shape, map_flat_output_to_shape, map_shape_output_to_flat
from example_implementation.helpers.metrics import metrics_set
from helpers import Y, X, AddInfo_Y


class MethanolPL(PassiveLearner):

    def __init__(self, lock: Lock, x_test, y_test, entity=properties.eval_entities["ia"]) -> None:
        self._entity = entity
        self._lock: Lock = lock

        self._scaler, self._models = [], []
        for i in range(properties.al_training_params["amount_internal_models"]):
            self._scaler.append(create_scaler())
            self._models.append(create_model(self._scaler[i]))

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
            np.save(os.path.join(filename, entity + "_" + str(i) + properties.loss_history_suffix), np.asarray([]))

    def load_model(self) -> None:
        filename = os.path.abspath(properties.model_storage_location)

        self._lock.acquire()
        for i in range(properties.al_training_params["amount_internal_models"]):
            self._models[i]: EnergyGradientModel = create_model(self._scaler[i])
            self._models[i].load_weights(os.path.join(filename, self._entity + "__" + str(i) + properties.model_storage_suffix))
        self._lock.release()

    def close_model(self) -> None:
        self._models = [None for _ in range(properties.al_training_params["amount_internal_models"])]

    def save_model(self) -> None:
        filename = os.path.abspath(properties.model_storage_location)
        os.makedirs(filename, exist_ok=True)

        self._lock.acquire()
        for i in range(properties.al_training_params["amount_internal_models"]):
            self._models[i].save_weights(os.path.join(filename, self._entity + "__" + str(i) + properties.model_storage_suffix))
            self._models[i] = None
        self._lock.release()

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

        filename = os.path.abspath(os.path.abspath(properties.results_location["active_metrics_over_iterations"]))
        np.save(os.path.join(filename, self._entity + "_train" + properties.mae_history_suffix), np.asarray(self._mae_train_history))
        np.save(os.path.join(filename, self._entity + "_test" + properties.mae_history_suffix), np.asarray(self._mae_test_history))
        np.save(os.path.join(filename, self._entity + "_train" + properties.r2_history_suffix), np.asarray(self._r2_train_history))
        np.save(os.path.join(filename, self._entity + "_test" + properties.r2_history_suffix), np.asarray(self._r2_test_history))

    def _train_batch(self, x, y, batch_size, max_epochs, min_epochs, thr):
        for i, model in enumerate(self._models):
            model.precomputed_features = True
            x_scaled, y_scaled = self._scaler[i].fit_transform(x=map_flat_input_to_shape(x), y=map_flat_output_to_shape(y))

            feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=batch_size)
            model.set_const_normalization_from_features(feat_x)

            model.fit(x=[feat_x, feat_grad], y=[y_scaled[0], y_scaled[1]], batch_size=batch_size, epochs=max_epochs, verbose=2,
                      callbacks=[CallbackStopIfLossLow(min_epoch=min_epochs, thr=thr), CallbackDocumentation(entity=self._entity + "_" + str(i))])
            model.precomputed_features = False

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y]) -> None:
        self._x_train, self._y_train = x_train, y_train

        self._train_batch(self._x_train, self._y_train, batch_size=properties.al_training_params["initial_batch_size"], max_epochs=properties.al_training_params["initial_max_epochs"],
                          min_epochs=properties.al_training_params["initial_min_epochs"], thr=properties.al_training_params["initial_thr"])
        self._evaluate_metrics()

    def train(self, x: X, y: Y) -> None:
        self._x_train = np.append(self._x_train, [x], axis=0)
        self._y_train = np.append(self._y_train, [y], axis=0)

        batch_size = properties.al_training_params["batch_size"]
        if len(self._x_train) % batch_size == 0:
            self._train_batch(self._x_train, self._y_train, batch_size, max_epochs=properties.al_training_params["max_epochs"],
                              min_epochs=properties.al_training_params["min_epochs"], thr=properties.al_training_params["thr"])
            self._evaluate_metrics()

        filename = os.path.abspath(os.path.abspath(properties.al_training_data_storage_location))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, properties.al_training_data_storage_x), np.asarray(self._x_train))
        np.save(os.path.join(filename, properties.al_training_data_storage_y), np.asarray(self._y_train))

    def predict(self, x: X) -> Tuple[Y, Optional[AddInfo_Y]]:
        preds, vars = self.predict_set(np.expand_dims(x, axis=0))
        return preds[0], vars[0]

    def predict_set(self, xs: Sequence[X]) -> Tuple[Sequence[Y], Sequence[AddInfo_Y]]:
        ys = []

        for i, model in enumerate(self._models):
            xs_scaled, _ = self._scaler[i].transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(np.zeros((len(xs), 1 + 1 * 6 * 3))))
            preds_scaled = model.predict(xs_scaled)
            _, preds = self._scaler[i].inverse_transform(x=xs_scaled, y=preds_scaled)
            ys.append(map_shape_output_to_flat(preds))

        return np.mean(np.array(ys), axis=0), np.var(np.array(ys), axis=0)

    def sl_model_satisfies_evaluation(self) -> bool:
        return (len(self._mae_test_history) > properties.min_al_n and self._mae_test_history[-1] < properties.al_mae_thr) or (len(self._x_train) >= 96)
