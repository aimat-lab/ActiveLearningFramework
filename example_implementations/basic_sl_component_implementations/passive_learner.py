import logging
import os
from typing import Tuple, Optional, Sequence

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback

from basic_sl_component_interfaces import PassiveLearner
from example_implementations.evaluation.metrics import metrics_set
from example_implementations.helpers.mapper import map_shape_output_to_flat, map_flat_input_to_shape, map_flat_output_to_shape, map_shape_input_to_flat
from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel
from example_implementations.pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from example_implementations.pyNNsMD.utils.loss import ScaledMeanAbsoluteError
from helpers import X, Y, AddInfo_Y

RUN_NUMBER = 2
model_location = "assets/saved_models/pbs/" + RUN_NUMBER
weight_file_name = {
    "a": 'weights_a.h5',
    "b": 'weights_b.h5',
    "c": 'weights_c.h5'
}
save_results_location = "assets/evaluation/pbs/" + RUN_NUMBER
result_file_name = {
    "mae_test": 'result_mae_test.npy',
    "mae_train": 'result_mae_train.npy',
    "r2_test": 'result_r2_test.npy',
    "r2_train": 'result_r2_train.npy',
    "train_data_x": 'train_data_x_flat.npy',
    "train_data_y": 'train_data_y_flat.npy',
    "test_data_x": 'test_data_x_flat.npy',
    "test_data_y": 'test_data_y_flat.npy'
}


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


class ButenePassiveLearner(PassiveLearner):

    def __init__(self, x_test, eng_test, grads_test):
        self.model_a, self.scaler_a = _create_model_and_scaler()
        self.model_b, self.scaler_b = _create_model_and_scaler()
        self.model_c, self.scaler_c = _create_model_and_scaler()

        self.x_train, self.y_train = np.array([]), np.array([])
        self.x_test, self.eng_test, self.grads_test = x_test, eng_test, grads_test
        self.mae_train_history, self.r2_train_history, self.mae_test_history, self.r2_test_history = [], [], [], []

    def save_results(self):
        filename = os.path.abspath(os.path.abspath(save_results_location))
        os.makedirs(filename, exist_ok=True)
        np.save(os.path.join(filename, result_file_name["test_data_x"]), np.asarray(map_shape_input_to_flat(self.x_test)))
        np.save(os.path.join(filename, result_file_name["test_data_y"]), np.asarray(map_shape_output_to_flat([self.eng_test, self.grads_test])))
        np.save(os.path.join(filename, result_file_name["train_data_x"]), np.asarray(self.x_train))
        np.save(os.path.join(filename, result_file_name["train_data_y"]), np.asarray(self.y_train))
        np.save(os.path.join(filename, result_file_name["mae_train"]), np.asarray(self.mae_train_history))
        np.save(os.path.join(filename, result_file_name["mae_test"]), np.asarray(self.mae_test_history))
        np.save(os.path.join(filename, result_file_name["r2_train"]), np.asarray(self.r2_train_history))
        np.save(os.path.join(filename, result_file_name["r2_test"]), np.asarray(self.r2_test_history))

    def initial_training(self, x_train: Sequence[X], y_train: Sequence[Y]) -> None:
        self.x_train = x_train
        self.y_train = y_train

        # initial evaluation
        x_test, y_test = map_shape_input_to_flat(self.x_test), map_shape_output_to_flat([self.eng_test, self.grads_test])
        x_training, y_training = self.x_train, self.y_train

        pred_test = self.predict_set(x_test)[0]
        pred_training = self.predict_set(x_training)[0]

        mae_test, r2_test = metrics_set(y_test, pred_test)
        mae_training, r2_training = metrics_set(y_training, pred_training)

        self.mae_test_history.append(mae_test)
        self.r2_test_history.append(r2_test)
        self.mae_train_history.append(mae_training)
        self.r2_train_history.append(r2_training)

        logging.info(f"mae train: {self.mae_train_history}")
        logging.info(f"r2 train: {self.r2_train_history}")
        logging.info(f"mae test: {self.mae_test_history}")
        logging.info(f"r2 test: {self.r2_test_history}")

        # actual initial training
        max_epochs, thr, min_epochs = 2000, 1, 20

        self.model_a.precomputed_features = True
        x_scaled_a, y_scaled_a = self.scaler_a.fit_transform(x=map_flat_input_to_shape(x_train), y=map_flat_output_to_shape(y_train))
        feat_x, feat_grad = self.model_a.precompute_feature_in_chunks(x_scaled_a, batch_size=4)
        self.model_a.set_const_normalization_from_features(feat_x)
        self.model_a.fit(x=[feat_x, feat_grad], y=y_scaled_a, batch_size=4, epochs=max_epochs, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epochs)])
        self.model_a.precomputed_features = False

        self.model_b.precomputed_features = True
        x_scaled_b, y_scaled_b = self.scaler_b.fit_transform(x=map_flat_input_to_shape(x_train), y=map_flat_output_to_shape(y_train))
        feat_x, feat_grad = self.model_b.precompute_feature_in_chunks(x_scaled_b, batch_size=4)
        self.model_b.set_const_normalization_from_features(feat_x)
        self.model_b.fit(x=[feat_x, feat_grad], y=y_scaled_b, batch_size=4, epochs=max_epochs, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epochs)])
        self.model_b.precomputed_features = False

        self.model_c.precomputed_features = True
        x_scaled_c, y_scaled_c = self.scaler_c.fit_transform(x=map_flat_input_to_shape(x_train), y=map_flat_output_to_shape(y_train))
        feat_x, feat_grad = self.model_c.precompute_feature_in_chunks(x_scaled_c, batch_size=4)
        self.model_c.set_const_normalization_from_features(feat_x)
        self.model_c.fit(x=[feat_x, feat_grad], y=y_scaled_c, batch_size=4, epochs=max_epochs, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epochs)])
        self.model_c.precomputed_features = False

        self.save_results()

    def load_model(self) -> None:
        filename = os.path.abspath(os.path.abspath(model_location))
        os.makedirs(filename, exist_ok=True)

        self.model_a: EnergyGradientModel = _create_model_and_scaler()[0]
        self.model_a.load_weights(os.path.join(filename, weight_file_name["a"]))

        self.model_b: EnergyGradientModel = _create_model_and_scaler()[0]
        self.model_b.load_weights(os.path.join(filename, weight_file_name["b"]))

        self.model_c: EnergyGradientModel = _create_model_and_scaler()[0]
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
        x_scaled_a, _ = self.scaler_a.transform(x=map_flat_input_to_shape(np.array([x])), y=map_flat_output_to_shape(np.array([np.zeros(2 + (2 * 12 * 3))])))
        x_scaled_b, _ = self.scaler_b.transform(x=map_flat_input_to_shape(np.array([x])), y=map_flat_output_to_shape(np.array([np.zeros(2 + (2 * 12 * 3))])))
        x_scaled_c, _ = self.scaler_c.transform(x=map_flat_input_to_shape(np.array([x])), y=map_flat_output_to_shape(np.array([np.zeros(2 + (2 * 12 * 3))])))

        prediction_scaled_a = self.model_a.predict(x_scaled_a)
        prediction_scaled_b = self.model_b.predict(x_scaled_b)
        prediction_scaled_c = self.model_c.predict(x_scaled_c)

        _, prediction_a = self.scaler_a.inverse_transform(x=x_scaled_a, y=prediction_scaled_a)
        _, prediction_b = self.scaler_b.inverse_transform(x=x_scaled_b, y=prediction_scaled_b)
        _, prediction_c = self.scaler_c.inverse_transform(x=x_scaled_c, y=prediction_scaled_c)

        y_a, y_b, y_c = map_shape_output_to_flat(prediction_a)[0], map_shape_output_to_flat(prediction_b)[0], map_shape_output_to_flat(prediction_c)[0]
        return np.mean(np.array([y_a, y_b, y_c]), axis=0), np.var(np.array([y_a, y_b, y_c]), axis=0)

    def predict_set(self, xs: Sequence[X]) -> Tuple[Sequence[Y], Sequence[AddInfo_Y]]:
        x_scaled_a, _ = self.scaler_a.transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(np.zeros([len(xs), 2 + (2 * 12 * 3)])))
        x_scaled_b, _ = self.scaler_b.transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(np.zeros([len(xs), 2 + (2 * 12 * 3)])))
        x_scaled_c, _ = self.scaler_c.transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(np.zeros([len(xs), 2 + (2 * 12 * 3)])))

        prediction_scaled_a = self.model_a.predict(x_scaled_a)
        prediction_scaled_b = self.model_b.predict(x_scaled_b)
        prediction_scaled_c = self.model_c.predict(x_scaled_c)

        _, prediction_a = self.scaler_a.inverse_transform(x=x_scaled_a, y=prediction_scaled_a)
        _, prediction_b = self.scaler_b.inverse_transform(x=x_scaled_b, y=prediction_scaled_b)
        _, prediction_c = self.scaler_c.inverse_transform(x=x_scaled_c, y=prediction_scaled_c)

        y_a, y_b, y_c = map_shape_output_to_flat(prediction_a), map_shape_output_to_flat(prediction_b), map_shape_output_to_flat(prediction_c)
        return np.mean(np.array([y_a, y_b, y_c]), axis=0), np.var(np.array([y_a, y_b, y_c]), axis=0)

    def train(self, x: X, y: Y) -> None:
        if len(self.x_train) == 0:
            self.x_train = np.array([x])
            self.y_train = np.array([y])
        else:
            self.x_train = np.append(self.x_train, [x], axis=0)
            self.y_train = np.append(self.y_train, [y], axis=0)

        batch_size = 8
        print(f"TRAINING SIZE of passive learner (sl model): x_size = {len(self.x_train)}, y_size = {len(self.y_train)}")
        if len(self.x_train) % batch_size == 0:
            self.train_batch_early_stopping(self.x_train, self.y_train, batch_size, 0.7, 20, 2000)

        self.save_results()

    def train_batch_early_stopping(self, xs: Sequence[X], ys: Sequence[Y], batch_size: int, thr, min_epoch, max_epoch):
        self.model_a.precomputed_features = True
        x_scaled_a, y_scaled_a = self.scaler_a.fit_transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(ys))
        feat_x, feat_grad = self.model_a.precompute_feature_in_chunks(x=x_scaled_a, batch_size=batch_size)
        self.model_a.set_const_normalization_from_features(feat_x)
        self.model_a.fit(x=[feat_x, feat_grad], y=[y_scaled_a[0], y_scaled_a[1]], batch_size=batch_size, epochs=max_epoch, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epoch)])
        self.model_a.precomputed_features = False

        self.model_b.precomputed_features = True
        x_scaled_b, y_scaled_b = self.scaler_b.fit_transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(ys))
        feat_x, feat_grad = self.model_b.precompute_feature_in_chunks(x=x_scaled_b, batch_size=batch_size)
        self.model_b.set_const_normalization_from_features(feat_x)
        self.model_b.fit(x=[feat_x, feat_grad], y=[y_scaled_b[0], y_scaled_b[1]], batch_size=batch_size, epochs=max_epoch, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epoch)])
        self.model_b.precomputed_features = False

        self.model_c.precomputed_features = True
        x_scaled_c, y_scaled_c = self.scaler_c.fit_transform(x=map_flat_input_to_shape(xs), y=map_flat_output_to_shape(ys))
        feat_x, feat_grad = self.model_c.precompute_feature_in_chunks(x=x_scaled_c, batch_size=batch_size)
        self.model_c.set_const_normalization_from_features(feat_x)
        self.model_c.fit(x=[feat_x, feat_grad], y=[y_scaled_c[0], y_scaled_c[1]], batch_size=batch_size, epochs=max_epoch, verbose=2, callbacks=[CallbackStopIfLossLow(thr=thr, min_epoch=min_epoch)])
        self.model_c.precomputed_features = False

        x_test, y_test = map_shape_input_to_flat(self.x_test), map_shape_output_to_flat([self.eng_test, self.grads_test])
        x_training, y_training = self.x_train, self.y_train

        pred_test = self.predict_set(x_test)[0]
        pred_training = self.predict_set(x_training)[0]

        mae_test, r2_test = metrics_set(y_test, pred_test)
        mae_training, r2_training = metrics_set(y_training, pred_training)

        self.mae_test_history.append(mae_test)
        self.r2_test_history.append(r2_test)
        self.mae_train_history.append(mae_training)
        self.r2_train_history.append(r2_training)

        logging.info(f"mae train: {self.mae_train_history}")
        logging.info(f"r2 train: {self.r2_train_history}")
        logging.info(f"mae test: {self.mae_test_history}")
        logging.info(f"r2 test: {self.r2_test_history}")

    def sl_model_satisfies_evaluation(self) -> bool:
        return (not len(self.mae_test_history) == 0) and self.mae_test_history[-1] < 1
