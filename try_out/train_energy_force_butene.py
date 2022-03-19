import time

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError
import matplotlib.pyplot as plt
from pyNNsMD.plots.pred import plot_scatter_prediction

from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel


class CallbackStopIfLossLow(Callback):

    def __init__(self, min_epoch, thr):
        super().__init__()
        self.thr, self.min_epoch = thr, min_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('loss') <= self.thr and epoch-1 >= self.min_epoch:
            self.model.stop_training = True


def run():
    test_set_size = 256

    t0 = time.time()

    # Load data
    x_loaded = np.load("../example_implementations/butene_data/butene_x.npy")
    random_idx = np.arange(len(x_loaded))
    np.random.shuffle(random_idx)
    x_loaded = np.array([x_loaded[i] for i in random_idx])
    x = x_loaded[test_set_size:]
    x_test = x_loaded[:test_set_size]
    eng = np.load("../example_implementations/butene_data/butene_energy.npy")
    grads = np.load("../example_implementations/butene_data/butene_force.npy")
    eng = np.array([eng[i] for i in random_idx])
    grads = np.array([grads[i] for i in random_idx])
    eng, grads = eng[test_set_size:], grads[test_set_size:]
    eng_test, grads_test = eng[:test_set_size], grads[:test_set_size]
    print(x.shape, eng.shape, grads.shape)

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

    x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
    scaler.print_params_info()

    t1 = time.time()

    # Precompute features plus derivative
    # Features are normalized automatically
    model.precomputed_features = True
    feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=32)
    model.set_const_normalization_from_features(feat_x)

    # Now set the model to coordinates and predict the test data
    model.precomputed_features = False
    x_scaled_test, y_scaled_test = scaler.fit_transform(x=x_test, y=[eng_test, grads_test])
    y_pred = model.predict(x_scaled_test)

    # invert standardization
    x_pred, y_pred = scaler.inverse_transform(x=x_scaled_test, y=y_pred)

    y_pred_flat = np.array([np.append(y_pred[0][i], y_pred[1][i].flatten()) for i in range(len(y_pred[0]))])
    y_flat = np.array([np.append(eng_test[i], grads_test[i].flatten()) for i in range(len(eng_test))])
    mae = np.mean(np.array(
        [np.mean(np.abs(y_pred_flat[i] - y_flat[i])) for i in range(len(y_pred_flat))]
    ))
    rsquared = np.mean(np.array(
        [(np.corrcoef(y_flat[i], y_pred_flat[i])[0, 1]) ** 2 for i in range(len(y_pred_flat))]
    ))
    print(f"test         AE: {mae}, r-squared: {rsquared} (BEFORE)")

    # Plot Prediction
    fig = plot_scatter_prediction(eng_test, y_pred[0], plot_title="Prediction test set BEFORE training")
    plt.show()

    # Now set the model to coordinates and predict the training data
    model.precomputed_features = False
    x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
    y_pred = model.predict(x_scaled)

    # invert standardization
    x_pred, y_pred = scaler.inverse_transform(x=x_scaled, y=y_pred)

    y_pred_flat = np.array([np.append(y_pred[0][i], y_pred[1][i].flatten()) for i in range(len(y_pred[0]))])
    y_flat = np.array([np.append(eng[i], grads[i].flatten()) for i in range(len(eng))])
    mae = np.mean(np.array(
        [np.mean(np.abs(y_pred_flat[i] - y_flat[i])) for i in range(len(y_pred_flat))]
    ))
    rsquared = np.mean(np.array(
        [(np.corrcoef(y_flat[i], y_pred_flat[i])[0, 1]) ** 2 for i in range(len(y_pred_flat))]
    ))
    print(f"training    MAE: {mae}, r-squared: {rsquared} (BEFORE)")

    # Plot Prediction
    fig = plot_scatter_prediction(eng, y_pred[0], plot_title="Prediction training set BEFORE training")
    plt.show()

    # print("Feature norm: ", model.get_layer('feat_std').get_weights())

    t2 = time.time()
    # fit with precomputed features and normalized energies, gradients
    model.fit(x=[feat_x, feat_grad], y=y_scaled, batch_size=32, epochs=2000, verbose=2, callbacks=[CallbackStopIfLossLow(thr=0.7, min_epoch=20)])
    t3 = time.time()

    print(f"\n\ntime: initialisation {t1 - t0}, training {t3 - t2}, whole {(t3 - t0) - (t2 - t1)}")
    print(f"Size training set: {len(eng)}, size test set: {test_set_size}")

    # Now set the model to coordinates and predict the test data
    model.precomputed_features = False
    x_scaled_test, y_scaled_test = scaler.fit_transform(x=x_test, y=[eng_test, grads_test])
    y_pred = model.predict(x_scaled_test)

    # invert standardization
    x_pred, y_pred = scaler.inverse_transform(x=x_scaled_test, y=y_pred)

    y_pred_flat = np.array([np.append(y_pred[0][i], y_pred[1][i].flatten()) for i in range(len(y_pred[0]))])
    y_flat = np.array([np.append(eng_test[i], grads_test[i].flatten()) for i in range(len(eng_test))])
    mae = np.mean(np.array(
        [np.mean(np.abs(y_pred_flat[i] - y_flat[i])) for i in range(len(y_pred_flat))]
    ))
    rsquared = np.mean(np.array(
        [(np.corrcoef(y_flat[i], y_pred_flat[i])[0, 1])**2 for i in range(len(y_pred_flat))]
    ))
    print(f"test         AE: {mae}, r-squared: {rsquared}")

    # Plot Prediction
    fig = plot_scatter_prediction(eng_test, y_pred[0], plot_title="Prediction test set")
    plt.show()

    # Now set the model to coordinates and predict the training data
    model.precomputed_features = False
    x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
    y_pred = model.predict(x_scaled)

    # invert standardization
    x_pred, y_pred = scaler.inverse_transform(x=x_scaled, y=y_pred)

    y_pred_flat = np.array([np.append(y_pred[0][i], y_pred[1][i].flatten()) for i in range(len(y_pred[0]))])
    y_flat = np.array([np.append(eng[i], grads[i].flatten()) for i in range(len(eng))])
    mae = np.mean(np.array(
        [np.mean(np.abs(y_pred_flat[i] - y_flat[i])) for i in range(len(y_pred_flat))]
    ))
    rsquared = np.mean(np.array(
        [(np.corrcoef(y_flat[i], y_pred_flat[i])[0, 1]) ** 2 for i in range(len(y_pred_flat))]
    ))
    print(f"training    MAE: {mae}, r-squared: {rsquared}")

    # Plot Prediction
    fig = plot_scatter_prediction(eng, y_pred[0], plot_title="Prediction training set")
    plt.show()


run()
