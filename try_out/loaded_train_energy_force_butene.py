import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pyNNsMD.plots.pred import plot_scatter_prediction
from example_implementations.pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError

from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel

model_location = "assets/saved_models/pbs/"
weight_file_name = {
    "a": '2__weights_a.h5',
    "b": '2__weights_b.h5',
    "c": '2__weights_c.h5'
}


def _create_model(scaler):
    model = EnergyGradientModel(atoms=12, states=2, invd_index=True)
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    mae_energy = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
    mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
    mae_energy.set_scale(scaler.energy_std)
    mae_force.set_scale(scaler.gradient_std)
    model.compile(optimizer=optimizer,
                  loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
                  metrics=[[mae_energy], [mae_force]])
    return model


def load():
    # Load data
    x = np.load("example_implementations/butene_data/butene_x.npy")
    eng = np.load("example_implementations/butene_data/butene_energy.npy")
    grads = np.load("example_implementations/butene_data/butene_force.npy")
    print(x.shape, eng.shape, grads.shape)

    # Scale in- and output
    # Important: x, energy and gradients can not be scaled completely independent!!
    scaler = EnergyGradientStandardScaler()

    filename = os.path.abspath(os.path.abspath(model_location))
    os.makedirs(filename, exist_ok=True)

    model_a: EnergyGradientModel = _create_model(scaler)
    model_a.load_weights(os.path.join(filename, weight_file_name["a"]))
    model_b: EnergyGradientModel = _create_model(scaler)
    model_b.load_weights(os.path.join(filename, weight_file_name["b"]))
    model_c: EnergyGradientModel = _create_model(scaler)
    model_c.load_weights(os.path.join(filename, weight_file_name["c"]))

    x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
    scaler.print_params_info()

    # Precompute features plus derivative
    # Features are normalized automatically
    model_a.precomputed_features = True
    model_b.precomputed_features = True
    model_c.precomputed_features = True
    feat_x, feat_grad = model_a.precompute_feature_in_chunks(x_scaled, batch_size=32)
    model_a.set_const_normalization_from_features(feat_x)
    model_b.set_const_normalization_from_features(feat_x)
    model_c.set_const_normalization_from_features(feat_x)

    # Now set the model to coordinates and predict the test data
    model_a.precomputed_features = False
    y_pred_a = model_a.predict(x_scaled)
    model_b.precomputed_features = False
    y_pred_b = model_b.predict(x_scaled)
    model_c.precomputed_features = False
    y_pred_c = model_c.predict(x_scaled)

    # invert standardization
    _, y_pred_a = scaler.inverse_transform(x=x_scaled, y=y_pred_a)
    _, y_pred_b = scaler.inverse_transform(x=x_scaled, y=y_pred_b)
    x_pred, y_pred_c = scaler.inverse_transform(x=x_scaled, y=y_pred_c)

    y_pred = np.mean(np.array([y_pred_a[0], y_pred_b[0], y_pred_c[0]]), axis=0)

    y_pred_flat = np.array([np.append(y_pred[0][i], y_pred[1][i].flatten()) for i in range(len(y_pred[0]))])
    y_flat = np.array([np.append(eng[i], grads[i].flatten()) for i in range(len(eng))])
    mae = np.mean(np.array(
        [np.mean(np.abs(y_pred_flat[i] - y_flat[i])) for i in range(len(y_pred_flat))]
    ))
    rsquared = np.mean(np.array(
        [(np.corrcoef(y_flat[i], y_pred_flat[i])[0, 1]) ** 2 for i in range(len(y_pred_flat))]
    ))
    print(f"MAE: {mae}, r-squared: {rsquared}")

    # Plot Prediction
    fig = plot_scatter_prediction(eng, y_pred[0])
    plt.show()

    # Plot Prediction
    fig = plot_scatter_prediction(eng, np.mean(np.array([y_pred_a[0], y_pred_b[0], y_pred_c[0]]), axis=0))
    plt.show()
