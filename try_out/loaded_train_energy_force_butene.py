import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pyNNsMD.plots.pred import plot_scatter_prediction
from example_implementations.pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError

from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel

model_location = "assets/saved_models/"
weight_file_name = {
    "a": 'weights_a.h5',
    "b": 'weights_b.h5'
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

    model: EnergyGradientModel = _create_model(scaler)
    model.load_weights(os.path.join(filename, weight_file_name["a"]))

    x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
    scaler.print_params_info()

    # Precompute features plus derivative
    # Features are normalized automatically
    model.precomputed_features = True
    feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=32)
    model.set_const_normalization_from_features(feat_x)
    print("Feature norm: ", model.get_layer('feat_std').get_weights())

    # Now set the model to coordinates and predict the test data
    model.precomputed_features = False
    y_pred = model.predict(x_scaled[2000:])

    # invert standardization
    x_pred, y_pred = scaler.inverse_transform(x=x_scaled[2000:], y=y_pred)

    # Plot Prediction
    fig = plot_scatter_prediction(eng[2000:], y_pred[0])
    plt.show()
