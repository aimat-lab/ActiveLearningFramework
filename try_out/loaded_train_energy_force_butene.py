import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.plots.pred import plot_scatter_prediction
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler


def load():
    # Load data
    x = np.load("example_implementations/butene_data/butene_x.npy")
    eng = np.load("example_implementations/butene_data/butene_energy.npy")
    grads = np.load("example_implementations/butene_data/butene_force.npy")
    print(x.shape, eng.shape, grads.shape)

    # Scale in- and output
    # Important: x, energy and gradients can not be scaled completely independent!!
    scaler = EnergyGradientStandardScaler()

    model = None
    try:
        model = tf.keras.models.load_model("assets/saved_models/a_butene_energy_force", custom_objects={"EnergyGradientModel": EnergyGradientModel}, compile=False)
    except Exception as e:
        print(e)
        raise e

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
