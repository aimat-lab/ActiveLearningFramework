import numpy as np
import tensorflow as tf

from example_implementations.pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError


def store():
    # Load data
    x = np.load("example_implementations/butene_data/butene_x.npy")
    eng = np.load("example_implementations/butene_data/butene_energy.npy")
    grads = np.load("example_implementations/butene_data/butene_force.npy")
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
    model.compile(optimizer=optimizer,
                  loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5],
                  metrics=[[mae_energy], [mae_force]])

    x_scaled, y_scaled = scaler.fit_transform(x=x, y=[eng, grads])
    scaler.print_params_info()

    # Precompute features plus derivative
    # Features are normalized automatically
    model.precomputed_features = True
    feat_x, feat_grad = model.precompute_feature_in_chunks(x_scaled, batch_size=32)
    model.set_const_normalization_from_features(feat_x)
    print("Feature norm: ", model.get_layer('feat_std').get_weights())

    # fit with precomputed features and normalized energies, gradients
    model.fit(x=[feat_x[:2000], feat_grad[:2000]], y=[y_scaled[0][:2000], y_scaled[1][:2000]],
              batch_size=32, epochs=20, verbose=2)

    # Save Model
    model.save("assets/saved_models/a_butene_energy_force")
    del model
