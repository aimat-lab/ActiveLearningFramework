import tensorflow as tf
from pyNNsMD.models.mlp_eg import EnergyGradientModel

from pyNNsMD.scaler.energy import EnergyGradientStandardScaler
from pyNNsMD.utils.loss import ScaledMeanAbsoluteError


def create_scaler():
    return EnergyGradientStandardScaler()


def create_model(scaler):
    model = EnergyGradientModel(atoms=6, states=1, invd_index=True)

    lr = 5e-4
    optimizer = tf.keras.optimizers.Adam(lr)
    mae_energy = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
    mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
    mae_energy.set_scale(scaler.energy_std)
    mae_force.set_scale(scaler.gradient_std)

    model.compile(optimizer=optimizer, loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 5], metrics=[[mae_energy], [mae_force]])

    return model
