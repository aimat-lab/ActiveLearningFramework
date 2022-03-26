RUN_NUMBER = str(0)

test_set_size = 256
sl_training_params = {
    "min_epochs": 100,
    "thr": 0.6,
    "max_epochs": 2000,
    "batch_size": 32
}
al_training_params = {
    "amount_internal_models": 2,
    "batch_size": 16,
    "min_epochs": 20,
    "thr": 0.6,
    "max_epochs": 100,
    "initial_min_epochs": 0,
    "initial_thr": 2,
    "initial_max_epochs": 50,
    "initial_set_size": 16
}
al_mae_thr = 0.5
min_al_n = 0  # minimum amount of al training iterations

_data_location_prefix = "example_implementation/butene_data"
data_location = {
    "x": f"{_data_location_prefix}/butene_x.npy",
    "energy": f"{_data_location_prefix}/butene_energy.npy",
    "force": f"{_data_location_prefix}/butene_force.npy"
}

models_storage_location = "assets/saved_models/pbs/" + RUN_NUMBER
models_storage_suffix = "__weights.h5"
al_training_data_storage_location = "assets/saved_models/pbs/" + RUN_NUMBER + "/data"
al_training_data_storage_x = "training_x.npy"
al_training_data_storage_y = "training_y.npy"

_results_location_prefix = "assets/evaluation/pbs/" + RUN_NUMBER
results_location = {
    "prediction_image": f"{_results_location_prefix}/preds_graph/",  # plot of energy
    "loss_over_epochs": f"{_results_location_prefix}/loss_history/",  # loss development per epoch
    "active_metrics_over_iterations": f"{_results_location_prefix}/metrics_history/"  # mae and r2 development for test set
}
prediction_image_suffix = "__pred.png"
loss_history_suffix = "__loss.npy"
mae_history_suffix = "__mae.npy"
r2_history_suffix = "__r2.npy"

entities = {
    "up": "up", "ia": "ia", "ip": "ip", "ua": "ua"
}
