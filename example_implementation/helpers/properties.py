RUN_NUMBER = str(0)

model_storage_location = f"assets/saved_models/sbs_housing/{RUN_NUMBER}"
model_storage_suffix = "__weights.h5"
al_training_data_storage_location = f"assets/saved_models/sbs_housing/{RUN_NUMBER}/data"
al_training_data_storage_x = "training_x.npy"
al_training_data_storage_y = "training_y.npy"

_results_location_prefix = "assets/evaluation/sbs_housing/" + RUN_NUMBER
results_location = {
    "prediction_image": f"{_results_location_prefix}/preds_graph/",  # plot of energy
    "loss_over_epochs": f"{_results_location_prefix}/loss_history/",  # loss development per epoch
    "active_metrics_over_iterations": f"{_results_location_prefix}/metrics_history/"  # mae and r2 development for test set
}
prediction_image_suffix = "__pred.png"
loss_history_suffix = "__loss.npy"
mae_history_suffix = "__mae.npy"
r2_history_suffix = "__r2.npy"

test_set_size = 512

al_training_params = {
    "amount_internal_models": 2,
    "initial_batch_size": 32,
    "initial_max_epochs": 25,
    "initial_min_epochs": 0,
    "initial_thr": 0,
    "batch_size": 32,
    "max_epochs": 300,
    "min_epochs": 0,
    "thr": 0.5
}

al_mae_thr = 1.7
min_al_n = 0  # minimum amount of al training iterations

sl_training_params = {
    "min_epochs": 100,
    "thr": 0.6,
    "max_epochs": 2000,
    "batch_size": 32
}

eval_entities = {
    "ia": "ia", "ip": "ip", "ua": "ua", "up": "up"
}
