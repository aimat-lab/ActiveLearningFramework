import os

import numpy as np
from matplotlib import pyplot as plt

from basic_sl_component_interfaces import PassiveLearner
from example_implementations.evaluation.metrics import print_evaluation
from example_implementations.evaluation.reduced_sl_model import run_evaluation_reduced_sl_model

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


def final_evaluation_al(pl: PassiveLearner):
    pl.load_model()

    filename = os.path.abspath(os.path.abspath(save_results_location))
    x_test, y_test = np.load(os.path.join(filename, result_file_name["test_data_x"])), np.load(os.path.join(filename, result_file_name["test_data_y"]))
    x_training, y_training = np.load(os.path.join(filename, result_file_name["train_data_x"])), np.load(os.path.join(filename, result_file_name["train_data_y"]))

    pred_test = pl.predict_set(x_test)[0]
    pred_training = pl.predict_set(x_training)[0]

    print_evaluation("al trained model test", y_test, pred_test)
    print_evaluation("al trained model training", y_training, pred_training)

    # plot history
    mae_test_history = np.load(os.path.join(filename, result_file_name["mae_test"]))
    mae_train_history = np.load(os.path.join(filename, result_file_name["mae_train"]))
    plt.plot(range(len(mae_test_history)), mae_test_history)
    plt.plot(range(len(mae_train_history)), mae_train_history)
    plt.show()

    r2_test_history = np.load(os.path.join(filename, result_file_name["r2_test"]))
    r2_train_history = np.load(os.path.join(filename, result_file_name["r2_train"]))
    plt.plot(range(len(r2_test_history)), r2_test_history)
    plt.plot(range(len(r2_test_history)), r2_test_history)
    plt.plot(range(len(r2_train_history)), r2_train_history)
    plt.show()

    # TRAINING and evaluation of REDUCED SL MODEL (train with reduced labelled set)
    run_evaluation_reduced_sl_model(flat_x=x_training, flat_x_test=x_test, flat_y=y_training, flat_y_test=y_test)
