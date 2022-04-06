import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split

from example_implementation.evaluation.ia__execution import run_al
from example_implementation.helpers import properties

logging.basicConfig(format='\nLOGGING: %(name)s, %(levelname)s: %(message)s :END LOGGING', level=logging.INFO)
log = logging.getLogger("Main logger")

if __name__ == '__main__':
    # source for boston code: https://github.com/rodrigobressan/keras_boston_housing_price

    # Select data (test and training data separation)
    housing_dataset = fetch_california_housing()

    random_idx = np.arange(len(housing_dataset["data"]))
    np.random.shuffle(random_idx)
    x = np.array([housing_dataset["data"][i] for i in random_idx])
    y = np.array([np.array([housing_dataset["target"][i]]) for i in random_idx])

    x_train, x_test, y_train, y_test = x[properties.test_set_size:], x[:properties.test_set_size], y[properties.test_set_size:], y[:properties.test_set_size]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Evaluation of part: IA -> actual AL model
    ia_results = run_al(x, x_test, y, y_test)

    print("IA")
    print(ia_results)

    plt.clf()
    color_ia_test = "#f37735"
    color_ia_train = "#C25F2A"

    # plot metrics history
    filename = os.path.abspath(os.path.abspath(properties.results_location["active_metrics_over_iterations"]))

    ia_test_mae = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_test" + properties.mae_history_suffix))
    ia_train_mae = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_train" + properties.mae_history_suffix))

    plt.title("MAE history for actively trained entities over the training iterations")
    plt.xlabel("training iterations")
    plt.ylabel("MAE value (calculated for test set or current training set)")
    plt.yscale('log')

    plt.plot([i * 16 for i in range(len(ia_test_mae))], ia_test_mae, label="IA test mae", color=color_ia_test)
    plt.plot([i * 16 for i in range(len(ia_train_mae))], ia_train_mae, label="IA train mae", color=color_ia_train)

    plt.legend()
    plt.savefig(properties.results_location["active_metrics_over_iterations"] + "mae_plot")
    plt.clf()

    ia_test_r2 = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_test" + properties.r2_history_suffix))
    ia_train_r2 = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_train" + properties.r2_history_suffix))

    plt.title("R-squared history for actively trained entities over the training iterations")
    plt.xlabel("training iterations")
    plt.ylabel("R2 value (calculated for test set or current training set)")
    plt.yscale('log')

    plt.plot([i * 16 for i in range(len(ia_test_r2))], ia_test_r2, label="IA test r2", color=color_ia_test)
    plt.plot([i * 16 for i in range(len(ia_train_r2))], ia_train_r2, label="IA train r2", color=color_ia_train)

    plt.legend()
    plt.savefig(properties.results_location["active_metrics_over_iterations"] + "r2_plot")
    plt.clf()

    # plot loss history
    filename = os.path.abspath(os.path.abspath(properties.results_location["loss_over_epochs"]))
    ia_loss = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_0" + properties.loss_history_suffix))  # always use first of the internal models for comparison

    plt.title("Loss history for all entities over epochs")
    plt.xlabel("epochs")
    plt.ylabel("training loss")
    plt.yscale('log')

    plt.plot(range(len(ia_loss)), ia_loss, label="IA loss", color=color_ia_test, linewidth=0.75)

    plt.legend()
    plt.savefig(properties.results_location["loss_over_epochs"] + "loss_plot")
