import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from example_implementation.evaluation.ia__execution import run_al
from example_implementation.evaluation.ip_execution import run_sl
from example_implementation.evaluation.ua__execution import run_al__unfiltered
from example_implementation.helpers import properties
from example_implementation.helpers.mapper import map_shape_input_to_flat, map_shape_output_to_flat

logging.basicConfig(format='\nLOGGING: %(name)s, %(levelname)s: %(message)s :END LOGGING', level=logging.INFO)
log = logging.getLogger("Main logger")

if __name__ == '__main__':
    # Select data (test and training data separation)
    geos = np.load(properties.data_location["coord"])  # in A
    energies = np.load(properties.data_location["energy"])  # in eV
    grads = np.load(properties.data_location["force"]) * 27.21138624598853 / 0.52917721090380  # from H/B to eV/A
    energies = np.expand_dims(energies, axis=1)
    grads = np.expand_dims(grads, axis=1)

    random_idx = np.arange(len(geos))
    np.random.shuffle(random_idx)
    geos = np.array([geos[i] for i in random_idx])
    energies = np.array([energies[i] for i in random_idx])
    grads = np.array([grads[i] for i in random_idx])

    x, x_test = map_shape_input_to_flat(geos[properties.test_set_size:]), map_shape_input_to_flat(geos[:properties.test_set_size])
    y, y_test = map_shape_output_to_flat([energies[properties.test_set_size:], grads[properties.test_set_size:]]), map_shape_output_to_flat([energies[:properties.test_set_size], grads[:properties.test_set_size]])

    # Evaluation of part: IA -> actual AL model
    ia_results = run_al(x, x_test, y, y_test)

    # Evaluation of part: IP -> reduced SL model
    reduced_x, reduced_y = np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_x)), np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_y))
    ip_results = run_sl(x=x, x_test=x_test, x_train=reduced_x, y=y, y_test=y_test, y_train=reduced_y)

    # Evaluation of part: UA -> unintelligent AL model
    ua_results = run_al__unfiltered(x, x_test, y, y_test)

    print("IA")
    print(ia_results)

    print("IP")
    print(ip_results)

    print("UA")
    print(ua_results)

    plt.clf()
    color_ua_test = "#00aedb"
    color_ua_train = "#007999"
    color_ia_test = "#f37735"
    color_ia_train = "#C25F2A"
    color_ip = "#ffc425"

    # plot metrics history
    filename = os.path.abspath(os.path.abspath(properties.results_location["active_metrics_over_iterations"]))

    ia_test_mae = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_test" + properties.mae_history_suffix))
    ia_train_mae = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_train" + properties.mae_history_suffix))
    ua_test_mae = np.load(os.path.join(filename, properties.eval_entities["ua"] + "_test" + properties.mae_history_suffix))
    ua_train_mae = np.load(os.path.join(filename, properties.eval_entities["ua"] + "_train" + properties.mae_history_suffix))

    plt.title("MAE history for actively trained entities over the training iterations")
    plt.xlabel("training iterations")
    plt.ylabel("MAE value (calculated for test set or current training set)")
    plt.yscale('log')

    plt.plot([i * 16 for i in range(len(ia_test_mae))], ia_test_mae, label="IA test mae", color=color_ia_test)
    plt.plot([i * 16 for i in range(len(ia_train_mae))], ia_train_mae, label="IA train mae", color=color_ia_train)
    plt.plot([i * 16 for i in range(len(ua_test_mae))], ua_test_mae, label="UA test mae", color=color_ua_test)
    plt.plot([i * 16 for i in range(len(ua_train_mae))], ua_train_mae, label="UA train mae", color=color_ua_train)

    plt.legend()
    plt.savefig(properties.results_location["active_metrics_over_iterations"] + "mae_plot")
    plt.clf()

    ia_test_r2 = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_test" + properties.r2_history_suffix))
    ia_train_r2 = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_train" + properties.r2_history_suffix))
    ua_test_r2 = np.load(os.path.join(filename, properties.eval_entities["ua"] + "_test" + properties.r2_history_suffix))
    ua_train_r2 = np.load(os.path.join(filename, properties.eval_entities["ua"] + "_train" + properties.r2_history_suffix))

    plt.title("R-squared history for actively trained entities over the training iterations")
    plt.xlabel("training iterations")
    plt.ylabel("R2 value (calculated for test set or current training set)")
    plt.yscale('log')

    plt.plot([i * 16 for i in range(len(ia_test_r2))], ia_test_r2, label="IA test r2", color=color_ia_test)
    plt.plot([i * 16 for i in range(len(ia_train_r2))], ia_train_r2, label="IA train r2", color=color_ia_train)
    plt.plot([i * 16 for i in range(len(ua_test_r2))], ua_test_r2, label="UA test r2", color=color_ua_test)
    plt.plot([i * 16 for i in range(len(ua_train_r2))], ua_train_r2, label="UA train r2", color=color_ua_train)

    plt.legend()
    plt.savefig(properties.results_location["active_metrics_over_iterations"] + "r2_plot")
    plt.clf()

    # plot loss history
    filename = os.path.abspath(os.path.abspath(properties.results_location["loss_over_epochs"]))
    ia_loss = np.load(os.path.join(filename, properties.eval_entities["ia"] + "_0" + properties.loss_history_suffix))  # always use first of the internal models for comparison
    ua_loss = np.load(os.path.join(filename, properties.eval_entities["ua"] + "_0" + properties.loss_history_suffix))  # always use first of the internal models for comparison
    ip_loss = np.load(os.path.join(filename, properties.eval_entities["ip"] + properties.loss_history_suffix))

    plt.title("Loss history for all entities over epochs")
    plt.xlabel("epochs")
    plt.ylabel("training loss")
    plt.yscale('log')

    plt.plot(range(len(ia_loss)), ia_loss, label="IA loss", color=color_ia_test, linewidth=0.75)
    plt.plot(range(len(ua_loss)), ua_loss, label="UA loss", color=color_ua_test, linewidth=0.75)
    plt.plot(range(len(ip_loss)), ip_loss, label="IP loss", color=color_ip, linewidth=0.75)

    plt.legend()
    plt.savefig(properties.results_location["loss_over_epochs"] + "loss_plot")

