import logging

import numpy as np
from matplotlib import pyplot as plt
from pyNNsMD.plots.pred import plot_scatter_prediction

from example_implementation.helpers import properties


def mae_single_instance(y, pred):
    return np.mean(np.abs(pred - y))


def mae_set(ys, preds):
    assert len(ys) == len(preds)
    return np.mean(np.array(
        [np.mean(np.abs(preds[i] - ys[i])) for i in range(len(ys))]
    ))


def r2_single_instance(y, pred):
    return 1 - (y - pred) ** 2 / (y ** 2)


def r2_set(ys, preds):
    assert len(ys) == len(preds)
    return np.mean(np.array([
        1 - (ys[i] - preds[i]) ** 2 / (ys[i] ** 2) for i in range(len(ys))
    ]))


def metrics_single_instance(y, pred):
    return mae_single_instance(y, pred), r2_single_instance(y, pred)


def metrics_set(ys, preds):
    return mae_set(ys, preds), r2_set(ys, preds)


def print_evaluation(title, ys, preds):
    logging.info(f"{title}: len = {len(ys)}")

    logging.info(f"{title}: mae = {mae_set(ys, preds)}, r2 = {r2_set(ys, preds)}")

    # Plot Prediction
    fig = plot_scatter_prediction(y_val=ys, y_pred=preds, plot_title=f"Prediction {title}")
    plt.show()


def calc_final_evaluation(ys, preds, title, location):
    # Plot Prediction
    plot_scatter_prediction(y_val=ys, y_pred=preds, plot_title=f"{title}")
    plt.savefig(properties.results_location["prediction_image"] + location)

    return len(ys), mae_set(ys, preds), r2_set(ys, preds)
