import logging
import time

from example_implementations.helpers import properties
from example_implementations.helpers.mapper import map_flat_output_to_shape, map_flat_input_to_shape
from example_implementations.evaluation.sl_model import SLModel


log = logging.getLogger("LOGGER reduced sl model:  ")


class ReducedSLModel(SLModel):

    def __init__(self, flat_x, flat_x_test, flat_y, flat_y_test):
        super().__init__(map_flat_input_to_shape(flat_x), map_flat_input_to_shape(flat_x_test), map_flat_output_to_shape(flat_y), map_flat_output_to_shape(flat_y_test), entity=properties.entities["ip"])


def run_evaluation_reduced_sl_model(flat_x, flat_x_test, flat_y, flat_y_test):
    # INIT
    t0 = time.time()
    reduced_sl_model = ReducedSLModel(flat_x, flat_x_test, flat_y, flat_y_test)
    t1 = time.time()

    # TRAINING
    reduced_sl_model.train(max_epochs=properties.sl_training_params["max_epochs"], min_epochs=properties.sl_training_params["min_epochs"], thr=properties.sl_training_params["thr"])
    t2 = time.time()

    # TEST after training
    size_training_set, mae_test, r2_test, mae_train, r2_train = reduced_sl_model.evaluate()

    log.info(f"time: initialisation {t1 - t0}, training {t2 - t1}, whole {t2 - t0}")
    return {
        "time_init": t1 - t0,
        "time_training": t2 - t1,
        "size_training_set": size_training_set,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "r2_train": r2_train
    }
