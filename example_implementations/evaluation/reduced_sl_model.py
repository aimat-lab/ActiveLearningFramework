import logging
import time

from example_implementations.helpers.mapper import map_flat_output_to_shape, map_flat_input_to_shape
from example_implementations.evaluation.sl_model import SLModel


log = logging.getLogger("LOGGER reduced sl model:  ")


class ReducedSLModel(SLModel):

    def __init__(self, flat_x, flat_x_test, flat_y, flat_y_test):
        super().__init__(map_flat_input_to_shape(flat_x), map_flat_input_to_shape(flat_x_test), map_flat_output_to_shape(flat_y), map_flat_output_to_shape(flat_y_test), title_prefix="reduced sl model ")


def run_evaluation_reduced_sl_model(flat_x, flat_x_test, flat_y, flat_y_test):
    # INIT
    t0 = time.time()
    reduced_sl_model = ReducedSLModel(flat_x, flat_x_test, flat_y, flat_y_test)
    t1 = time.time()

    # TEST before training
    reduced_sl_model.evaluate()
    t2 = time.time()

    # TRAINING
    reduced_sl_model.train(max_epochs=2000, min_epochs=20, thr=0.7)
    t3 = time.time()

    # TEST after training
    reduced_sl_model.evaluate()

    log.info(f"time: initialisation {t1 - t0}, training {t3 - t2}, whole {(t3 - t0) - (t2 - t1)}")