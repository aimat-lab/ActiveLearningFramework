import logging
import time

from example_implementations.helpers import properties
from example_implementations.evaluation.sl_model import SLModel


class OriginalSLModel(SLModel):

    def __init__(self, x, x_test, y, y_test):
        super().__init__(x, x_test, y, y_test, entity=properties.entities["up"])


def run_evaluation_original_sl_model(x, x_test, y, y_test):
    # INIT
    t0 = time.time()
    original_sl_model = OriginalSLModel(x, x_test, y, y_test)
    t1 = time.time()

    # TRAINING
    original_sl_model.train(max_epochs=properties.sl_training_params["max_epochs"], min_epochs=properties.sl_training_params["min_epochs"], thr=properties.sl_training_params["thr"])
    t2 = time.time()

    # TEST after training
    size_training_set, mae_test, r2_test, mae_train, r2_train = original_sl_model.evaluate()

    logging.info(f"Original SL model time: initialisation {t1 - t0}, training {t2 - t1}, whole {t2 - t0}")
    return {
        "time_init": t1-t0,
        "time_training": t2-t1,
        "size_training_set": size_training_set,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "r2_train": r2_train
    }
