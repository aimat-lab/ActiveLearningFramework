import logging
import time

from new_example_implementation.helpers import properties
from new_example_implementation.sl_implementations.sl_model import SLModel


def run_sl(x, x_train, x_test, y, y_train, y_test):
    # INIT
    t0 = time.time()
    reduced_sl_model = SLModel(x_test, y_test, entity=properties.entities["ip"])
    t1 = time.time()

    # TRAINING
    reduced_sl_model.train(x=x_train, y=y_train, batch_size=properties.sl_training_params["batch_size"], max_epochs=properties.sl_training_params["max_epochs"], min_epochs=properties.sl_training_params["min_epochs"], thr=properties.sl_training_params["thr"])
    t2 = time.time()

    # TEST after training
    size_training_set, mae_test, r2_test, mae_train, r2_train = reduced_sl_model.evaluate(x=x, x_test=x_test, y=y, y_test=y_test)

    logging.info(f"Reduced SL model time: initialisation {t1 - t0}, training {t2 - t1}, whole {t2 - t0}")
    return {
        "time_init": t1 - t0,
        "time_training": t2 - t1,
        "size_training_set": size_training_set,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "r2_train": r2_train
    }


