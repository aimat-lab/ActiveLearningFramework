import logging
import time

import numpy as np

from example_implementations.evaluation.sl_model import SLModel

logging.basicConfig(format='\n%(name)s, %(levelname)s: %(message)s :END LOGGING', level=logging.INFO)
log = logging.getLogger("LOGGER original sl model:  ")


class OriginalSLModel(SLModel):

    def __init__(self, test_set_size):
        x_loaded = np.load("../butene_data/butene_x.npy")
        random_idx = np.arange(len(x_loaded))
        np.random.shuffle(random_idx)

        x_loaded = np.array([x_loaded[i] for i in random_idx])
        eng = np.load("../butene_data/butene_energy.npy")
        grads = np.load("../butene_data/butene_force.npy")
        eng = np.array([eng[i] for i in random_idx])
        grads = np.array([grads[i] for i in random_idx])

        super().__init__(x_loaded[test_set_size:], x_loaded[:test_set_size], [eng[test_set_size:], grads[test_set_size:]], [eng[:test_set_size], grads[:test_set_size]], title_prefix="original sl model ")


def run_evaluation_original_sl_model():
    # INIT
    t0 = time.time()
    original_sl_model = OriginalSLModel(256)
    t1 = time.time()

    # TEST before training
    original_sl_model.evaluate()
    t2 = time.time()

    # TRAINING
    original_sl_model.train(max_epochs=2000, min_epochs=20, thr=0.7)
    t3 = time.time()

    # TEST after training
    original_sl_model.evaluate()

    log.info(f"time: initialisation {t1 - t0}, training {t3 - t2}, whole {(t3 - t0) - (t2 - t1)}")
