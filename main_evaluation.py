import logging
import os

import numpy as np

from example_implementations.evaluation.original_sl_model import run_evaluation_original_sl_model
from example_implementations.evaluation.reduced_sl_model import run_evaluation_reduced_sl_model
from example_implementations.helpers import properties
from example_implementations.helpers.active_runners import run_al, run_al_unfiltered
from example_implementations.helpers.mapper import map_shape_input_to_flat, map_shape_output_to_flat

logging.basicConfig(format='\nLOGGING: %(name)s, %(levelname)s: %(message)s :END LOGGING', level=logging.INFO)
log = logging.getLogger("Main logger")

if __name__ == '__main__':
    # Select data (test and training data separation)
    x_loaded = np.load(properties.data_location["x"])
    random_idx = np.arange(len(x_loaded))
    np.random.shuffle(random_idx)

    x_loaded = np.array([x_loaded[i] for i in random_idx])
    eng = np.load(properties.data_location["energy"])
    grads = np.load(properties.data_location["force"])
    eng = np.array([eng[i] for i in random_idx])
    grads = np.array([grads[i] for i in random_idx])

    x, x_test, y, y_test = x_loaded[properties.test_set_size:], x_loaded[:properties.test_set_size], [eng[properties.test_set_size:], grads[properties.test_set_size:]], [eng[:properties.test_set_size], grads[:properties.test_set_size]]

    # Evaluation of part: UP -> original SL model
    up_results = run_evaluation_original_sl_model(x, x_test, y, y_test)

    # Evaluation of part: IA -> actual AL model
    ia_results = run_al(x, x_test, y, y_test)

    # Evaluation of part: IP -> reduced SL model
    reduced_x, reduced_y = np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_x)), np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_y))
    ip_results = run_evaluation_reduced_sl_model(flat_x=reduced_x, flat_x_test=map_shape_input_to_flat(x_test), flat_y=reduced_y, flat_y_test=map_shape_output_to_flat(y_test))

    # Evaluation of part: UA -> unintelligent AL model
    ua_results = run_al_unfiltered(x, x_test, y, y_test)

    print("UP")
    print(up_results)

    print("IA")
    print(ia_results)

    print("IP")
    print(ip_results)

    print("UA")
    print(ua_results)
