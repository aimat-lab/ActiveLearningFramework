import logging
import os

import numpy as np

from new_example_implementation.evaluation.ia__execution import run_al
from new_example_implementation.evaluation.ip_execution import run_sl
from new_example_implementation.evaluation.ua__execution import run_al__unfiltered
from new_example_implementation.evaluation.up_execution import run_sl__unfiltered
from new_example_implementation.helpers import properties
from new_example_implementation.helpers.mapper import map_shape_input_to_flat, map_shape_output_to_flat

logging.basicConfig(format='\nLOGGING: %(name)s, %(levelname)s: %(message)s :END LOGGING', level=logging.INFO)
log = logging.getLogger("Main logger")

if __name__ == '__main__':
    # Select data (test and training data separation)
    x_loaded = np.load(properties.data_location["x"])
    random_idx = np.arange(len(x_loaded))
    np.random.shuffle(random_idx)
    random_idx = random_idx

    x_loaded = np.array([x_loaded[i] for i in random_idx])
    eng = np.load(properties.data_location["energy"])
    grads = np.load(properties.data_location["force"])
    eng = np.array([eng[i] for i in random_idx])
    grads = np.array([grads[i] for i in random_idx])

    x, x_test = map_shape_input_to_flat(x_loaded[properties.test_set_size:]), map_shape_input_to_flat(x_loaded[:properties.test_set_size])
    y, y_test = map_shape_output_to_flat([eng[properties.test_set_size:], grads[properties.test_set_size:]]), map_shape_output_to_flat([eng[:properties.test_set_size], grads[:properties.test_set_size]])

    # Evaluation of part: UP -> original SL model
    up_results = run_sl__unfiltered(x, x_test, y, y_test)

    # Evaluation of part: IA -> actual AL model
    ia_results = run_al(x, x_test, y, y_test)

    # Evaluation of part: IP -> reduced SL model
    reduced_x, reduced_y = np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_x)), np.load(os.path.join(properties.al_training_data_storage_location, properties.al_training_data_storage_y))
    ip_results = run_sl(x=x, x_test=x_test, x_train=reduced_x, y=y, y_test=y_test, y_train=reduced_y)

    # Evaluation of part: UA -> unintelligent AL model
    ua_results = run_al__unfiltered(x, x_test, y, y_test)

    print("UP")
    print(up_results)

    print("IA")
    print(ia_results)

    print("IP")
    print(ip_results)

    print("UA")
    print(ua_results)
