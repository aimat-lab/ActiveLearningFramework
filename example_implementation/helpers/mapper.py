import numpy as np


def map_flat_input_to_shape(flat_xs):
    return np.array(flat_xs).reshape((len(flat_xs), 12, 3))


def map_flat_output_to_shape(flat_ys):
    return [flat_ys[:, 0:2], np.array(flat_ys[:, 2:]).reshape((len(flat_ys), 2, 12, 3))]


def map_shape_input_to_flat(xs):
    return np.array([instance.flatten() for instance in xs])


def map_shape_output_to_flat(ys):
    return np.array([np.append(ys[0][i], ys[1][i].flatten()) for i in range(len(ys[0]))])

