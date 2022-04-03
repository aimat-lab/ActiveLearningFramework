import numpy as np


def map_shape_input_to_flat(shape_xs):
    # shape input: (n, 6, 3)
    # flat input: (n, 6*3)
    return np.array([x.flatten() for x in shape_xs])


def map_flat_input_to_shape(flat_xs):
    # flat input: (n, 6*3)
    # shape input: (n, 6, 3)
    return np.array(flat_xs).reshape((len(flat_xs), 6, 3))


def map_shape_output_to_flat(shape_ys):
    # shape input: [(n, 1), (n, 1, 6, 3)] -> [energy, force]
    # flat output: (n, 1 + 1*6*3)
    eng, grads = shape_ys[0], shape_ys[1]
    return np.array([np.append(eng[i], grads[i].flatten()) for i in range(len(eng))])


def map_flat_output_to_shape(flat_ys):
    # flat output: (n, 1 + 1*6*3)
    # shape input: [(n, 1), (n, 1, 6, 3)] -> [energy, force]
    return [flat_ys[:, 0:1], np.array(flat_ys[:, 1:]).reshape((len(flat_ys), 1, 6, 3))]
