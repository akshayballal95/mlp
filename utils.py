import numpy as np


def relu(z):
    A = np.maximum(0, z)
    cache = z
    return A, cache


def relu_activation(z):
    return (z * (z > 0), z)


def relu_backward(da, cache):
    Z = cache
    dZ = np.array(da, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape
    return dZ


def sigmoid(z):
    A = 1 / (1 + np.exp(-z))
    cache = z

    return A, cache


def sigmoid_activation(z):
    return (sigmoid(z), z)


def sigmoid_backward(da, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = da * s * (1 - s)

    assert dZ.shape == Z.shape

    return dZ


def linear_forward(a, w, b):
    z = np.dot(w, a) + b
    cache = (a, w, b)
    return (z, cache)


def cost(al, y):
    m = y.shape[0]
    cost = -(1 / m) * (np.dot(y, np.log(al)) + np.dot(1 - y, np.log((1 - al))))
    return np.sum(cost)


def linear_forward_activation(a_prev, w, b, activation):
    if activation == "sigmoid":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)

    if activation == "relu":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)

    cache = (linear_cache, activation_cache)
    return (a, cache)


def linear_backward(dz, linear_cache):
    a_prev, w, _ = linear_cache
    m = a_prev.shape[1]
    dw = (1.0 / m) * (np.dot(dz, a_prev.T))
    db = (1.0 / m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(w.T, dz)

    return (da_prev, dw, db)


def linear_backward_activation(da, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dz = relu_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)

    if activation == "sigmoid":
        dz = sigmoid_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)

    return (da_prev, dw, db)
