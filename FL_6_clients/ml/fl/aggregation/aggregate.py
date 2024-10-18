import copy
import math
from typing import List, Tuple
from functools import reduce

import numpy as np


def simple_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> np.ndarray:
    """Compute a simple average."""
    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    weights_prime: np.ndarray = [
        reduce(np.add, layer_updates) / len(weights)
        for layer_updates in zip(*weights)
    ]
    return weights_prime


def median_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> np.ndarray:
    """Compute median across weights."""
    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    weights_prime: np.ndarray = [
        np.median(layer_updates, axis=0)
        for layer_updates in zip(*weights)
    ]
    return weights_prime


def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> np.ndarray:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: np.ndarray = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def fednova_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
                      rho: float = 0.) -> List[np.ndarray]:
    """Compute weighted average according to FedNova."""
    num_examples = [num_examples for _, num_examples in results]
    num_examples_total = sum(num_examples)

    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    taus = copy.deepcopy(num_examples)
    alphas = [taus[i] - rho * (1 - math.pow(rho, taus[i])) / (1 - rho) / (1 - rho) for i in range(len(taus))]

    diffs = copy.deepcopy(weights)
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            diffs[i][j] = (previous_model[j] - weights[i][j]) / alphas[i]

    d_total_round = [np.zeros_like(previous_model[i]) for i in range(len(previous_model))]

    for i in range(len(diffs)):
        d_para = diffs[i]
        for j in range(len(diffs[i])):
            d_total_round[j] = np.add(d_total_round[j], d_para[j] * num_examples[i] / num_examples_total)

    coeff = 0.
    for i in range(len(diffs)):
        coeff = np.add(coeff, alphas[i] * num_examples[i] / num_examples_total)

    weights_prime: List[np.ndarray] = copy.deepcopy(previous_model)
    for i in range(len(weights_prime)):
        weights_prime[i] = np.subtract(weights_prime[i], coeff * d_total_round[i])

    return weights_prime


def fedadagrad_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
                         m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                         beta_1: float = 0., eta: float = 0.1,
                         tau: float = 1e-2) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    fedavg_aggregated = fedavg_aggregate(results)

    delta_t: List[np.ndarray] = [
        x - y for x, y in zip(fedavg_aggregated, previous_model)
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [x + np.multiply(y, y) for x, y in zip(v_t, delta_t)]

    new_weights = [
        x + eta * y / (np.sqrt(z) + tau)
        for x, y, z in zip(previous_model, m_t, v_t)
    ]
    return new_weights, m_t, v_t


def fedyogi_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
                      m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                      beta_1: float = 0.9, beta_2: float = 0.99,
                      eta: float = 0.01,
                      tau: float = 1e-3) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    fedavg_aggregated = fedavg_aggregate(results)

    delta_t: List[np.ndarray] = [
        x - y for x, y in zip(fedavg_aggregated, previous_model)
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [x - (1.0 - beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
           for x, y in zip(v_t, delta_t)]

    new_weights = [
        x + eta * y / (np.sqrt(z) + tau)
        for x, y, z in zip(previous_model, m_t, v_t)
    ]
    return new_weights, m_t, v_t


def fedadam_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
                      m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                      beta_1: float = 0.9, beta_2: float = 0.99,
                      eta: float = 0.01,
                      tau: float = 1e-3) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    fedavg_aggregated = fedavg_aggregate(results)

    delta_t: List[np.ndarray] = [
        x - y for x, y in zip(fedavg_aggregated, previous_model)
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [beta_2 * x + (1 - beta_2) * np.multiply(y, y)
           for x, y in zip(v_t, delta_t)]

    new_weights = [
        x + eta * y / (np.sqrt(z) + tau)
        for x, y, z in zip(previous_model, m_t, v_t)
    ]
    return new_weights, m_t, v_t


def fedavgm_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
                      server_momentum: float = 0., momentum_vector: List[np.ndarray] = None,
                      server_lr: float = 1.) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    fedavg_aggregated = fedavg_aggregate(results)

    pseudo_gradient: List[np.ndarray] = [
        x - y for x, y in zip(previous_model, fedavg_aggregated)
    ]
    if server_momentum > 0.0:
        if momentum_vector is not None:
            momentum_vector = [
                server_momentum * x + y
                for x, y in zip(momentum_vector, pseudo_gradient)
            ]
        else:
            momentum_vector = pseudo_gradient
        pseudo_gradient = copy.deepcopy(momentum_vector)

    new_weights = [
        x - server_lr * y
        for x, y in zip(previous_model, pseudo_gradient)
    ]

    return new_weights, pseudo_gradient


def test_aggregate() -> None:
    """Test aggregate function."""

    # Prepare
    initial_weights0_0 = np.array([[0., 0., 0.], [0., 0., 0.]])
    initial_weights0_1 = np.array([0., 0., 0., 0.])
    weights0_0 = np.array([[0.1, 0.1, 0.1], [0.15, 0.15, 0.15]])
    weights0_1 = np.array([0.5, 0.5, 0.5, 0.5])
    weights1_0 = np.array([[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]])
    weights1_1 = np.array([1., 1., 1., 1.])
    weights2_0 = np.array([[0.3, 0.3, 0.3], [0.8, 0.8, 0.8]])
    weights2_1 = np.array([1.5, 1.5, 1.5, 1.5])
    weights3_0 = np.array([[0.4, 0.4, 0.4], [1., 1., 1.]])
    weights3_1 = np.array([2., 2., 2., 2.])

    results = [([weights0_0, weights0_1], 1), ([weights1_0, weights1_1], 10),
               ([weights2_0, weights2_1], 10), ([weights3_0, weights3_1], 10)]

    avg = simple_aggregate(results)
    expected = [np.array([[.25, .25, .25], [.6375, .6375, .6375]]), np.array([1.25, 1.25, 1.25, 1.25])]
    np.testing.assert_equal(expected, avg)

    fedavg = fedavg_aggregate(results)

    median = median_aggregate(results)
    expected = [np.array([[0.25, 0.25, 0.25],
                          [0.7, 0.7, 0.7]]), np.array([1.25, 1.25, 1.25, 1.25])]
    np.testing.assert_equal(expected, median)

    print("Simple Average", avg)
    print("FedAvg", fedavg)
    print("Simple Median", median)
    initial_weights = [initial_weights0_0, initial_weights0_1]

    print("FedNova with rho=0.", fednova_aggregate(results, initial_weights))
    print("FedNova with rho=0.9", fednova_aggregate(results, initial_weights, rho=0.9))

    aggregated, _, _ = fedadagrad_aggregate(results, initial_weights, beta_1=0., eta=0.1, tau=1e-2)
    print("Fedadagrad with beta_1=0., eta =0.1, tau=1e-2", aggregated)
    aggregated, _, _ = fedadagrad_aggregate(results, initial_weights, beta_1=0., eta=0.01, tau=1e-5)
    print("Fedadagrad with beta_1=0., eta =0.01, tau=1e-5", aggregated)

    aggregated, _, _ = fedyogi_aggregate(results, initial_weights, beta_1=0.9, beta_2=0.99, eta=1e-1, tau=1e-3)
    print("Fedyogi with beta_1=0.9, beta_2=0.99, eta=0.1, tau=1e-3", aggregated)
    aggregated, _, _ = fedyogi_aggregate(results, initial_weights, beta_1=0., beta_2=0.99, eta=1e-2, tau=1e-3)
    print("Fedyogi with beta_1=0., beta_2=0.99, eta=0.01, tau=1e-3", aggregated)

    aggregated, _, _ = fedadam_aggregate(results, initial_weights, beta_1=0.9, beta_2=0.99, eta=1e-1, tau=1e-3)
    print("FedAdam with beta_1=0.9, beta_2=0.99, eta=0.1, tau=1e-3", aggregated)
    aggregated, _, _ = fedadam_aggregate(results, initial_weights, beta_1=0., beta_2=0.99, eta=1e-2, tau=1e-3)
    print("FedAdam with beta_1=0., beta_2=0.99, eta=0.01, tau=1e-3", aggregated)

    print("FedAdagrad")
    previous = [np.array([0.1, 0.1, 0.1, 0.1])]
    param_0 = [np.array([0.2, 0.2, 0.2, 0.2])]
    param_1 = [np.array([1., 1., 1., 1.])]
    results_ = [(param_0, 5), (param_1, 5)]
    expected = [np.array([0.15, 0.15, 0.15, 0.15])]
    agg, _, _ = fedadagrad_aggregate(results_, previous, eta=0.1, tau=0.5)
    print("Expected:", expected, " -- Actual", agg)

    aggregated, _ = fedavgm_aggregate(results, initial_weights, server_momentum=0., server_lr=1.)
    print("FedAvgM with server_momentum=0., server_lr=1.", aggregated)
    aggregated, _ = fedavgm_aggregate(results, initial_weights, server_momentum=0.7, server_lr=0.5)
    print("FedAvgM with server_momentum=0.7, server_lr=1.", aggregated)

    previous = [np.array([[0, 0, 0], [0, 0, 0]]), np.array([0, 0, 0, 0])]
    param_0_0 = np.array([[1, 2, 3], [4, 5, 6]])
    param_0_1 = np.array([7, 8, 9, 10])
    param_1_0 = np.array([[1, 2, 3], [4, 5, 6]])
    param_1_1 = np.array([7, 8, 9, 10])

    results_ = [([param_0_0, param_0_1], 5), ([param_1_0, param_1_1], 5)]
    expected = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([7, 8, 9, 10])]
    agg, _ = fedavgm_aggregate(results_, previous, server_lr=1. + 1e-9)
    print("Expected:", expected, " -- Actual", agg)
    agg, mom_vec = fedavgm_aggregate(results_, previous, server_lr=1. + 1e-9, server_momentum=1e-9)
    print("First round Expected:", expected, " -- Actual", agg)
    agg, _ = fedavgm_aggregate(results_, agg, momentum_vector=mom_vec, server_lr=1.0 + 1e-9, server_momentum=1e-9)
    print("Second round Expected:", expected, " -- Actual", agg)