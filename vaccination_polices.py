from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_parameterized_policy(name: str = None, **kwargs) -> Callable:
    """Creates the wrapper for a parameterized policy using the parameters provided in kwargs

    Args:
        name (str, optional): The name of the policy. Defaults to the name of the policy function.

    Returns:
        Callable: The function wrapper or decorator.
    """
    def policy_wrapper(func):
        def policy(s, i, r, v):
            return func(s, i, r, v, **kwargs)
        policy.__name__ = func.__name__ if name is None else name
        return policy
    return policy_wrapper

def zero_policy(s: float, i: float, r: float, v: float) -> float:
    """Vaccination policy that always returns 0

    Args:
        s (float): The current susceptible proportion
        i (float): The current infected proportion
        r (float): The current recovered proportion
        v (float): The current vaccinated proportion

    Returns:
        float: the derivative of the vaccinated proportion
    """
    return 0

@make_parameterized_policy(name="test_2", c=2)
def example_policy(s: float, i: float, r: float, v: float, c: float) -> float:
    """Vaccination policy that returns a derivative proportional to the susceptible proportion

    Args:
        s (float): The current susceptible proportion
        i (float): The current infected proportion
        r (float): The current recovered proportion
        v (float): The current vaccinated proportion
        c (float): The parameter of the policy

    Returns:
        float: the derivative of the vaccinated proportion
    """
    return c * s

hidden_layer_dim = 10

def neural_policy(
    s: float,
    i: float,
    r: float,
    v: float,
    weights1: np.array,
    bias1: np.array,
    weights2: np.array,
    bias2: np.array,
) -> float:
    """Vaccination policy that is learned using the neat algorithm

    Args:
        s (float): The current susceptible proportion
        i (float): The current infected proportion
        r (float): The current recovered proportion
        v (float): The current vaccinated proportion
        weights1 (np.array): The first weight matrix in the neural net
        bias1 (np.array): The first bias vector in the neural net
        weights2 (np.array): The second weight matrix in the neural net
        bias2 (np.array): The second bias vector in the neural net

    Returns:
        float: The derivative of the vaccinated proportion
    """

    input_vec = np.array([[s, i, r, v]]).T

    cur = input_vec

    cur = weights1 @ cur + bias1
    cur = cur * (cur > 0)
    cur = weights2 @ cur + bias2
    cur = 1 / (1 + np.exp(-cur))

    cur = cur[0][0]
    return cur * s


def get_saved_neural_policy() -> Callable:
    """Creates the neural policy with the saved weights from previous training

    Returns:
        Callable: The neural policy
    """
    hidden_layer_dim = 10
    weights1 = np.loadtxt(
        "results/models/weights1.csv",
        delimiter=",",
    ).reshape((hidden_layer_dim, 4))
    bias1 = np.loadtxt("results/models/bias1.csv", delimiter=",").reshape(
        (hidden_layer_dim, 1)
    )
    weights2 = np.loadtxt("results/models/weights2.csv", delimiter=",").reshape(
        (1, hidden_layer_dim)
    )
    bias2 = np.loadtxt("results/models/bias2.csv", delimiter=",").reshape((1, 1))

    policy = make_parameterized_policy(
        name=f"neural_policy_0",
        weights1=weights1,
        bias1=bias1,
        weights2=weights2,
        bias2=bias2,
    )(neural_policy)

    return policy