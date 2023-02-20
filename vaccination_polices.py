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
        def policy(s, i, r, v, tao, kappa):
            return func(s, i, r, v, tao, kappa, **kwargs)

        policy.__name__ = func.__name__ if name is None else name
        return policy

    return policy_wrapper


def zero_policy(
    s: float, i: float, r: float, v: float, tao: float, kappa: float
) -> float:
    """Vaccination policy that always returns 0

    Args:
        s (float): The current susceptible proportion
        i (float): The current infected proportion
        r (float): The current recovered proportion
        v (float): The current vaccinated proportion
        tao (float): The infection rate
        kappa (float): The recovery time

    Returns:
        float: the derivative of the vaccinated proportion
    """
    return 0


@make_parameterized_policy(name="test_2", c=2)
def example_policy(
    s: float, i: float, r: float, v: float, tao: float, kappa: float, c: float
) -> float:
    """Vaccination policy that returns a derivative proportional to the susceptible proportion

    Args:
        s (float): The current susceptible proportion
        i (float): The current infected proportion
        r (float): The current recovered proportion
        v (float): The current vaccinated proportion
        tao (float): The infection rate
        kappa (float): The recovery time
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
    tao: float,
    kappa: float,
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
        tao (float): The infection rate
        kappa (float): The recovery time
        weights1 (np.array): The first weight matrix in the neural net
        bias1 (np.array): The first bias vector in the neural net
        weights2 (np.array): The second weight matrix in the neural net
        bias2 (np.array): The second bias vector in the neural net

    Returns:
        float: The derivative of the vaccinated proportion
    """

    input_vec = np.array([[s, i, r, v, tao, kappa]]).T

    cur = input_vec

    cur = weights1 @ cur + bias1
    cur = cur * (cur > 0)
    cur = weights2 @ cur + bias2
    cur = 1 / (1 + np.exp(-cur))

    cur = cur[0][0]
    return cur * s


def get_saved_neural_policy(tao: float = 0.8, kappa: float = 4) -> Callable:
    """Creates the neural policy with the saved weights from previous training

    Args:
        tao (float, optional): The infection rate. Defaults to 0.8.
        kappa (float, optional): The recovery time. Defaults to 4.

    Returns:
        Callable: The neural policy
    """
    dir_name = f"tao_{tao}_{tao}_kappa_{kappa}_{kappa}"

    weights1 = np.loadtxt(
        f"results/models/{dir_name}/weights1.csv",
        delimiter=",",
    ).reshape((hidden_layer_dim, 6))
    bias1 = np.loadtxt(f"results/models/{dir_name}/bias1.csv", delimiter=",").reshape(
        (hidden_layer_dim, 1)
    )
    weights2 = np.loadtxt(
        f"results/models/{dir_name}/weights2.csv", delimiter=","
    ).reshape((1, hidden_layer_dim))
    bias2 = np.loadtxt(f"results/models/{dir_name}/bias2.csv", delimiter=",").reshape(
        (1, 1)
    )

    policy = make_parameterized_policy(
        name=f"neural_policy_0",
        weights1=weights1,
        bias1=bias1,
        weights2=weights2,
        bias2=bias2,
    )(neural_policy)

    return policy
