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