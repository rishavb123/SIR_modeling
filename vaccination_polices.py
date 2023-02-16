from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from simulation import  simulation_results, unpack_values, get_args

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

def test_policy(s, i, r, v, c):
    return c * r * s


def main() -> None:
    """main runner function"""
    args = get_args()

    c_space = np.linspace(0.1, 10, 100)
    stop_ts = []
    final_vs = []

    for c in c_space:

        policy = make_parameterized_policy(name="not_test", c=c)(test_policy)

        sol = simulation_results(
            s0=args.s0,
            i0=args.i0,
            r0=args.r0,
            v0=args.v0,
            tao=args.tao,
            kappa=args.kappa,
            log=args.log,
            force_run=args.force_run,
            show_plot=args.plot,
            generate_plot=False,
            save_results=False,
            vaccination_policy=policy,
        )
        t, s, i, r, v, stop_t = unpack_values(sol)
        stop_ts.append(stop_t)
        final_vs.append(v[-1])

    stop_ts = np.array(stop_ts)
    final_vs = np.array(final_vs)

    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    alpha = 0.5
    beta = 0.5

    stop_ts = normalize(stop_ts)
    final_vs = normalize(final_vs)

    scores = alpha * stop_ts + beta * final_vs

    scores = normalize(scores)

    plt.plot(c_space, scores)
    plt.plot(c_space, stop_ts)
    plt.plot(c_space, final_vs)

    c_opt = c_space[np.argmax(scores)]
    print(f"Optimal c parameter: {c_opt}")

    plt.show()


if __name__ == "__main__":
    main()
