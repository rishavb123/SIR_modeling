from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from vaccination_polices import make_parameterized_policy
from simulation import simulation_results

def try_policy(policy: Callable) -> Any:
    """Tries a policy out and generates simulation results with default parameters

    Args:
        policy (Callable): The vaccination policy

    Returns:
        Any: The simulation results
    """
    return simulation_results(
        force_run=True,
        log=True,
        show_plot=True,
        generate_plot=True,
        save_results=False,
        vaccination_policy=policy
    )

def test_policy(s: float, i: float, r: float, v: float, c: float) -> float:
    """Vaccination policy that returns a derivative proportional to the susceptible proportion divided by the infected proportion

    Args:
        s (float): The current susceptible proportion
        i (float): The current infected proportion
        r (float): The current recovered proportion
        v (float): The current vaccinated proportion
        c (float): The parameter of the policy

    Returns:
        float: the derivative of the vaccinated proportion
    """
    return c * s / i

def main() -> None:
    """main runner function"""
    from simulation import simulation_results, unpack_values, get_args

    args = get_args()

    c_space = np.linspace(0, 2, 20)
    final_ss = []
    stop_ts = []
    final_vs = []
    final_rs = []

    for c in tqdm(c_space):

        policy = make_parameterized_policy(name="test_1", c=c)(test_policy)

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
        final_ss.append(s[-1])
        final_rs.append(r[-1])

    stop_ts = np.array(stop_ts)
    final_vs = np.array(final_vs)
    final_ss = np.array(final_ss)
    final_rs = np.array(final_rs)

    scores = final_ss

    plt.plot(c_space, scores, label="Final SS")

    plt.xlabel("c")
    plt.ylabel("metrics")
    plt.title("Scores vs C Parameter")

    c_opt = c_space[np.argmax(scores)]
    print(f"Optimal c parameter: {c_opt}")

    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()