from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from vaccination_polices import make_parameterized_policy, designed_policy_beta
from simulation import simulation_results, unpack_values, get_args

def try_policy(policy: Callable, tao: float=0.8, kappa: float=4) -> Any:
    """Tries a policy out and generates simulation results with default parameters

    Args:
        policy (Callable): The vaccination policy
        tao (float, optional): The infection rate. Defaults to 0.8.
        kappa (float, optional): The recovery time. Defaults to 4.

    Returns:
        Any: The simulation results
    """
    return simulation_results(
        force_run=True,
        log=True,
        show_plot=True,
        generate_plot=True,
        save_results=False,
        vaccination_policy=policy,
        tao=tao,
        kappa=kappa
    )

def test_policy(s: float, i: float, r: float, v: float, c: float, tao: float, kappa: float) -> float:
    """Vaccination policy that returns a derivative proportional to the susceptible proportion divided by the infected proportion

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
    return c * s / i

def main() -> None:
    """main runner function"""
    args = get_args()

    c_space = np.linspace(0, 1, 500)
    final_ss = []
    stop_ts = []
    final_vs = []
    final_rs = []

    for c in tqdm(c_space):

        policy = make_parameterized_policy(c=c)(designed_policy_beta)

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

    plt.plot(c_space, scores, label="Final Susceptible Population")

    plt.xlabel("c")
    plt.ylabel("metrics")
    plt.title("Scores vs C Parameter")

    c_opt = c_space[np.argmax(scores)]
    print(f"Optimal c parameter: {c_opt}")

    plt.legend()

    plt.savefig(f"results/analysis/designed_model_beta_tune_tao_{args.tao}_kappa_{args.kappa}.png")

    plt.show()

if __name__ == "__main__":
    main()