from typing import Any, List, Tuple, Union

import os

import argparse
import pickle
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def f(t: float, x: List[float], tao: float = 0.8, kappa: float = 4) -> List[float]:
    """The differential equation governing the SIR model

    Args:
        t (float): The input time
        x (List[float]): The current state vector
        tao (float, optional): The infection rate. Defaults to 0.8.
        kappa (float, optional): The recovery time. Defaults to 4.

    Returns:
        List[float]: The derivative of the state vector
    """
    s, i, _ = x
    return [-tao * s * i, tao * s * i - i / kappa, i / kappa]


def run_simulation(
    s0: float = 0.99,
    i0: float = 0.01,
    r0: float = 0,
    tao: float = 0.8,
    kappa: float = 4,
    log: bool = False,
) -> Any:
    """Runs the simulation by solving the IVP

    Args:
        s0 (float, optional): The initial susceptible population proportion. Defaults to 0.99.
        i0 (float, optional): The initial infected population proportion. Defaults to 0.01.
        r0 (float, optional): The initial recovered population proportion. Defaults to 0.
        tao (float, optional): The infection rate. Defaults to 0.8.
        kappa (float, optional): The recovery time. Defaults to 4.
        log (bool, optional): Whether or not to log the results. Defaults to False.

    Returns:
        Any: The solution of the IVP
    """
    assert i0 + s0 + r0 == 1, "Initial conditions must sum to 1"

    x0 = [s0, i0, r0]

    def stopping_condition(t, x, tao, kappa):
        _, i, _ = x
        return i - 1e-4

    stopping_condition.terminal = True

    start_t = 0
    end_t = 100
    N = 1000

    result = scipy.integrate.solve_ivp(
        f,
        (0, 100),
        x0,
        events=[stopping_condition],
        args=(tao, kappa),
        t_eval=np.linspace(start_t, end_t, N),
    )

    if log:
        print(result)

    return result


def unpack_values(result: Any) -> Tuple[np.array, np.array, np.array, np.array, float]:
    """Unpacks the values from the results

    Args:
        result (Any): The simulation solution results

    Returns:
        Tuple[np.array, np.array, np.array, np.array, float]: time array, susceptible array, infected array, recovered array, t at which the stopping condition was met
    """
    return result.t, result.y[0], result.y[1], result.y[2], result.t_events[0][0]


def get_results(title: str) -> Union[None, Any]:
    """Gets the results from title string

    Args:
        title (str): The name of the folder to get results

    Returns:
        Union[None, Any]: The results or None if the files do not exist
    """
    dr = f"results/sims/{title}/"
    if os.path.isdir(dr):
        with open(f"{dr}/sol.pkl", "rb") as f:
            return pickle.load(f)
    return None


def store_results(title: str, sol: Any) -> None:
    """Stores the results into files

    Args:
        title (str): The folder to create and store the results to
        sol (Any): The simulation solution results
    """
    dr = f"results/sims/{title}/"
    if not os.path.isdir(dr):
        os.mkdir(dr)
    with open(f"{dr}/sol.pkl", "wb") as f:
        pickle.dump(sol, f)


def simulation_results(
    s0: float = 0.99,
    i0: float = 0.01,
    r0: float = 0,
    tao: float = 0.8,
    kappa: float = 4,
    log: bool = False,
    force_run: bool = False,
    plot: bool = False,
) -> Any:
    """Gets the simulation results either through a run or from stored results and plots them

    Args:
        s0 (float, optional): The initial susceptible population proportion. Defaults to 0.99.
        i0 (float, optional): The initial infected population proportion. Defaults to 0.01.
        r0 (float, optional): The initial recovered population proportion. Defaults to 0.
        tao (float, optional): The infection rate. Defaults to 0.8.
        kappa (float, optional): The recovery time. Defaults to 4.
        log (bool, optional): Whether to print the full results. Defaults to False.
        force_run (bool, optional): Whether to force a new run of the simulation. Defaults to False.
        plot (bool, optional): Whether or not to show the plot of the results. Defaults to False

    Returns:
        Any: The simulation solution results including many solution properties
    """
    title = f"sir_model_s0_{s0}_i0_{i0}_r0_{r0}_tao_{tao}_kappa_{kappa}"

    sol = get_results(title) if not force_run else None
    loaded = not sol is None

    if not loaded:
        sol = run_simulation(s0=s0, i0=i0, r0=r0, tao=tao, kappa=kappa, log=log)
        store_results(title, sol)

    t, s, i, r, stop_t = unpack_values(sol)

    print("Stopping Condition at", stop_t)

    plt.plot(t, s, label="Susceptible %")
    plt.plot(t, i, label="Infected %")
    plt.plot(t, r, label="Recovered %")

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("proportion of population")

    plt.legend()
    if not loaded:
        plt.savefig(f"results/sims/{title}/plot.png")

    if plot:
        plt.show()

    return sol


def get_args() -> argparse.Namespace:
    """Gets the command line arguments passed in (and fills in default values)

    Returns:
        argparse.Namespace: The arguments
    """

    parser = argparse.ArgumentParser(
        prog="Simulation Runner",
        description="Runs the SIR simulation",
    )
    parser.add_argument(
        "-s",
        "--s0",
        type=float,
        default=0.99,
        help="The initial susceptible population proportion",
    )
    parser.add_argument(
        "-i",
        "--i0",
        type=float,
        default=0.01,
        help="The initial infected population proportion",
    )
    parser.add_argument(
        "-r",
        "--r0",
        type=float,
        default=0,
        help="The initial recovered population proportion",
    )
    parser.add_argument(
        "-t",
        "--tao",
        type=float,
        default=0.8,
        help="The infection or spread rate parameter",
    )
    parser.add_argument(
        "-k", "--kappa", type=float, default=4, help="The recovery time parameter"
    )

    parser.add_argument(
        "-l",
        "--log",
        action="store_true",
        default=False,
        help="Whether or not to log the full results",
    )
    parser.add_argument(
        "-f",
        "--force_run",
        action="store_true",
        default=False,
        help="Whether or not to force a new simulation run",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        default=False,
        help="Whether or not to plot the results",
    )

    return parser.parse_args()


def main() -> None:
    """main runner function"""
    args = get_args()
    _ = simulation_results(
        s0=args.s0,
        i0=args.i0,
        r0=args.r0,
        tao=args.tao,
        kappa=args.kappa,
        log=args.log,
        force_run=args.force_run,
    )


if __name__ == "__main__":
    main()
