from typing import Tuple

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

from simulation import simulation_results, unpack_values


def run_simulations_over_parameter_space(
    tao_space: np.array,
    kappa_space: np.array,
    s0: float = 0.99,
    i0: float = 0.01,
    r0: float = 0,
) -> np.array:
    """Runs the SIR simulation over the specific configuration spaces

    Args:
        tao_space (np.array): The tao space to run over with shape (n,)
        kappa_space (np.array): The kappa space to run over with shape (m,)
        s0 (float, optional): The initial susceptible population proportion. Defaults to 0.99.
        i0 (float, optional): The initial infected population proportion. Defaults to 0.01.
        r0 (float, optional): The initial recovered population proportion. Defaults to 0.

    Returns:
        np.array: a 2D numpy array with shape (n, m) containing all the stopping times from the runs
    """
    n = tao_space.shape[0]
    m = kappa_space.shape[0]
    stopping_ts = np.zeros((n, m))

    for i, j in tqdm(list(itertools.product(range(n), range(m)))):
        sol = simulation_results(
            s0=s0,
            i0=i0,
            r0=r0,
            tao=tao_space[i],
            kappa=kappa_space[j],
            log=False,
            force_run=False,
            show_plot=False,
            generate_plot=False,
        )
        _, _, _, _, stop_t = unpack_values(sol)
        stopping_ts[i, j] = stop_t

    return stopping_ts.T


def plot_heatmap(
    tao_rng: Tuple[float] = (0, 4),
    kappa_rng: Tuple[float] = (1, 5),
    tao_N: int = 20,
    kappa_N: int = 20,
    s0: float = 0.99,
    i0: float = 0.01,
    r0: float = 0,
    show_plot: bool=False,
) -> None:
    """Plots the heatmap from the ranges of configurations

    Args:
        tao_rng (Tuple[float], optional): The range of tao values. Defaults to (0, 4).
        kappa_rng (Tuple[float], optional): The range of kappa values. Defaults to (1, 5).
        tao_N (int, optional): The number of tao points to use. Defaults to 20.
        kappa_N (int, optional): The number of kappa points to use. Defaults to 20.
        s0 (float, optional): The initial s0 condition. Defaults to 0.99.
        i0 (float, optional): The initial i0 condition. Defaults to 0.01.
        r0 (float, optional): The initial r0 condition. Defaults to 0.
        show_plot (bool, optional): Whether or not to show the plot. Defaults to False.
    """
    tao_space = np.linspace(*tao_rng, tao_N)
    kappa_space = np.linspace(*kappa_rng, kappa_N)
    map = run_simulations_over_parameter_space(
        tao_space=tao_space, kappa_space=kappa_space, s0=s0, i0=i0, r0=r0
    )

    _, ax = plt.subplots(figsize=(14, 8))
    heatmap = ax.pcolor(map)

    ax.set_xticks(np.arange(tao_N) + 0.5)
    ax.set_yticks(np.arange(kappa_N) + 0.5)

    ax.set_xticklabels(tao_space.round(2))
    ax.set_yticklabels(kappa_space.round(2))

    title = f"tao_kappa_stop_t_plot_tao_{tao_rng[0]}_{tao_rng[1]}_{tao_N}_kappa_{kappa_rng[0]}_{kappa_rng[1]}_{kappa_N}"

    plt.colorbar(heatmap)

    plt.title(title)
    plt.xlabel("tao values")
    plt.ylabel("kappa values")

    plt.savefig(f"results/analysis/{title}.png")

    if show_plot:
        plt.show()


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
        "-ts",
        "--tao-start",
        type=float,
        default=0,
        help="The infection or spread rate parameter start",
    )
    parser.add_argument(
        "-ks",
        "--kappa-start",
        type=float,
        default=1,
        help="The recovery time parameter start",
    )
    parser.add_argument(
        "-te",
        "--tao-end",
        type=float,
        default=4,
        help="The infection or spread rate parameter end",
    )
    parser.add_argument(
        "-ke",
        "--kappa-end",
        type=float,
        default=5,
        help="The recovery time parameter end",
    )
    parser.add_argument(
        "-tn",
        "--tao-n",
        type=int,
        default=20,
        help="The infection or spread rate parameter number of points",
    )
    parser.add_argument(
        "-kn",
        "--kappa-n",
        type=int,
        default=20,
        help="The recovery time parameter number of points",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Whether or not to display the plotted results",
    )

    return parser.parse_args()


def main() -> None:
    """main runner function"""
    args = get_args()
    plot_heatmap(
        tao_rng=(args.tao_start, args.tao_end),
        kappa_rng=(args.kappa_start, args.kappa_end),
        tao_N=args.tao_n,
        kappa_N=args.kappa_n,
        s0=args.s0,
        i0=args.i0,
        r0=args.r0,
        show_plot=args.plot,
    )


if __name__ == "__main__":
    main()
