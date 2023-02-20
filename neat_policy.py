from typing import Callable

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

from simulation import simulation_results, unpack_values
from vaccination_polices import (
    make_parameterized_policy,
    get_saved_neural_policy,
    hidden_layer_dim,
    neural_policy,
)
from vaccination_tester import try_policy

tao = 0.8
kappa = 4

def neat_algorithm() -> None:
    """Runs the NEAT algorithm (Neuro-Evolution of Augmenting Topologies) to optimize the score defined by alpha and beta and then stores the model on disk"""
    n_iterations = 300

    n_population = 50
    n_keep = 1

    # tao_rng = (0, 4)
    # kappa_rng = (1, 9)
    # tao_N = 20
    # kappa_N = 30

    tao_rng = (tao, tao)
    kappa_rng = (kappa, kappa)
    tao_N = 1
    kappa_N = 1

    plot_learning_curve = True

    population = [
        (
            np.random.random((hidden_layer_dim, 6)) * 2 - 1,
            np.random.random((hidden_layer_dim, 1)) * 2 - 1,
            np.random.random((1, hidden_layer_dim)) * 2 - 1,
            np.random.random((1, 1)) * 2 - 1,
        )
        for _ in range(n_population)
    ]

    def fitness(weights1, bias1, weights2, bias2, name="0"):
        policy = make_parameterized_policy(
            name=f"neural_policy_{name}",
            weights1=weights1,
            bias1=bias1,
            weights2=weights2,
            bias2=bias2,
        )(neural_policy)

        score = 0

        tao_space = np.random.uniform(*tao_rng, size=(tao_N,))
        kappa_space = np.random.uniform(*kappa_rng, size=(kappa_N,))

        for i, j in list(itertools.product(range(tao_N), range(kappa_N))):
            sol = simulation_results(
                tao=tao_space[i],
                kappa=kappa_space[j],
                log=False,
                force_run=True,
                show_plot=False,
                generate_plot=False,
                save_results=False,
                vaccination_policy=policy,
            )

            t, s, i, r, v, stop_t = unpack_values(sol)

            score += s[-1]
        return score

    def breed(e1, e2):
        return tuple((e1[i] + e2[i]) / 2 for i in range(4))

    def mutate(e):
        return tuple(e[i] + np.random.normal(0, 0.5, e[i].shape) for i in range(4))

    learning_curve = []

    iters = tqdm(range(n_iterations))
    for itr in iters:
        fitnesses = []
        for i, e in enumerate(population):
            weights1, bias1, weights2, bias2 = e
            fitnesses.append(fitness(weights1, bias1, weights2, bias2, name=i))

        p = np.array(fitnesses)
        p = p + 1

        learning_curve.append(p.max())

        iters.set_description(
            f"Processing generation {itr}; Current best score: {learning_curve[-1]}",
            refresh=True,
        )

        thresh = np.partition(p, -10)[-10]

        p[p < thresh] = 0

        p = (p - p.min()) / p.sum()

        parents_idx = np.random.choice(n_population, (n_population - n_keep, 2), p=p)

        new_pop = [
            population[j]
            for j in list(sorted(range(n_population), key=lambda i: fitnesses[i]))[
                :n_keep
            ]
        ]

        for i in range(n_population - n_keep):
            parent_1 = population[parents_idx[i][0]]
            parent_2 = population[parents_idx[i][1]]
            child = mutate(breed(parent_1, parent_2))
            new_pop.append(child)

        population = new_pop

    fitnesses = []
    for i, e in enumerate(population):
        weights1, bias1, weights2, bias2 = e
        fitnesses.append(fitness(weights1, bias1, weights2, bias2, name=i))

    best = population[max(range(n_population), key=lambda i: fitnesses[i])]

    weights1, bias1, weights2, bias2 = best

    dir_name = f"tao_{tao_rng[0]}_{tao_rng[1]}_kappa_{kappa_rng[0]}_{kappa_rng[1]}"

    if os.path.isdir(f"results/models/{dir_name}"):
        shutil.rmtree(f"results/models/{dir_name}")
    os.mkdir(f"results/models/{dir_name}")

    np.savetxt(f"results/models/{dir_name}/weights1.csv", weights1, delimiter=",")
    np.savetxt(f"results/models/{dir_name}/bias1.csv", bias1, delimiter=",")
    np.savetxt(f"results/models/{dir_name}/weights2.csv", weights2, delimiter=",")
    np.savetxt(f"results/models/{dir_name}/bias2.csv", bias2, delimiter=",")

    if plot_learning_curve:
        learning_curve = np.array(learning_curve)
        running_average = np.convolve(learning_curve, np.ones(5) / 5, mode="valid")
        plt.plot(learning_curve, label="scores")
        plt.plot(running_average, label="running average")
        plt.title("NEAT learning curve")
        plt.xlabel("generation")
        plt.ylabel("best score")
        plt.legend()
        plt.savefig(f"results/models/{dir_name}/learning_curve.png")
        plt.show()


def test_neural_policy() -> None:
    """Tries out the current best neural policy generated from the NEAT algorithm above with default simulation parameters"""
    policy = get_saved_neural_policy(tao=tao, kappa=kappa)
    try_policy(policy, tao=tao, kappa=kappa)


def main() -> None:
    """main runner method"""
    neat_algorithm()
    test_neural_policy()


if __name__ == "__main__":
    main()
