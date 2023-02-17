import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulation import simulation_results, unpack_values
from vaccination_polices import make_parameterized_policy, try_policy


def neural_policy(s, i, r, v, weights1, bias1, weights2, bias2):
    input_vec = np.array([[s, i, r, v]]).T

    cur = input_vec

    cur = weights1 @ cur + bias1
    cur = cur * (cur > 0)
    cur = weights2 @ cur + bias2
    cur = 1 / (1 + np.exp(-cur))

    cur = cur[0][0]
    return cur * s

def neat_algorithm() -> None:
    """Runs the NEAT algorithm (Neuro-Evolution of Augmenting Topologies) to optimize the score defined by alpha and beta and then stores the model on disk
    """
    
    alpha = 0.5
    beta = 0.5

    n_iterations = 100

    n_population = 50
    n_keep = 1

    hidden_layer_dim = 10

    plot_learning_curve = True

    population = [(
        np.random.random((hidden_layer_dim, 4)) * 2 - 1,
        np.random.random((hidden_layer_dim, 1)) * 2 - 1,
        np.random.random((1, hidden_layer_dim)) * 2 - 1,
        np.random.random((1, 1)) * 2 - 1,
    ) for _ in range(n_population)]

    def score(stop_t, final_v):
        return -alpha * stop_t / 500 - beta * final_v

    def fitness(weights1, bias1, weights2, bias2, name="0"):
        policy = make_parameterized_policy(name=f"neural_policy_{name}", weights1=weights1, bias1=bias1, weights2=weights2, bias2=bias2)(neural_policy)

        sol = simulation_results(
            log=False,
            force_run=True,
            show_plot=False,
            generate_plot=False,
            save_results=False,
            vaccination_policy=policy,
        )
        t, s, i, r, v, stop_t = unpack_values(sol)

        return score(stop_t, v[-1])
    
    def breed(e1, e2):
        return tuple(
            (e1[i] + e2[i]) / 2
            for i in range(4)
        )
    
    def mutate(e):
        return tuple(
            e[i] + np.random.normal(0, 0.5, e[i].shape)
            for i in range(4)
        )

    learning_curve = []
    
    iters = tqdm(range(n_iterations))
    for itr in iters:
        fitnesses = []
        for i, e in enumerate(population):
            weights1, bias1, weights2, bias2 = e
            fitnesses.append(fitness(weights1, bias1, weights2, bias2, name=i))

        p = np.array(fitnesses)
        p = p + 3

        learning_curve.append(p.max())

        iters.set_description(f"Processing generation {itr}; Current best score: {learning_curve[-1]}", refresh=True)

        thresh = np.partition(p, -10)[-10]

        p[p < thresh] = 0

        p = p / p.sum()

        parents_idx = np.random.choice(n_population, (n_population - n_keep, 2), p=p)

        new_pop = [population[j] for j in list(sorted(range(n_population), key=lambda i: fitnesses[i]))[:n_keep]]

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

    np.savetxt("results/models/weights1.csv", weights1, delimiter=',')
    np.savetxt("results/models/bias1.csv", bias1, delimiter=',')
    np.savetxt("results/models/weights2.csv", weights2, delimiter=',')
    np.savetxt("results/models/bias2.csv", bias2, delimiter=',')

    if plot_learning_curve:
       plt.plot(learning_curve)
       plt.title("NEAT learning curve")
       plt.xlabel("generation")
       plt.ylabel("best score")
       plt.show()

def test_neural_policy() -> None:
    """Tries out the current best neural policy generated from the NEAT algorithm above with default simulation parameters
    """
    hidden_layer_dim = 10
    weights1 = np.loadtxt("results/models/weights1.csv", delimiter=',', ).reshape((hidden_layer_dim, 4))
    bias1 = np.loadtxt("results/models/bias1.csv", delimiter=',').reshape((hidden_layer_dim, 1))
    weights2 = np.loadtxt("results/models/weights2.csv", delimiter=',').reshape((1, hidden_layer_dim))
    bias2 = np.loadtxt("results/models/bias2.csv", delimiter=',').reshape((1, 1))

    policy = make_parameterized_policy(name=f"neural_policy_0", weights1=weights1, bias1=bias1, weights2=weights2, bias2=bias2)(neural_policy)

    try_policy(policy)

def main() -> None:
    """main runner method
    """
    neat_algorithm()
    test_neural_policy()

if __name__ == "__main__":
    main()