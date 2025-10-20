# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from pcgym.envs.helper import get_string_map

#######################################################################
# Exercise 1: Evolutionary Algorithms
#######################################################################

# Evaluate the Himmelblau function for a single vectorised individual.
# f(x,y)=(x^{2}+y-11)^{2}+(x+y^{2}-7)^{2}
@partial(np.vectorize, signature="(d)->()")
def himmelblau_function(pop):  # this is kind of our fitness function (except we a minimizing)
    return (pop[0]**2 + pop[1] - 11)**2 + (pop[0] + pop[1]**2 - 7)**2

# Return the lower and upper bounds for each dimension of the search space.
def himmelblau_domain():
    return -5.0, 5.0

# Apply Gaussian noise to a population and clamp it back into the domain.
def mutate(sigma, pop, domain): 
    mutated = pop + np.random.normal(0, sigma, pop.shape)
    return np.clip(mutated, domain[0], domain[1])

# Run one (mu+lambda)-ES step returning the next parent population and their loss.
def step_es(pop, domain, cfg):
    # Evaluate the current parent population.
    parents_loss = himmelblau_function(pop)
    mu = pop.shape[0]
    proportion = getattr(cfg, "proportion", 1.0)
    if proportion <= 0:
        raise ValueError("cfg.proportion must be greater than 0 for ES.")

    # Determine the offspring count (lambda) based on the desired selection proportion.
    lambda_ = max(mu, int(np.ceil(mu / proportion)))
    # Sample parents with replacement to produce lambda offspring that mutate independently.
    parent_indices = np.random.randint(0, mu, size=lambda_)
    offspring = mutate(cfg.sigma, pop[parent_indices], domain)
    offspring_loss = himmelblau_function(offspring)

    # Combine the parents and offspring for (mu + lambda) survivor selection.
    combined = np.vstack((pop, offspring))
    combined_loss = np.concatenate((parents_loss, offspring_loss))
    # Pick the indices of the lowest-loss individuals and keep the top mu as the new parents.
    order = np.argsort(combined_loss)
    survivors = combined[order[:mu]]
    survivor_loss = combined_loss[order[:mu]]
    return survivors, survivor_loss

# Run one classic GA step returning the newly created offspring population and their loss.
def step_ga(pop, domain, cfg):
    # Evaluate current parents and select the top fraction according to proportion.
    parent_loss = himmelblau_function(pop)
    mu = pop.shape[0]
    num_selected = max(2, int(np.ceil(mu * getattr(cfg, "proportion", 1.0))))
    selected_indices = np.argsort(parent_loss)[:num_selected]
    parents = pop[selected_indices]

    # Sample parent pairs with replacement and blend them using uniform random alphas.
    pair_indices = np.random.randint(0, parents.shape[0], size=(mu, 2))
    alphas = np.random.rand(mu, 1)
    offspring = alphas * parents[pair_indices[:, 0]] + (1.0 - alphas) * parents[pair_indices[:, 1]]

    # Mutate offspring and clamp them back into the domain.
    offspring = mutate(cfg.sigma, offspring, domain)
    offspring_loss = himmelblau_function(offspring)
    return offspring, offspring_loss

# Render a heatmap of the provided fitness function across the search domain.
def our_plot_function(fn):
    lower, upper = himmelblau_domain()
    x1 = np.linspace(lower, upper, 200)
    x2 = np.linspace(lower, upper, 200)
    xs = np.stack(np.meshgrid(x1, x2), axis=-1)
    ys = fn(xs)
    plt.imshow(ys, cmap="viridis", extent=[lower, upper, lower, upper], origin="lower")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def entry_evolutionary_algorithm(cfg):
    if cfg.dimensions != 2:
        raise ValueError("Himmelblau function is defined for 2 dimensions")

    algorithm = getattr(cfg, "algorithm", "ES").lower()
    if algorithm == "es":
        step_fn = step_es
    elif algorithm == "ga":
        step_fn = step_ga
    else:
        raise ValueError("cfg.algorithm must be either 'GA' or 'ES'")

    domain: Tuple[float, float] = himmelblau_domain()
    # Draw the initial set of parents uniformly from the search space.
    pop = np.random.uniform(domain[0], domain[1], (cfg.population, cfg.dimensions))
    fitnesses = []

    for gen in range(cfg.generation):
        # Advance the evolutionary process by one generation and track the best fitness to visualise convergence.
        pop, fitness = step_fn(pop, domain, cfg)
        fitnesses.append(fitness.min())
        print(f"[{algorithm.upper()}] Generation {gen}: Best fitness = {fitness.min()}")

    our_plot_function(himmelblau_function)
    plt.plot(fitnesses)
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.show()

#######################################################################
# Exercise 2: Content Generation
#######################################################################

# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    """Initialise a PCG Gym environment and return a random map population."""
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    # Create a population of random tile indices matching the environment map shape.
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    return env, pop

# Compute fitness and behavioural descriptors for a single level layout.
def evaluate_level(env: PcgrlEnv, tile_map: np.ndarray):
    string_map = get_string_map(tile_map, env._prob.get_tile_types())
    stats = env._prob.get_stats(string_map)

    # Fitness: inverse distance from the goal (dist-win == 0 means the goal is reachable).
    dist_to_goal = stats.get("dist-win", np.inf)
    fitness = 1.0 / (1.0 + dist_to_goal)

    # Behavioural descriptors: non-fitness traits to characterise the level.
    behaviours = {
        "jumps": stats.get("jumps", 0),
        "enemies": stats.get("enemies", 0),
        "empty": stats.get("empty", 0),
    }

    return fitness, behaviours, stats

# Randomly perturb tiles in the level while keeping the map shape intact.
def mutate_level(level: np.ndarray, tile_count: int, mutation_rate: float) -> np.ndarray:
    mutated = level.copy()
    mask = np.random.rand(*mutated.shape) < mutation_rate
    if mask.any():
        mutated[mask] = np.random.randint(0, tile_count, size=int(mask.sum()))
    return mutated

# Discretise behaviour descriptors into archive coordinates.
def behaviour_key(behaviours: dict, cfg) -> tuple[int, int, int]:
    def discretise(value: float, max_value: float, bins: int) -> int:
        if bins <= 1:
            return 0
        clipped = np.clip(value, 0, max_value)
        if max_value <= 0:
            return 0
        if clipped >= max_value:
            return bins - 1
        return int((clipped / max_value) * bins)

    jumps_bin = discretise(behaviours.get("jumps", 0), getattr(cfg, "jumps_max", 20), getattr(cfg, "jumps_bins", 10))
    enemies_bin = discretise(
        behaviours.get("enemies", 0), getattr(cfg, "enemies_max", 300), getattr(cfg, "enemies_bins", 10)
    )
    empty_bin = discretise(behaviours.get("empty", 0), getattr(cfg, "empty_max", 500), getattr(cfg, "empty_bins", 10))
    return jumps_bin, enemies_bin, empty_bin

# Run a MAP-Elites loop over level layouts and return the archive and fitness trace.
def run_map_elites(env: PcgrlEnv, cfg, initial_population: np.ndarray):
    tile_count = env.get_num_tiles()
    map_shape = env._rep._map.shape
    budget = getattr(cfg, "map_budget", 200)
    init_fraction = getattr(cfg, "map_init_fraction", 0.1)
    mutation_rate = getattr(cfg, "map_mutation_rate", 0.05)
    init_random = max(1, int(budget * init_fraction))

    archive: dict[tuple[int, int, int], dict] = {}
    best_fitness_trace: list[float] = []

    def evaluate_candidate(level: np.ndarray):
        fitness, behaviours, stats = evaluate_level(env, level)
        key = behaviour_key(behaviours, cfg)
        cell = archive.get(key)
        if cell is None or fitness > cell["fitness"]:
            archive[key] = {
                "fitness": fitness,
                "behaviours": behaviours,
                "level": level,
                "stats": stats,
            }
        best_fitness_trace.append(max(best_fitness_trace[-1], fitness) if best_fitness_trace else fitness)

    evaluations = 0

    # Use the provided initial population first (counts towards the budget).
    for level in initial_population:
        if evaluations >= budget:
            break
        evaluate_candidate(level)
        evaluations += 1

    # Continue the MAP-Elites process until the evaluation budget is exhausted.
    while evaluations < budget:
        if evaluations < init_random or not archive:
            candidate = np.random.randint(0, tile_count, size=map_shape)
        else:
            keys = list(archive.keys())
            parent_key = keys[np.random.randint(0, len(keys))]
            parent_level = archive[parent_key]["level"]
            candidate = mutate_level(parent_level, tile_count, mutation_rate)

        evaluate_candidate(candidate)
        evaluations += 1

    return archive, best_fitness_trace


def entry_content_generation(cfg):
    env, pop = init_pcgym(cfg)
    archive, best_trace = run_map_elites(env, cfg, pop)

    total_cells = (
          getattr(cfg, "jumps_bins", 10)
        * getattr(cfg, "enemies_bins", 10)
        * getattr(cfg, "empty_bins", 10)
    )
    print(f"Filled {len(archive)} / {total_cells} archive cells.")

    if archive:
        best_entry = max(archive.values(), key=lambda cell: cell["fitness"])
        print(
            "Best elite -> fitness={:.4f}, behaviours={}, stats={}".format(
                best_entry["fitness"], best_entry["behaviours"], best_entry["stats"]
            )
        )
    else:
        print("Archive is empty")

    if best_trace:
        trace = np.clip(np.array(best_trace), 1e-6, None)
        plt.figure()
        plt.plot(trace)
        plt.yscale("log")
        plt.xlabel("Evaluations")
        plt.ylabel("Best fitness so far")
        plt.title("MAP-Elites convergence")
        plt.show()

#######################################################################
# Entry point
#######################################################################

# %% Setup
def main(cfg):
    match cfg.exercise:
        case "evolutionary_algorithm":
            entry_evolutionary_algorithm(cfg)
        case "content_generation":
            entry_content_generation(cfg)
        case _:
            raise ValueError("Exercise not recognized")
