import numpy as np
from elites import PyMapElites

# Example problem: 2D vector optimization
def evaluate(genome):
    """Returns (fitness, features)"""
    # Fitness is negative sum of squares (maximizing)
    fitness = -sum(x**2 for x in genome)
    # Features are the first two dimensions normalized to [0,1]
    features = [(x + 1.0) / 2.0 for x in genome[:2]]
    return fitness, features

def mutate(genome):
    """Add Gaussian noise to genome"""
    return [g + np.random.normal(0, 0.1) for g in genome]

def random_genome():
    """Generate random initial solution"""
    return list(np.random.uniform(-1, 1, size=2))

def main():
    # Create MAP-Elites instance
    map_elites = PyMapElites(
        feature_dimensions=2,
        bins_per_dimension=10,
        initial_population=100
    )

    # Run the algorithm
    map_elites.run(
        iterations=1000,
        evaluate_fn=evaluate,
        mutate_fn=mutate,
        random_fn=random_genome
    )

    # Get results
    solutions = map_elites.get_solutions()
    stats = map_elites.get_statistics()
    coverage = map_elites.coverage()

    print(f"Found {len(solutions)} solutions")
    print(f"Coverage: {coverage:.2f}")
    print("Statistics:", stats)

    # Print some example solutions
    for bins, (genome, fitness, features) in list(solutions.items())[:5]:
        print(f"\nBin {bins}:")
        print(f"  Genome: {[f'{x:.3f}' for x in genome]}")
        print(f"  Fitness: {fitness:.3f}")
        print(f"  Features: {[f'{x:.3f}' for x in features]}")

if __name__ == "__main__":
    main()