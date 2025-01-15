import time
import numpy as np
from elites import PyMapElites
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def create_sphere_problem(dimensions=2):
    """Simple sphere function for testing"""
    def evaluate(genome):
        fitness = -sum(x**2 for x in genome)  # Negative for maximization
        features = [(x + 1.0) / 2.0 for x in genome[:2]]
        return fitness, features

    def mutate(genome):
        return [g + np.random.normal(0, 0.1) for g in genome]

    def random_genome():
        return list(np.random.uniform(-1, 1, size=dimensions))

    return evaluate, mutate, random_genome

def create_rastrigin_problem(dimensions=2):
    """Rastrigin function - more complex test problem"""
    def evaluate(genome):
        A = 10
        fitness = -(A * dimensions + sum(x**2 - A * np.cos(2 * np.pi * x) for x in genome))
        features = [(x + 5.12) / 10.24 for x in genome[:2]]  # Scale from [-5.12, 5.12] to [0,1]
        return fitness, features

    def mutate(genome):
        return [min(5.12, max(-5.12, g + np.random.normal(0, 0.2))) for g in genome]

    def random_genome():
        return list(np.random.uniform(-5.12, 5.12, size=dimensions))

    return evaluate, mutate, random_genome

def run_benchmark(problem_name, evaluate, mutate, random_genome, iterations=10000, initial_pop=100, bins=10):
    print(f"\nRunning benchmark: {problem_name}")
    print("-" * (len(problem_name) + 19))
    
    map_elites = PyMapElites(
        feature_dimensions=2,
        bins_per_dimension=bins,
        initial_population=initial_pop
    )
    
    start_time = time.time()
    
    with tqdm(total=iterations + initial_pop) as pbar:
        map_elites.run(iterations, evaluate, mutate, random_genome)
        pbar.update(iterations + initial_pop)
    
    end_time = time.time()
    
    solutions = map_elites.get_solutions()
    coverage = map_elites.coverage()
    stats = map_elites.get_statistics()
    
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Solutions found: {len(solutions)}")
    print(f"Coverage: {coverage:.2f}")
    print(f"Best fitness: {stats[2]:.3f}")
    print(f"Evaluations per second: {(iterations + initial_pop)/(end_time - start_time):.1f}")
    
    return {
        'time': end_time - start_time,
        'solutions': len(solutions),
        'coverage': coverage,
        'best_fitness': stats[2],
        'evals_per_second': (iterations + initial_pop)/(end_time - start_time)
    }

def plot_scaling_results(results, save_path="scaling_analysis.png"):
    """Create detailed scaling plots"""
    plt.style.use('seaborn')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    dimensions = []
    sphere_speed = []
    sphere_coverage = []
    rastrigin_speed = []
    rastrigin_coverage = []
    
    for name, result in results.items():
        dim = int(name.split('_')[1][:-1])  # Extract dimension number
        if 'sphere' in name:
            dimensions.append(dim)
            sphere_speed.append(result['evals_per_second'])
            sphere_coverage.append(result['coverage'])
        else:
            rastrigin_speed.append(result['evals_per_second'])
            rastrigin_coverage.append(result['coverage'])
    
    # Performance Scaling
    ax1.plot(dimensions, sphere_speed, 'o-', label='Sphere', linewidth=2)
    ax1.plot(dimensions, rastrigin_speed, 's-', label='Rastrigin', linewidth=2)
    ax1.set_xlabel('Dimensions')
    ax1.set_ylabel('Evaluations per Second')
    ax1.set_title('Performance Scaling')
    ax1.legend()
    ax1.grid(True)
    
    # Coverage
    ax2.plot(dimensions, sphere_coverage, 'o-', label='Sphere', linewidth=2)
    ax2.plot(dimensions, rastrigin_coverage, 's-', label='Rastrigin', linewidth=2)
    ax2.set_xlabel('Dimensions')
    ax2.set_ylabel('Coverage')
    ax2.set_title('Solution Coverage vs Dimensions')
    ax2.legend()
    ax2.grid(True)
    
    # Log-scale Performance
    ax3.plot(dimensions, sphere_speed, 'o-', label='Sphere', linewidth=2)
    ax3.plot(dimensions, rastrigin_speed, 's-', label='Rastrigin', linewidth=2)
    ax3.set_yscale('log')
    ax3.set_xlabel('Dimensions')
    ax3.set_ylabel('Evaluations per Second (log scale)')
    ax3.set_title('Performance Scaling (Log Scale)')
    ax3.legend()
    ax3.grid(True)
    
    # Relative Performance
    relative_sphere = [s/sphere_speed[0] for s in sphere_speed]
    relative_rastrigin = [s/rastrigin_speed[0] for s in rastrigin_speed]
    
    ax4.plot(dimensions, relative_sphere, 'o-', label='Sphere', linewidth=2)
    ax4.plot(dimensions, relative_rastrigin, 's-', label='Rastrigin', linewidth=2)
    ax4.set_xlabel('Dimensions')
    ax4.set_ylabel('Relative Performance')
    ax4.set_title('Relative Performance vs 2D Baseline')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Test configurations
    DIMENSIONS = [2, 5, 10, 20, 50]  # Added 50D for better scaling analysis
    ITERATIONS = 10000
    INITIAL_POP = 100
    BINS = 10
    
    results = {}
    
    # Run benchmarks
    for dims in DIMENSIONS:
        # Sphere function
        sphere_fns = create_sphere_problem(dims)
        results[f'sphere_{dims}d'] = run_benchmark(
            f"Sphere function ({dims}D)",
            *sphere_fns,
            iterations=ITERATIONS,
            initial_pop=INITIAL_POP,
            bins=BINS
        )
        
        # Rastrigin function
        rastrigin_fns = create_rastrigin_problem(dims)
        results[f'rastrigin_{dims}d'] = run_benchmark(
            f"Rastrigin function ({dims}D)",
            *rastrigin_fns,
            iterations=ITERATIONS,
            initial_pop=INITIAL_POP,
            bins=BINS
        )
    
    # Create scaling plots
    plot_scaling_results(results)
    
    # Print summary
    print("\nBenchmark Summary")
    print("================")
    for name, result in sorted(results.items()):
        print(f"\n{name}:")
        print(f"  Evaluations/second: {result['evals_per_second']:.1f}")
        print(f"  Coverage: {result['coverage']:.2f}")
        print(f"  Best fitness: {result['best_fitness']:.3f}")

if __name__ == "__main__":
    main() 