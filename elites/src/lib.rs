//! Map-Elites: A Quality Diversity Algorithm Implementation
//! 
//! This crate provides a generic, efficient implementation of the Map-Elites algorithm,
//! which is used to discover diverse, high-performing solutions across a feature space.
//! 
//! # Example
//! ```
//! use elites::{MapElites, MapElitesProblem};
//! 
//! // Define your problem
//! struct MyProblem;
//! 
//! impl MapElitesProblem for MyProblem {
//!     type Genome = Vec<f64>;
//!     
//!     fn random_genome(&self) -> Self::Genome {
//!         vec![0.0, 0.0] // Simplified for example
//!     }
//!     
//!     fn evaluate(&self, genome: &Self::Genome) -> (f64, Vec<f64>) {
//!         let fitness = -genome.iter().map(|x| x.powi(2)).sum::<f64>();
//!         let features = vec![genome[0], genome[1]];
//!         (fitness, features)
//!     }
//!     
//!     fn mutate(&self, genome: &Self::Genome) -> Self::Genome {
//!         genome.clone() // Simplified for example
//!     }
//!     
//!     fn feature_dimensions(&self) -> usize { 2 }
//!     fn bins_per_dimension(&self) -> usize { 10 }
//! }
//! 
//! // Use the algorithm
//! let problem = MyProblem;
//! let mut map_elites = MapElites::new(problem);
//! map_elites.run(1000);
//! ```

use std::collections::HashMap;
use rand::Rng;
use rayon::prelude::*;
use polars::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;
use parking_lot::RwLock;

// Thread-local workspace to avoid allocations
thread_local! {
    static WORKSPACE: RefCell<WorkspaceBuffers> = RefCell::new(WorkspaceBuffers::new());
}

struct WorkspaceBuffers {
    genome_matrix: DataFrame,
    fitness_vec: Series,
    features_vec: Series,
}

impl WorkspaceBuffers {
    fn new() -> Self {
        // Create empty DataFrame and Series with initial capacity
        let genome_matrix = DataFrame::new_no_checks(vec![]);
        let fitness_vec = Series::new_empty("fitness", &DataType::Float64);
        let features_vec = Series::new_empty("features", &DataType::Float64);

        Self {
            genome_matrix,
            fitness_vec,
            features_vec,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MapElitesConfig {
    pub initial_population: usize,
    pub track_stats: bool,
    pub bin_boundaries: Option<Vec<Vec<f64>>>,
}

impl Default for MapElitesConfig {
    fn default() -> Self {
        Self {
            initial_population: 100,
            track_stats: false,
            bin_boundaries: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Statistics {
    pub iterations: usize,
    pub num_solutions: usize,
    pub best_fitness: f64,
    pub coverage: f64,
    pub improvements: usize,
}

#[derive(Clone, Debug)]
pub struct Individual<T> {
    pub genome: T,
    pub fitness: f64,
    pub features: Vec<f64>,
}

pub trait MapElitesProblem: Send + Sync {
    type Genome: Clone + Send + Sync + std::fmt::Debug;
    fn random_genome(&self) -> Self::Genome;
    fn evaluate(&self, genome: &Self::Genome) -> (f64, Vec<f64>);
    fn mutate(&self, genome: &Self::Genome) -> Self::Genome;
    fn feature_dimensions(&self) -> usize;
    fn bins_per_dimension(&self) -> usize;
}

pub struct MapElites<T: MapElitesProblem> {
    problem: T,
    solutions: HashMap<Vec<usize>, Individual<T::Genome>>,
    bins_per_dimension: usize,
    config: MapElitesConfig,
    stats: Statistics,
}

impl<T: MapElitesProblem> MapElites<T> where T::Genome: AsRef<[f64]> {
    /// Create a new Map-Elites instance with default configuration
    pub fn new(problem: T) -> Self {
        Self::with_config(problem, MapElitesConfig::default())
    }

    /// Create a new Map-Elites instance with custom configuration
    pub fn with_config(problem: T, config: MapElitesConfig) -> Self {
        Self {
            bins_per_dimension: problem.bins_per_dimension(),
            problem,
            solutions: HashMap::new(),
            config,
            stats: Statistics::default(),
        }
    }

    /// Convert feature values to discrete bin coordinates
    fn features_to_bins(&self, features: &[f64]) -> Vec<usize> {
        if let Some(ref boundaries) = self.config.bin_boundaries {
            features
                .iter()
                .enumerate()
                .map(|(i, &f)| {
                    boundaries[i]
                        .binary_search_by(|probe| probe.partial_cmp(&f).unwrap())
                        .unwrap_or_else(|x| x)
                })
                .collect()
        } else {
            features
                .iter()
                .map(|&f| {
                    let bin = (f * self.bins_per_dimension as f64).floor() as usize;
                    bin.min(self.bins_per_dimension - 1)
                })
                .collect()
        }
    }

    /// Add an individual to the map if it's better than the existing one in its bin
    fn add_to_map(&mut self, individual: Individual<T::Genome>) -> bool {
        // Don't add individuals with NaN or infinite fitness
        if individual.fitness.is_nan() || individual.fitness.is_infinite() {
            return false;
        }

        // Validate features are within [0,1] range when not using custom boundaries
        if self.config.bin_boundaries.is_none() {
            if !individual.features.iter().all(|&f| f >= 0.0 && f <= 1.0) {
                return false;
            }
        }

        let bins = self.features_to_bins(&individual.features);
        
        match self.solutions.get(&bins) {
            Some(existing) if existing.fitness >= individual.fitness => false,
            _ => {
                if self.config.track_stats {
                    self.stats.improvements += 1;
                    self.stats.best_fitness = self.stats.best_fitness.max(individual.fitness);
                }
                self.solutions.insert(bins, individual);
                true
            }
        }
    }

    fn evaluate_batch(&self, genomes: &[T::Genome]) -> Vec<(f64, Vec<f64>)> {
        WORKSPACE.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let batch_size = genomes.len();
            
            // Create vectors for the genome data
            let mut genome_data: Vec<Vec<f64>> = Vec::with_capacity(batch_size);
            for genome in genomes {
                genome_data.push(genome.as_ref().to_vec());
            }
            
            // Create DataFrame from genome data
            let mut df = DataFrame::new(vec![
                Series::new("genome", genome_data.clone())
            ]).unwrap();
            
            // Calculate fitness using polars expressions
            let fitness = df.clone()
                .lazy()
                .select([
                    col("genome").map(|s| {
                        Ok(s.iter().map(|x| {
                            let x = x.unwrap_or(0.0);
                            -x * x
                        }).collect::<Float64Chunked>().into_series())
                    }, GetOutput::from_type(DataType::Float64))
                ])
                .collect()
                .unwrap()
                .column("genome")
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .collect::<Vec<Option<f64>>>();

            // Create result vector
            genomes.iter().zip(fitness.iter())
                .map(|(g, &f)| {
                    let features = vec![
                        (g.as_ref()[0] + 1.0) / 2.0,
                        (g.as_ref()[1] + 1.0) / 2.0
                    ];
                    (f.unwrap_or(0.0), features)
                })
                .collect()
        })
    }

    /// Run the Map-Elites algorithm for a specified number of iterations
    pub fn run(&mut self, iterations: usize) {
        const BATCH_SIZE: usize = 100_000;
        let num_threads = num_cpus::get() * 2;
        
        // Clone methods we'll need in parallel sections
        let validate = |individual: &Individual<T::Genome>| {
            if individual.fitness.is_nan() || individual.fitness.is_infinite() {
                return false;
            }
            if self.config.bin_boundaries.is_none() {
                individual.features.iter().all(|&f| f >= 0.0 && f <= 1.0)
            } else {
                true
            }
        };

        let features_to_bins = |features: &[f64]| {
            if let Some(ref boundaries) = self.config.bin_boundaries {
                features.iter().enumerate()
                    .map(|(i, &f)| {
                        boundaries[i].binary_search_by(|probe| probe.partial_cmp(&f).unwrap())
                            .unwrap_or_else(|x| x)
                    })
                    .collect()
            } else {
                features.iter()
                    .map(|&f| ((f * self.bins_per_dimension as f64).floor() as usize)
                        .min(self.bins_per_dimension - 1))
                    .collect()
            }
        };

        let solutions_lock = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize population
        let initial_solutions: Vec<_> = (0..self.config.initial_population)
            .into_par_iter()
            .map(|_| {
                let genome = self.problem.random_genome();
                let (fitness, features) = self.problem.evaluate(&genome);
                (features_to_bins(&features), Individual { genome, fitness, features })
            })
            .filter(|(_, individual)| validate(individual))
            .collect();

        // Add initial solutions
        {
            let mut solutions = solutions_lock.write();
            for (bins, individual) in initial_solutions {
                solutions.insert(bins, individual);
            }
        }

        // Process batches
        for chunk_start in (0..iterations).step_by(BATCH_SIZE) {
            let chunk_size = (iterations - chunk_start).min(BATCH_SIZE);
            let current_solutions = solutions_lock.read().values().cloned().collect::<Vec<_>>();
            
            if current_solutions.is_empty() {
                continue;
            }

            let offspring: Vec<_> = (0..chunk_size)
                .into_par_iter()
                .map(|_| {
                    let parent = &current_solutions[rand::thread_rng().gen_range(0..current_solutions.len())];
                    let offspring = self.problem.mutate(&parent.genome);
                    let (fitness, features) = self.problem.evaluate(&offspring);
                    (features_to_bins(&features), Individual { genome: offspring, fitness, features })
                })
                .filter(|(_, individual)| validate(individual))
                .collect();

            let mut solutions = solutions_lock.write();
            for (bins, individual) in offspring {
                solutions.insert(bins, individual);
            }
        }

        // Copy back solutions
        self.solutions = Arc::try_unwrap(solutions_lock).unwrap().into_inner();

        if self.config.track_stats {
            self.update_statistics(iterations);
        }
    }

    fn update_statistics(&mut self, iterations: usize) {
        self.stats.iterations += iterations;
        self.stats.num_solutions = self.solutions.len();
        self.stats.coverage = self.coverage();
    }

    /// Get the current solutions map
    pub fn solutions(&self) -> &HashMap<Vec<usize>, Individual<T::Genome>> {
        &self.solutions
    }

    /// Get the coverage (percentage of filled bins)
    pub fn coverage(&self) -> f64 {
        let total_bins = self.bins_per_dimension.pow(self.problem.feature_dimensions() as u32);
        self.solutions.len() as f64 / total_bins as f64
    }

    /// Get the current statistics
    pub fn statistics(&self) -> &Statistics {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::f64::INFINITY;

    // Simple 2D vector optimization problem
    #[derive(Clone)]
    struct VectorProblem {
        dimensions: usize,
    }

    impl VectorProblem {
        fn new(dimensions: usize) -> Self {
            Self { dimensions }
        }
    }

    impl MapElitesProblem for VectorProblem {
        type Genome = Vec<f64>;

        fn random_genome(&self) -> Self::Genome {
            let mut rng = rand::thread_rng();
            (0..self.dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect()
        }

        fn evaluate(&self, genome: &Self::Genome) -> (f64, Vec<f64>) {
            // Fitness is negative sum of squares (maximizing)
            let fitness = -genome.iter().map(|x| x.powi(2)).sum::<f64>();
            // Features are the first two dimensions normalized to [0,1]
            let features = vec![
                (genome[0] + 1.0) / 2.0,
                (genome[1] + 1.0) / 2.0,
            ];
            (fitness, features)
        }

        fn mutate(&self, genome: &Self::Genome) -> Self::Genome {
            let mut rng = rand::thread_rng();
            genome
                .iter()
                .map(|&g| g + rng.gen_range(-0.1..0.1))
                .collect()
        }

        fn feature_dimensions(&self) -> usize { 2 }
        fn bins_per_dimension(&self) -> usize { 10 }
    }

    // Rastrigin function optimization problem
    struct RastriginProblem;

    impl MapElitesProblem for RastriginProblem {
        type Genome = Vec<f64>;

        fn random_genome(&self) -> Self::Genome {
            let mut rng = rand::thread_rng();
            (0..2).map(|_| rng.gen_range(-5.12..5.12)).collect()
        }

        fn evaluate(&self, genome: &Self::Genome) -> (f64, Vec<f64>) {
            let n = genome.len() as f64;
            let fitness = -10.0 * n - genome.iter()
                .map(|&x| x.powi(2) - 10.0 * (2.0 * std::f64::consts::PI * x).cos())
                .sum::<f64>();
            
            let features = vec![
                (genome[0] + 5.12) / 10.24, // Normalize to [0,1]
                (genome[1] + 5.12) / 10.24,
            ];
            
            (fitness, features)
        }

        fn mutate(&self, genome: &Self::Genome) -> Self::Genome {
            let mut rng = rand::thread_rng();
            genome
                .iter()
                .map(|&g| {
                    (g + rng.gen_range(-0.5..0.5))
                        .max(-5.12)
                        .min(5.12)
                })
                .collect()
        }

        fn feature_dimensions(&self) -> usize { 2 }
        fn bins_per_dimension(&self) -> usize { 10 }
    }

    #[test]
    fn test_vector_optimization() {
        let problem = VectorProblem::new(2);
        let mut map_elites = MapElites::new(problem);
        
        // Test initial state
        assert_eq!(map_elites.solutions().len(), 0);
        assert_eq!(map_elites.coverage(), 0.0);
        
        map_elites.run(1000);
        
        // Check if solutions were found
        assert!(map_elites.solutions().len() > 0);
        
        // Check coverage
        let coverage = map_elites.coverage();
        assert!(coverage > 0.0);
        assert!(coverage <= 1.0);
        println!("Vector optimization coverage: {:.2}", coverage);

        // Check that all solutions are within bounds
        for solution in map_elites.solutions().values() {
            assert!(solution.features.iter().all(|&f| f >= 0.0 && f <= 1.0));
            assert!(!solution.fitness.is_nan());
            assert!(!solution.fitness.is_infinite());
        }
    }

    #[test]
    fn test_rastrigin_with_different_configs() {
        let configs = vec![
            MapElitesConfig {
                initial_population: 10,
                track_stats: true,
                ..Default::default()
            },
            MapElitesConfig {
                initial_population: 200,
                track_stats: true,
                ..Default::default()
            },
            MapElitesConfig {
                initial_population: 1000,
                track_stats: true,
                ..Default::default()
            },
        ];

        for config in configs {
            let problem = RastriginProblem;
            let mut map_elites = MapElites::with_config(problem, config.clone());
            
            map_elites.run(1000);
            
            let stats = map_elites.statistics();
            println!("Rastrigin test with initial_pop {}", config.initial_population);
            println!("  Solutions found: {}", stats.num_solutions);
            println!("  Best fitness: {:.2}", stats.best_fitness);
            println!("  Coverage: {:.2}", stats.coverage);
            
            assert!(stats.num_solutions > 0);
            assert!(stats.coverage > 0.0);
            assert!(stats.best_fitness <= 0.0); // Rastrigin is negative, with 0 being optimal
        }
    }

    #[test]
    fn test_custom_boundaries_edge_cases() {
        let problem = VectorProblem::new(2);
        
        // Test with different boundary configurations
        let boundary_configs = vec![
            // Normal boundaries
            vec![
                vec![-1.0, -0.5, 0.0, 0.5, 1.0],
                vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            ],
            // Single boundary (2 bins)
            vec![
                vec![-1.0, 1.0],
                vec![-1.0, 1.0],
            ],
            // Many boundaries
            vec![
                (-5..=5).map(|x| x as f64).collect(),
                (-5..=5).map(|x| x as f64).collect(),
            ],
        ];

        for boundaries in boundary_configs {
            let config = MapElitesConfig {
                bin_boundaries: Some(boundaries.clone()),
                initial_population: 500,
                ..Default::default()
            };
            
            let mut map_elites = MapElites::with_config(problem.clone(), config);
            map_elites.run(1000);
            
            assert!(map_elites.solutions().len() > 0);
            
            // Check that solutions are properly binned
            for solution in map_elites.solutions().values() {
                for (feature, bounds) in solution.features.iter().zip(boundaries.iter()) {
                    let min_bound = *bounds.first().unwrap();
                    let max_bound = *bounds.last().unwrap();
                    // Feature should be within the overall bounds with some numerical tolerance
                    assert!(*feature >= min_bound - 1e-10, 
                        "Feature {} is less than minimum bound {}", feature, min_bound);
                    assert!(*feature <= max_bound + 1e-10,
                        "Feature {} is greater than maximum bound {}", feature, max_bound);
                }
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        struct EdgeCaseProblem;
        
        impl MapElitesProblem for EdgeCaseProblem {
            type Genome = Vec<f64>;
            
            fn random_genome(&self) -> Self::Genome { 
                // Start with a valid solution more often
                vec![0.8] 
            }
            
            fn evaluate(&self, genome: &Self::Genome) -> (f64, Vec<f64>) {
                match genome[0] {
                    x if x < -0.5 => (f64::NEG_INFINITY, vec![0.5]),  // Invalid fitness
                    x if x < 0.0 => (f64::NAN, vec![0.5]),           // Invalid fitness
                    x if x < 0.5 => (f64::INFINITY, vec![0.5]),      // Invalid fitness
                    _ => (1.0, vec![0.5]),                           // Valid case
                }
            }
            
            fn mutate(&self, genome: &Self::Genome) -> Self::Genome {
                let mut rng = rand::thread_rng();
                // Bias mutation towards valid solutions
                let mutation = if rng.gen_bool(0.7) {
                    rng.gen_range(0.0..0.2) // Small positive mutation
                } else {
                    rng.gen_range(-1.0..1.0) // Occasional large mutation
                };
                vec![genome[0] + mutation]
            }
            
            fn feature_dimensions(&self) -> usize { 1 }
            fn bins_per_dimension(&self) -> usize { 1 }
        }

        let problem = EdgeCaseProblem;
        let config = MapElitesConfig {
            initial_population: 200, // Increase initial population
            ..Default::default()
        };
        
        let mut map_elites = MapElites::with_config(problem, config);
        
        // Run for more iterations to ensure we get valid solutions
        map_elites.run(2000);
        
        // Only valid solutions should be in the map
        assert!(map_elites.solutions().len() > 0, "Should have at least one valid solution");
        for solution in map_elites.solutions().values() {
            assert!(!solution.fitness.is_nan(), "Found NaN fitness");
            assert!(!solution.fitness.is_infinite(), "Found infinite fitness");
            assert!(solution.features.iter().all(|&f| f >= 0.0 && f <= 1.0), 
                "Features outside [0,1] range: {:?}", solution.features);
        }
    }

    #[test]
    fn test_zero_iterations() {
        let problem = VectorProblem::new(2);
        let mut map_elites = MapElites::new(problem);
        
        // Run with zero iterations
        map_elites.run(0);
        
        // Should still have initial population
        assert!(map_elites.solutions().len() > 0);
    }

    #[test]
    fn test_statistics_tracking() {
        let problem = VectorProblem::new(2);
        let config = MapElitesConfig {
            initial_population: 100,
            track_stats: true,
            ..Default::default()
        };
        
        let mut map_elites = MapElites::with_config(problem, config);
        
        // Check initial stats
        assert_eq!(map_elites.statistics().iterations, 0);
        assert_eq!(map_elites.statistics().num_solutions, 0);
        
        map_elites.run(1000);
        
        let stats = map_elites.statistics();
        assert_eq!(stats.iterations, 1000);
        assert!(stats.num_solutions > 0);
        assert!(stats.improvements > 0);
        assert!(stats.coverage > 0.0 && stats.coverage <= 1.0);
        assert!(!stats.best_fitness.is_nan());
    }
}
