use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use numpy::PyArray1;
use rand::Rng;
use std::collections::HashMap;
use rayon::ThreadPoolBuilder;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[pyclass]
struct PyMapElites {
    solutions: HashMap<Vec<usize>, Individual>,
    bins_per_dimension: usize,
    feature_dimensions: usize,
    config: Config,
    stats: Stats,
}

#[derive(Clone)]
struct Individual {
    genome: Vec<f64>,
    fitness: f64,
    features: Vec<f64>,
}

#[derive(Clone)]
struct Config {
    initial_population: usize,
    track_stats: bool,
}

#[derive(Clone, Default)]
struct Stats {
    iterations: usize,
    num_solutions: usize,
    best_fitness: f64,
    coverage: f64,
    improvements: usize,
}

#[pymethods]
impl PyMapElites {
    #[new]
    #[pyo3(signature = (feature_dimensions, bins_per_dimension, initial_population=100))]
    fn new(feature_dimensions: usize, bins_per_dimension: usize, initial_population: usize) -> Self {
        Self {
            solutions: HashMap::new(),
            bins_per_dimension,
            feature_dimensions,
            config: Config {
                initial_population,
                track_stats: true,
            },
            stats: Stats::default(),
        }
    }

    #[pyo3(signature = (iterations, evaluate_fn, mutate_fn, random_fn))]
    fn run(
        &mut self,
        py: Python,
        iterations: usize,
        evaluate_fn: PyObject,
        mutate_fn: PyObject,
        random_fn: PyObject,
    ) -> PyResult<()> {
        // Initialize population
        for _ in 0..self.config.initial_population {
            Python::with_gil(|py| {
                let genome: Vec<f64> = random_fn.call0(py)?.extract(py)?;
                let (fitness, features) = self.evaluate_individual(py, &genome, &evaluate_fn)?;
                self.add_to_map(Individual { genome, fitness, features });
                Ok::<_, PyErr>(())
            })?;
        }

        // Main loop
        for _ in 0..iterations {
            if self.solutions.is_empty() {
                continue;
            }

            let solutions = self.solutions.values().cloned().collect::<Vec<_>>();
            let parent = &solutions[rand::thread_rng().gen_range(0..solutions.len())];

            Python::with_gil(|py| {
                let offspring: Vec<f64> = mutate_fn.call1(py, (parent.genome.clone(),))?.extract(py)?;
                let (fitness, features) = self.evaluate_individual(py, &offspring, &evaluate_fn)?;
                self.add_to_map(Individual {
                    genome: offspring,
                    fitness,
                    features,
                });
                Ok::<_, PyErr>(())
            })?;

            // Allow progress bar updates
            if iterations % 100 == 0 {
                Python::with_gil(|py| {
                    py.allow_threads(|| {
                        std::thread::sleep(std::time::Duration::from_micros(1));
                    });
                });
            }
        }

        Ok(())
    }

    fn get_solutions<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let solutions_dict = PyDict::new(py);
        
        for (bins, individual) in &self.solutions {
            let key = PyTuple::new(
                py,
                bins.iter().map(|&x| x as i64).collect::<Vec<_>>().as_slice()
            );
            
            let value = (
                individual.genome.clone(),
                individual.fitness,
                individual.features.clone(),
            );
            solutions_dict.set_item(key, value)?;
        }
        
        Ok(solutions_dict)
    }

    fn get_statistics(&self) -> (usize, usize, f64, f64, usize) {
        (
            self.stats.iterations,
            self.stats.num_solutions,
            self.stats.best_fitness,
            self.stats.coverage,
            self.stats.improvements,
        )
    }

    fn coverage(&self) -> f64 {
        let total_bins = self.bins_per_dimension.pow(self.feature_dimensions as u32);
        self.solutions.len() as f64 / total_bins as f64
    }
}

impl PyMapElites {
    fn evaluate_individual(
        &self,
        py: Python,
        genome: &[f64],
        evaluate_fn: &PyObject,
    ) -> PyResult<(f64, Vec<f64>)> {
        let args = (genome.to_vec(),);
        let result = evaluate_fn.call1(py, args)?;
        let (fitness, features): (f64, Vec<f64>) = result.extract(py)?;
        Ok((fitness, features))
    }

    fn add_to_map(&mut self, individual: Individual) -> bool {
        if individual.fitness.is_nan() || individual.fitness.is_infinite() {
            return false;
        }

        if !individual.features.iter().all(|&f| f >= 0.0 && f <= 1.0) {
            return false;
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

    fn features_to_bins(&self, features: &[f64]) -> Vec<usize> {
        features
            .iter()
            .map(|&f| {
                let bin = (f * self.bins_per_dimension as f64).floor() as usize;
                bin.min(self.bins_per_dimension - 1)
            })
            .collect()
    }
}

#[pymodule]
fn elites(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMapElites>()?;
    Ok(())
} 