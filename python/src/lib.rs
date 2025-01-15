use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use numpy::PyArray1;
use elites::{MapElites, MapElitesProblem, MapElitesConfig, Individual};

#[pyclass]
struct PyMapElites {
    solutions: std::collections::HashMap<Vec<usize>, Individual<Vec<f64>>>,
    bins_per_dimension: usize,
    feature_dimensions: usize,
    config: MapElitesConfig,
}

struct PythonProblem {
    evaluate_fn: PyObject,
    mutate_fn: PyObject,
    random_fn: PyObject,
    feature_dims: usize,
    bins_per_dim: usize,
}

impl MapElitesProblem for PythonProblem {
    type Genome = Vec<f64>;

    fn random_genome(&self) -> Self::Genome {
        Python::with_gil(|py| {
            let result = self.random_fn.call0(py).unwrap();
            let array: &PyArray1<f64> = result.extract(py).unwrap();
            array.to_vec().unwrap()
        })
    }

    fn evaluate(&self, genome: &Self::Genome) -> (f64, Vec<f64>) {
        Python::with_gil(|py| {
            let genome_array = PyArray1::from_slice(py, genome);
            let args = PyTuple::new(py, &[genome_array.into_py(py)]);
            let result = self.evaluate_fn.call1(py, args).unwrap();
            let (fitness, features): (f64, Vec<f64>) = result.extract(py).unwrap();
            (fitness, features)
        })
    }

    fn mutate(&self, genome: &Self::Genome) -> Self::Genome {
        Python::with_gil(|py| {
            let genome_array = PyArray1::from_slice(py, genome);
            let args = PyTuple::new(py, &[genome_array.into_py(py)]);
            let result = self.mutate_fn.call1(py, args).unwrap();
            let array: &PyArray1<f64> = result.extract(py).unwrap();
            array.to_vec().unwrap()
        })
    }

    fn feature_dimensions(&self) -> usize {
        self.feature_dims
    }

    fn bins_per_dimension(&self) -> usize {
        self.bins_per_dim
    }
}

#[pymethods]
impl PyMapElites {
    #[new]
    #[pyo3(signature = (feature_dimensions, bins_per_dimension, initial_population=100))]
    fn new(feature_dimensions: usize, bins_per_dimension: usize, initial_population: usize) -> Self {
        Self {
            solutions: std::collections::HashMap::new(),
            bins_per_dimension,
            feature_dimensions,
            config: MapElitesConfig {
                initial_population,
                track_stats: true,
                bin_boundaries: None,
            },
        }
    }

    fn run(
        &mut self,
        iterations: usize,
        evaluate_fn: PyObject,
        mutate_fn: PyObject,
        random_fn: PyObject,
    ) -> PyResult<()> {
        let problem = PythonProblem {
            evaluate_fn,
            mutate_fn,
            random_fn,
            feature_dims: self.feature_dimensions,
            bins_per_dim: self.bins_per_dimension,
        };

        let mut map_elites = MapElites::with_config(problem, self.config.clone());
        map_elites.run(iterations);

        self.solutions = map_elites.solutions().clone();
        Ok(())
    }

    fn get_solutions<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let solutions_dict = PyDict::new(py);
        
        for (bins, individual) in &self.solutions {
            let key = bins.to_vec();
            let genome = PyArray1::from_slice(py, &individual.genome);
            let features = PyArray1::from_slice(py, &individual.features);
            let value = (
                genome,
                individual.fitness,
                features,
            );
            solutions_dict.set_item(key, value)?;
        }
        
        Ok(solutions_dict)
    }

    fn coverage(&self) -> f64 {
        let total_bins = self.bins_per_dimension.pow(self.feature_dimensions as u32);
        self.solutions.len() as f64 / total_bins as f64
    }
}

#[pymodule]
fn elites(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMapElites>()?;
    Ok(())
} 