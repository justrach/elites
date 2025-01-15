
# Map-Elites: A Quality Diversity Algorithm Implementation

This repository provides a generic and efficient implementation of the **Map-Elites** algorithm, 
which is widely used to discover diverse, high-performing solutions across a feature space.

## Features
- **Modular Design**: Easily adaptable to various problem domains.
- **Customizable Configurations**: Configure bins, boundaries, and mutation strategies.
- **Statistics Tracking**: Monitor progress, coverage, and improvements over iterations.

## Getting Started

### Prerequisites
- **Rust**: Ensure you have [Rust](https://www.rust-lang.org/) installed.

### Installation
Add this crate to your `Cargo.toml` dependencies:

```toml
[dependencies]
elites = "0.1.0"
```

### Example Usage

Below is an example of how to use this crate for solving a simple optimization problem:

```rust
use elites::{MapElites, MapElitesProblem};

// Define your problem
struct MyProblem;

impl MapElitesProblem for MyProblem {
    type Genome = Vec<f64>;
    
    fn random_genome(&self) -> Self::Genome {
        vec![0.0, 0.0] // Simplified for example
    }
    
    fn evaluate(&self, genome: &Self::Genome) -> (f64, Vec<f64>) {
        let fitness = -genome.iter().map(|x| x.powi(2)).sum::<f64>();
        let features = vec![genome[0], genome[1]];
        (fitness, features)
    }
    
    fn mutate(&self, genome: &Self::Genome) -> Self::Genome {
        genome.clone() // Simplified for example
    }
    
    fn feature_dimensions(&self) -> usize { 2 }
    fn bins_per_dimension(&self) -> usize { 10 }
}

// Use the algorithm
let problem = MyProblem;
let mut map_elites = MapElites::new(problem);
map_elites.run(1000);
```

## Configuration

You can customize the algorithm using the `MapElitesConfig` struct:

```rust
let config = MapElitesConfig {
    initial_population: 200,
    track_stats: true,
    bin_boundaries: Some(vec![
        vec![0.0, 0.5, 1.0], // Custom boundaries for the first dimension
        vec![0.0, 0.5, 1.0], // Custom boundaries for the second dimension
    ]),
    ..Default::default()
};
```

## References

1. Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909.
## Testing

Run the tests to validate the implementation:

```bash
cargo test
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
