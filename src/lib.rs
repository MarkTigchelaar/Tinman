pub mod neural_network_parts;
pub mod data_and_config;
pub mod classifier_parts;

pub use neural_network_parts::neural_network::NeuralNetwork;
pub use data_and_config::neural_net_config_parts::NeuralNetSettings;
pub use data_and_config::dataset::DataSet;
pub use data_and_config::test_config::{TestConfig, Test};
pub use classifier_parts::nnet_trainer::NNetTrainer;
pub use classifier_parts::breeder::Breeder;
pub use classifier_parts::classifier::Classifier;
pub mod prelude;