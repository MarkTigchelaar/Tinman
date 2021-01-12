// exists
pub mod NeuralNetworkParts;
pub mod DataAndConfig;
pub mod ClassifierParts;
// want to use certain parts
pub use NeuralNetworkParts::neural_network::NeuralNetwork;
pub use DataAndConfig::neural_net_config_parts::NeuralNetSettings;
pub use DataAndConfig::dataset::DataSet;
pub use DataAndConfig::test_config::{TestConfig, Test};
pub use ClassifierParts::nnet_trainer::NNetTrainer;
pub use ClassifierParts::breeder::Breeder;
pub use ClassifierParts::classifier::Classifier;
pub mod prelude;
/*
use serde_json;
use std::fs::File;
use std::io::prelude::*;
use std::env;
use rand::Rng;
use std::io::Write;
*/
