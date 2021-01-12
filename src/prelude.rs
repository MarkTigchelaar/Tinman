//pub mod NeuralNetworkParts;
//pub mod DataAndConfig;
//pub mod ClassifierParts;
// want to use certain parts
pub use crate::NeuralNetworkParts::neural_network::NeuralNetwork;
pub use crate::DataAndConfig::neural_net_config_parts::NeuralNetSettings;
pub use crate::DataAndConfig::dataset::DataSet;
pub use crate::DataAndConfig::test_config::{TestConfig, Test};
pub use crate::ClassifierParts::nnet_trainer::NNetTrainer;
pub use crate::ClassifierParts::breeder::Breeder;
pub use crate::ClassifierParts::classifier::Classifier;