use serde_derive::{Serialize, Deserialize};
use super::super::data_and_config::neural_net_config_parts::NeuralNetSettings;

#[derive(Serialize, Deserialize,Clone)]
pub struct OptimizerParameters {
    pub temperature_drops : usize,
    pub heritability_bias_drops : usize,
    pub current_candidate_configuration : Box<NeuralNetSettings>,
    pub test_train_cutoff_idx : usize,
    pub train_rounds_per_epoch : usize,
    pub max_train_epochs : usize,
    pub max_config_changing_epochs : usize,
    pub final_number_of_nnet_settings : usize,
    pub winners_per_round : usize,
    pub tuned_settings : Vec<Box<NeuralNetSettings>>,
    pub min_acceptable_accuracy : f64,
    pub cpus_to_use : usize
}