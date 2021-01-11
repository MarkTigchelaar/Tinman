use super::super::DataAndConfig::neural_net_config_parts::LayerSettings;
use super::node::Node;
use super::nnet_errors::{
    weights_input_len_mismatch, 
    weights_node_count_mismatch
};
use rand::Rng;

pub fn make_new_nodes(layer_settings : &LayerSettings, new_nodes : &mut Vec<Node>, input_size : usize) {
    let mut rng = rand::thread_rng();
    match &layer_settings.layer_weights {
        Some(weight_vectors) => {
            if weight_vectors.len() != layer_settings.output_units {
                panic!(weights_input_len_mismatch());
            }
            for i in 0 .. weight_vectors.len() {
                if weight_vectors[i].len() != input_size {
                    panic!(weights_node_count_mismatch());
                }
                let mut weights : Vec<f64> = Vec::new();
                for j in 0 .. weight_vectors[i].len() {
                    weights.push(weight_vectors[i][j]);
                }
                new_nodes.push(Node::new(weights));
            }
        } None => {
            for _ in 0 .. layer_settings.output_units {
                let low : f64 = layer_settings.weight_range[0];
                let high : f64 = layer_settings.weight_range[1];
                let weights : Vec<f64> = vec![rng.gen_range(low, high); input_size];
                new_nodes.push(Node::new(weights));
            }
        }
    }
}