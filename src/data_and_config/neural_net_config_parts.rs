use serde_derive::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerSettings {
    pub activation_function : String,
    pub weight_range : [f64; 2],
    pub layer_weights : Option<Vec<Vec<f64>>>,
    pub output_units : usize,
    pub bias : f64,
    pub learning_rate : f64,
    pub momentum : f64
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeuralNetSettings {
    pub query_id : usize,
    pub config_id : usize,
    pub accuracy : f64,
    pub input_size : usize,
    pub layers : Vec<LayerSettings>
}