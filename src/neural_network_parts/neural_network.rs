use super::activator::Activator;
use super::node::Node;
use super::layer::Layer;
use super::make_new_nodes::make_new_nodes;
use super::nnet_errors::failed_prediction;
use super::super::data_and_config::neural_net_config_parts::{
    NeuralNetSettings,
    LayerSettings
};

pub struct NeuralNetwork {
    activator : Activator,
    query_id : usize,
    config_id : usize,
    input_size : usize,
    layers : Vec<Layer>,
    nodes : Vec<Node>
}

impl NeuralNetwork {
    pub fn new(nnet_settings : &NeuralNetSettings) -> NeuralNetwork {
        let mut activator = Activator::new();
        let mut input_size : usize = nnet_settings.input_size;
        let mut layers : Vec<Layer> = Vec::new();
        let mut nodes : Vec<Node> = Vec::new();
        for i in 0 .. nnet_settings.layers.len() {
            let act_fn_name : &String = &nnet_settings.layers[i].activation_function;
            let act_fn_code : usize = activator.get_fn_code_by_name(&act_fn_name);
            let start_index : usize = nodes.len();
            if i == 0 {
                layers.push(
                    Layer::new(
                        &nnet_settings.layers[i], 
                        0, 
                        act_fn_code, 
                        start_index
                    )
                );
            } else {
                layers.push(
                    Layer::new(
                        &nnet_settings.layers[i], 
                        input_size, 
                        act_fn_code, 
                        start_index
                    )
                );
            }
            make_new_nodes(
                &nnet_settings.layers[i], 
                &mut nodes, 
                input_size
            );
            input_size = nnet_settings.layers[i].output_units;
        }

        NeuralNetwork {
            activator: activator,
            query_id : nnet_settings.query_id,
            config_id : nnet_settings.config_id,
            input_size : input_size,
            layers : layers,
            nodes : nodes
        }
    }

    pub fn get_settings(&mut self) -> NeuralNetSettings {
        let mut layer_settings : Vec<LayerSettings> = Vec::new();
        for i in 0 .. self.layers.len() {
            continue;
        }
        let mut settings : NeuralNetSettings = NeuralNetSettings {
            query_id : self.query_id,
            config_id : self.config_id,
            input_size : self.input_size,
            accuracy : 0.0,
            layers : layer_settings
        };
        settings
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) {
        self.layers[0].input_layer_forward(
            inputs, 
            &mut self.nodes, 
            &mut self.activator
        );
        for i in 1 .. self.layers.len() {
            self.layers[i].hidden_layer_forward(
                &mut self.nodes, 
                &mut self.activator
            );
        }
    }

    pub fn set_error_delta(&mut self, correct_index : usize) {
        let last_layer_idx : usize = self.layers.len() - 1;
        self.layers[last_layer_idx].set_delta(
            correct_index, 
            &mut self.nodes
        );
    }

    pub fn backward(&mut self, inputs: &Vec<f64>) {
        let last_layer_idx : usize = self.layers.len();
        for i in (1 .. last_layer_idx).rev() {
            self.layers[i].hidden_layer_backward(&mut self.nodes);
        }
        self.layers[0].input_layer_backward(inputs, &mut self.nodes);
    }

    pub fn predict(&mut self, inputs: &Vec<f64>) -> usize {
        self.forward(inputs);
        let last_layer_idx : usize = self.layers.len() - 1;
        let prediction : i64 = self.layers[last_layer_idx].get_prediction(&mut self.nodes);
        if prediction < 0 {
            panic!(failed_prediction());
        }
        prediction as usize
    }

    pub fn get_id(&self) -> usize {
        self.config_id
    }

    pub fn update_state(&mut self, current_default : &NeuralNetSettings) {
        self.config_id = current_default.config_id;
        self.query_id = current_default.query_id;
        let layer_settings : &Vec<LayerSettings> = &current_default.layers;
        if layer_settings.len() != self.layers.len() {
            panic!("Updating settings does not match number of layers");
        }
        for i in 0 .. self.layers.len() {
            let layer_setting = &layer_settings[i];
            let fn_code = self.activator.get_fn_code_by_name(&layer_setting.activation_function);
            self.layers[i].update_state(layer_setting, fn_code, &mut self.nodes);
        }
    }
}