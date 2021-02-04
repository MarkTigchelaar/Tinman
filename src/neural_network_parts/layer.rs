use super::activator::Activator;
use super::node::Node;
use super::super::data_and_config::neural_net_config_parts::LayerSettings;

pub struct Layer {
    learning_rate : f64,
    activation_function_code : usize,
    momentum : f64,
    bias : f64,
    high : f64,
    low : f64,
    prev_layer_first_node_idx : usize,
    nodes_start_index : usize,
    nodes_stop_index : usize
}

impl Layer {
    pub fn new(settings : &LayerSettings, input_size : usize, fn_code : usize, start : usize) -> Layer {
        Layer {
            learning_rate : settings.learning_rate,
            activation_function_code : fn_code,
            momentum : settings.momentum,
            bias : settings.bias,
            high : settings.weight_range[1],
            low : settings.weight_range[0],
            prev_layer_first_node_idx : start - input_size,
            nodes_start_index : start,
            nodes_stop_index : start + settings.output_units
        }
    }

    pub fn input_layer_forward(&mut self, inputs : &Vec<f64>, nodes : &mut Vec<Node>, activator : &mut Activator) {
        activator.set_fn_code(self.activation_function_code);
        for i in self.nodes_start_index .. self.nodes_stop_index {
            nodes[i].input_layer_forward(inputs, activator, self.bias);
        }
    }

    pub fn hidden_layer_forward(&mut self, nodes : &mut Vec<Node>, activator : &mut Activator) {
        activator.set_fn_code(self.activation_function_code);
        for i in self.nodes_start_index .. self.nodes_stop_index {
            let mut result : f64 = 0.0;
            for j in self.prev_layer_first_node_idx .. self.nodes_start_index {
                result += nodes[i].get_weight_at(j - self.prev_layer_first_node_idx) * nodes[j].get_activated_output();
            }
            let activated : f64 = activator.activate(result);
            nodes[i].set_activated_output(activated);
            let activated_prime : f64 = activator.activate_prime(result);
            nodes[i].set_activated_prime_output(activated_prime);
        }
    }

    pub fn set_delta(&mut self, correct_index : usize, nodes : &mut Vec<Node>) {
        let mut yhat : f64;
        for i in self.nodes_start_index .. self.nodes_stop_index {
            let output_prime : f64 = nodes[i].get_activated_prime_output();
            let output : f64 = nodes[i].get_activated_output();
            if i - self.nodes_start_index == correct_index {
                yhat = 1.0;
            } else {
                yhat = 0.0;
            }
            nodes[i].set_delta((output - yhat) * output_prime);
        }
    }

    pub fn get_prediction(&mut self, nodes : &mut Vec<Node>) -> i64 {
        let mut max_idx : i64 = -1;
        let mut max_val = 0.0;
        for i in self.nodes_start_index .. self.nodes_stop_index {
            let current : f64 = nodes[i].get_activated_output();
            if current > max_val {
                max_val = current;
                max_idx = i as i64;
            }
        }
        if max_idx < self.nodes_start_index as i64 {
            max_idx = self.nodes_start_index as i64;
        } else {
            max_idx -= self.nodes_start_index as i64;
        }
        max_idx
    }

    pub fn hidden_layer_backward(&mut self, nodes : &mut Vec<Node>) {
        for i in self.prev_layer_first_node_idx .. self.nodes_start_index {
            let mut error_for_node = 0.0;
            for j in self.nodes_start_index .. self.nodes_stop_index {
                let adjusted_idx : usize = i - self.prev_layer_first_node_idx;
                let delta : f64 = nodes[j].get_delta();
                error_for_node += delta * nodes[j].get_weight_at(adjusted_idx);
                let delta_w = nodes[i].get_activated_output() * delta;
                let momentum_adjustment : f64 = self.momentum * nodes[j].get_prev_weight_at(adjusted_idx);
                let adjusted_delta_w : f64 = self.learning_rate * delta_w;
                nodes[j].set_weight_at(adjusted_idx, adjusted_delta_w + momentum_adjustment);
                nodes[j].set_prev_weight_at(adjusted_idx, adjusted_delta_w + momentum_adjustment);
            }
            let prev_output_prime : f64 = nodes[i].get_activated_prime_output();
            nodes[i].set_delta(error_for_node * prev_output_prime);
        }
    }

    pub fn input_layer_backward(&mut self, inputs : &Vec<f64>, nodes : &mut Vec<Node>) {
        for i in 0 .. inputs.len() {
            for j in self.nodes_start_index .. self.nodes_stop_index {
                let delta_w : f64 = nodes[j].get_delta() * inputs[i];
                let momentum_adjustment : f64 = self.momentum * nodes[j].get_prev_weight_at(i);
                let adjusted_delta_w : f64 = self.learning_rate * delta_w;
                nodes[j].set_weight_at(i, (self.learning_rate * delta_w) + momentum_adjustment);
                nodes[j].set_prev_weight_at(i, adjusted_delta_w + momentum_adjustment);
            }
        }
    }

    pub fn update_state(
        &mut self, 
        default_settings : &LayerSettings, 
        fn_code : usize,
        nodes : &mut Vec<Node>
    )
    {
        self.learning_rate = default_settings.learning_rate;
        self.activation_function_code = fn_code;
        self.momentum = default_settings.momentum;
        self.bias = default_settings.bias;
        self.high = default_settings.weight_range[1];
        self.low = default_settings.weight_range[0];
        match &default_settings.layer_weights {
            Some(weights) => {
                for i in self.nodes_start_index .. self.nodes_stop_index {
                    nodes[i].update_state(&weights[i- self.nodes_start_index]);
                }
            } None => {
                panic!("attempting to update state, found no weight vectors");
            }
        }
    }
}