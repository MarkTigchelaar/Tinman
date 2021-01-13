use super::activator::Activator;

pub struct Node {
    weights : Vec<f64>,
    prev_weights : Vec<f64>,
    activated_output : f64,
    activated_output_prime : f64,
    delta : f64
}

impl Node {
    pub fn new(weight_vector : Vec<f64>) -> Node {
        let vec_len : usize = weight_vector.len();
        Node {
            weights : weight_vector,
            prev_weights : vec![0.0; vec_len],
            activated_output : 0.0,
            activated_output_prime : 0.0,
            delta : 0.0
        }
    }

    pub fn input_layer_forward(&mut self, input : &Vec<f64>, activator : &mut Activator, bias : f64) {
        let mut result : f64 = 0.0;
        for index in 0..input.len() {
            result += self.weights[index] * input[index];
        }
        self.activated_output = activator.activate(result + bias);
        self.activated_output_prime = activator.activate_prime(result);
    }

    pub fn get_activated_output(&mut self) -> f64 {
        self.activated_output
    }

    pub fn set_activated_output(&mut self, new_value : f64) {
        self.activated_output = new_value;
    }

    pub fn get_activated_prime_output(&mut self) -> f64 {
        self.activated_output_prime
    }

    pub fn set_activated_prime_output(&mut self, new_value : f64) {
        self.activated_output_prime = new_value;
    }

    pub fn set_delta(&mut self, delta: f64) {
        self.delta = delta;
    }

    pub fn get_delta(&mut self) -> f64 {
        self.delta
    }

    pub fn get_weight_at(&mut self, index: usize) -> f64 {
        self.weights[index]
    }

    pub fn set_weight_at(&mut self, index: usize, amount : f64) {
        self.weights[index] -= amount;
    }

    pub fn get_prev_weight_at(&mut self, index: usize) -> f64 {
        self.prev_weights[index]
    }

    pub fn set_prev_weight_at(&mut self, index: usize, amount : f64) {
        self.prev_weights[index] = amount;
    }
}