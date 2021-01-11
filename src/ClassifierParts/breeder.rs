use super::super::DataAndConfig::neural_net_config_parts::NeuralNetSettings;
use rand::Rng;

pub struct Breeder {
    max_annealing_temp : f64,
    max_current_annealing_temp : f64,
    current_annealing_temp : f64,
    min_annealing_temp : f64,
    temp_drop : f64,
    max_p1_favourability : f64,
    max_current_p1_favourability : f64,
    current_p1_favourability : f64,
    min_p1_favourability : f64,
    favourability_drop : f64,
    generation : usize,
    current_config_id : usize,
    hyper_parameters : [String; 5],
}

impl Breeder {
    pub fn new(temp_drop : f64, fave_drop : f64) -> Breeder {
        Breeder {
            max_annealing_temp : 0.99,
            max_current_annealing_temp : 0.99,
            current_annealing_temp : 0.99,
            min_annealing_temp : 0.01,
            temp_drop : temp_drop,
            max_p1_favourability : 0.95,
            max_current_p1_favourability : 0.95,
            current_p1_favourability : 0.95,
            min_p1_favourability : 0.55,
            favourability_drop : fave_drop,
            generation : 0,
            current_config_id : 0,
            hyper_parameters : [
                "learning_rate".to_string(),
                "bias".to_string(),
                "momentum".to_string(),
                "upper_weight_limit".to_string(),
                "lower_weight_limit".to_string()
            ]
        }
    }

    pub fn child_w_sa_hyper_parameters(&mut self, parent : &mut NeuralNetSettings) -> NeuralNetSettings {
        let mut child : NeuralNetSettings = parent.clone();
        let mut rng = rand::thread_rng();
        let mut rand_float = rand::thread_rng();
        for j in 0 .. child.layers.len() {
            for i in 0 .. self.hyper_parameters.len() {
                let mutation : f64 = rand_float.gen_range(0.0, self.current_annealing_temp);
                let rand : usize = rng.gen_range(0, 1);
                let mut sign : f64 = 1.0;
                if rand == 0 {
                    sign = -1.0;
                }
                if self.hyper_parameters[i] == "learning_rate".to_string() {
                    if child.layers[j].learning_rate + sign * mutation > 0.0 {
                        child.layers[j].learning_rate += sign * mutation;
                    } else {
                        child.layers[j].learning_rate -= sign * mutation;
                    }
                } else if self.hyper_parameters[i] == "bias".to_string() {
                    child.layers[j].bias += sign * mutation;
                } else if self.hyper_parameters[i] == "momentum".to_string() {
                    if child.layers[j].momentum + sign * mutation > 0.0 {
                        child.layers[j].momentum += sign * mutation;
                    } else {
                        child.layers[j].momentum -= sign * mutation;
                    }
                } else if self.hyper_parameters[i] == "upper_weight_limit".to_string() {
                    child.layers[j].weight_range[1] += sign * mutation;
                    if child.layers[j].weight_range[1] > 1.0 {
                        child.layers[j].weight_range[1] = 1.0;
                    }
                } else if self.hyper_parameters[i] == "lower_weight_limit".to_string() {
                    child.layers[j].weight_range[0] += sign * mutation;
                    if child.layers[j].weight_range[0] < -1.0 {
                        child.layers[j].weight_range[0] = -1.0;
                    }
                    if child.layers[j].weight_range[0] >= child.layers[j].weight_range[1] {
                        child.layers[j].weight_range[1] = child.layers[j].weight_range[0] + 0.1;
                    }
                }
            }
        }
        child.config_id = self.current_config_id;
        self.current_config_id += 1;
        child
    }

    pub fn child_w_ga_hyper_params(
        &mut self, 
        child : &mut NeuralNetSettings, 
        parent2 : &mut NeuralNetSettings
    )
    {
        for j in 0 .. child.layers.len() {
            for i in 0 .. self.hyper_parameters.len() {
                let child_w_average : f64;
                let p2_w_average : f64;
                let favoured : f64 = self.current_p1_favourability;
                if self.hyper_parameters[i] == "learning_rate".to_string() {
                    child_w_average = child.layers[j].learning_rate * favoured;
                    p2_w_average = parent2.layers[j].learning_rate * (1.0 - favoured);
                    child.layers[j].learning_rate = (child_w_average + p2_w_average);
                } else if self.hyper_parameters[i] == "bias".to_string() {
                    child_w_average = child.layers[j].bias * favoured;
                    p2_w_average = parent2.layers[j].bias * (1.0 - favoured);
                    child.layers[j].bias = (child_w_average + p2_w_average);
                } else if self.hyper_parameters[i] == "momentum".to_string() {
                    child_w_average = child.layers[j].momentum * favoured;
                    p2_w_average = parent2.layers[j].momentum * (1.0 - favoured);
                    child.layers[j].momentum = (child_w_average + p2_w_average);
                } else if self.hyper_parameters[i] == "upper_weight_limit".to_string() {
                    child_w_average = child.layers[j].weight_range[1] * favoured;
                    p2_w_average = parent2.layers[j].weight_range[1] * (1.0 - favoured);
                    child.layers[j].weight_range[1] = (child_w_average + p2_w_average);
                } else if self.hyper_parameters[i] == "lower_weight_limit".to_string() {
                    child_w_average = child.layers[j].weight_range[0] * favoured;
                    p2_w_average = parent2.layers[j].weight_range[0] * (1.0 - favoured);
                    child.layers[j].weight_range[0] = (child_w_average + p2_w_average);
                    if child.layers[j].weight_range[0] >= child.layers[j].weight_range[1] {
                        child.layers[j].weight_range[1] = child.layers[j].weight_range[0] + 0.1;
                    }
                }
            }
        }
        child.config_id = self.current_config_id;
        self.current_config_id += 1;
    }

    pub fn child_w_sa_weights(&mut self, parent : &mut NeuralNetSettings) -> NeuralNetSettings {
        let mut child : NeuralNetSettings = parent.clone();
        let mut rng = rand::thread_rng();
        let mut rand_float = rand::thread_rng();
        for j in 0 .. child.layers.len() {
            match &mut child.layers[j].layer_weights {
                Some(weight_vecs) => {
                    for k in 0 .. weight_vecs.len() {
                        for l in 0 .. weight_vecs[k].len() {
                            let current_temp : f64 = self.current_annealing_temp;
                            let mutation : f64 = rand_float.gen_range(0.0, current_temp);
                            let rand : usize = rng.gen_range(0, 1);
                            let mut sign : f64 = 1.0;
                            if rand == 0 {
                                sign = -1.0;
                            }
                            weight_vecs[k][l] += sign * mutation;
                        }
                    }
                } None => {
                    panic!(sa_missing_weights_error());
                }
            }
        }
        child.config_id = self.current_config_id;
        self.current_config_id += 1;
        child
    }

    pub fn child_w_ga_weights(
        &mut self, 
        parent1 : &mut NeuralNetSettings, 
        parent2 : &mut NeuralNetSettings
    ) -> NeuralNetSettings 
    {
        let mut child : NeuralNetSettings = parent1.clone();
        let mut rand_float = rand::thread_rng();
        for j in 0 .. child.layers.len() {
            match &mut child.layers[j].layer_weights {
                Some(weight_vecs) => {
                    match &mut parent2.layers[j].layer_weights {
                        Some(p2_weight_vecs) => {
                            for k in 0 .. weight_vecs.len() {
                                let favourable : f64 = self.current_p1_favourability;
                                let preference : f64 = rand_float.gen_range(0.0, 1.0);
                                if preference > favourable {
                                    for l in 0 .. weight_vecs[k].len() {
                                        weight_vecs[k][l] = p2_weight_vecs[k][l];
                                    }
                                }
                            }
                        } None => {
                            panic!(ga_missing_weights_error());
                        }
                    }
                } None => {
                    panic!(ga_missing_weights_error());
                }
            }
        }
        child.config_id = self.current_config_id;
        self.current_config_id += 1;
        child
    }

    pub fn drop_p1_favourability(&mut self) {
        self.current_p1_favourability -= self.favourability_drop;
        if self.current_p1_favourability < self.min_p1_favourability {
            self.current_p1_favourability = self.min_p1_favourability;
        } 
    }

    pub fn reset_p1_fav_to_adjusted_max(&mut self) {
        self.max_current_p1_favourability -= self.favourability_drop;
        if self.max_current_p1_favourability < self.min_p1_favourability {
            self.max_current_p1_favourability = self.min_p1_favourability;
        } 
        self.current_p1_favourability = self.max_current_p1_favourability;
    }

    pub fn min_current_p1_fav_reached(&mut self) -> bool {
        self.current_p1_favourability <= self.min_p1_favourability
    }

    pub fn min_GA_mutations_reached(&mut self) -> bool {
        self.max_current_p1_favourability <= self.min_p1_favourability
    }

    pub fn drop_temp(&mut self) {
        self.current_annealing_temp -= self.temp_drop;
        if self.current_annealing_temp < self.min_annealing_temp {
            self.current_annealing_temp = self.min_annealing_temp;
        }
    }

    pub fn reset_temp_to_adjusted_max(&mut self) {
        self.max_current_annealing_temp -= self.temp_drop;
        if self.max_current_annealing_temp < self.min_annealing_temp {
            self.max_current_annealing_temp = self.min_annealing_temp;
        }
        self.current_annealing_temp = self.max_current_annealing_temp;
    }

    pub fn min_temp_reached(&mut self) -> bool {
        self.current_annealing_temp <= self.min_annealing_temp
    }

    pub fn min_SA_mutations_reached(&mut self) -> bool {
        self.max_current_annealing_temp <= self.min_annealing_temp
    }

    pub fn total_reset(&mut self) {
        self.current_annealing_temp = self.max_annealing_temp;
        self.current_p1_favourability = self.max_p1_favourability;
        self.max_current_annealing_temp = self.max_annealing_temp;
        self.max_current_p1_favourability = self.max_p1_favourability;
    }

    pub fn inc_generation_count(&mut self) {
        self.generation += 1;
    }

    pub fn generation_count(&mut self) -> usize {
        self.generation
    }
}

fn ga_missing_weights_error() -> String {
    "Genetic algoritm for weights must be done on neural network layers containing weight vectors!".to_string()
}

fn sa_missing_weights_error() -> String {
    "Simulated Annealing for weights must be done on neural network layers containing weight vectors!".to_string()  
}