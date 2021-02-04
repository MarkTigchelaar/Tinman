use super::super::data_and_config::neural_net_config_parts::{NeuralNetSettings, LayerSettings};
use super::classifier_errors::{
    ga_missing_weights_error, 
    sa_missing_weights_error
};
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
    alter_learning_rate : bool,
    alter_bias : bool,
    alter_momentum : bool,
    alter_upper_weight_limit : bool,
    alter_lower_weight_limit : bool,
    swap_weights : bool
}

impl Breeder {
    pub fn new(temp_drop : f64, fave_drop : f64) -> Breeder {
        Breeder {
            max_annealing_temp : 1.0,
            max_current_annealing_temp : 1.0,
            current_annealing_temp : 1.0,
            min_annealing_temp : 0.01,
            temp_drop : temp_drop,
            max_p1_favourability : 0.95,
            max_current_p1_favourability : 0.95,
            current_p1_favourability : 0.95,
            min_p1_favourability : 0.55,
            favourability_drop : fave_drop,
            generation : 0,
            current_config_id : 0,
            alter_learning_rate : true,
            alter_bias : true,
            alter_momentum : true,
            alter_upper_weight_limit : true,
            alter_lower_weight_limit : true,
            swap_weights : false
        }
    }

    pub fn child_w_sa_hyper_parameters(&mut self, child : &mut NeuralNetSettings, parent : &mut NeuralNetSettings) {
        let mut rng = rand::thread_rng();
        let mut rand_float = rand::thread_rng();
        for j in 0 .. child.layers.len() {
            let mutation : f64 = rand_float.gen_range(0.0, self.current_annealing_temp);
            let rand : usize = rng.gen_range(0, 1);
            let mut sign : f64 = 1.0;
            if rand == 0 {
                sign = -1.0;
            }
            if self.alter_learning_rate == true {
                if parent.layers[j].learning_rate + sign * mutation > 0.0 {
                    child.layers[j].learning_rate = parent.layers[j].learning_rate + sign * mutation;
                } else {
                    child.layers[j].learning_rate = parent.layers[j].learning_rate - sign * mutation;
                }
            }
            if self.alter_bias == true {
                child.layers[j].bias = parent.layers[j].bias + sign * mutation;
            }
            if self.alter_momentum == true {
                if parent.layers[j].momentum + sign * mutation > 0.0 {
                    child.layers[j].momentum = parent.layers[j].momentum + sign * mutation;
                } else {
                    child.layers[j].momentum = parent.layers[j].momentum - sign * mutation;
                }
            }
            if self.alter_upper_weight_limit == true {
                if parent.layers[j].weight_range[1] + sign * mutation <= 0.0 {
                    sign *= -1.0;
                }
                child.layers[j].weight_range[1] = parent.layers[j].weight_range[1] + sign * mutation;
                //if child.layers[j].weight_range[1] > 1.0 {
                //    child.layers[j].weight_range[1] = 1.0;
                //}
            }
            if self.alter_lower_weight_limit == true {
                if parent.layers[j].weight_range[0] + sign * mutation >= 0.0 {
                    sign *= -1.0;
                }
                child.layers[j].weight_range[0] = parent.layers[j].weight_range[0] + sign * mutation;
                //if child.layers[j].weight_range[0] < -1.0 {
                //    child.layers[j].weight_range[0] = -1.0;
                //}
                if child.layers[j].weight_range[0] >= child.layers[j].weight_range[1] {
                    child.layers[j].weight_range[1] = child.layers[j].weight_range[0] + 0.1;
                }
            }
        }
        //child.config_id = self.current_config_id;
        self.current_config_id += 1;
    }

    pub fn child_w_ga_hyper_params(
        &mut self, 
        child : &mut NeuralNetSettings,
        parent1 : & NeuralNetSettings,
        parent2 : & NeuralNetSettings
    )
    {
        for j in 0 .. child.layers.len() {
            let mut p1_w_average : f64;
            let mut p2_w_average : f64;
            let favoured : f64 = self.current_p1_favourability;
            if self.alter_learning_rate == true {
                p1_w_average = parent1.layers[j].learning_rate * favoured;
                p2_w_average = parent2.layers[j].learning_rate * (1.0 - favoured);
                child.layers[j].learning_rate = p1_w_average + p2_w_average;
            }
            if self.alter_bias == true {
                p1_w_average = parent1.layers[j].bias * favoured;
                p2_w_average = parent2.layers[j].bias * (1.0 - favoured);
                child.layers[j].bias = p1_w_average + p2_w_average;
            }
            if self.alter_momentum == true {
                p1_w_average = parent1.layers[j].momentum * favoured;
                p2_w_average = parent2.layers[j].momentum * (1.0 - favoured);
                child.layers[j].momentum = p1_w_average + p2_w_average;
            }
            if self.alter_upper_weight_limit == true {
                p1_w_average = parent1.layers[j].weight_range[1] * favoured;
                p2_w_average = parent2.layers[j].weight_range[1] * (1.0 - favoured);
                child.layers[j].weight_range[1] = p1_w_average + p2_w_average;
            }
            if self.alter_lower_weight_limit == true {
                p1_w_average = parent1.layers[j].weight_range[0] * favoured;
                p2_w_average = parent2.layers[j].weight_range[0] * (1.0 - favoured);
                child.layers[j].weight_range[0] = p1_w_average + p2_w_average;
                if child.layers[j].weight_range[0] >= child.layers[j].weight_range[1] {
                    child.layers[j].weight_range[1] = child.layers[j].weight_range[0] + 0.1;
                }
            }
        }
        //child.config_id = self.current_config_id;
        self.current_config_id += 1;
    }

    pub fn child_w_sa_weights(
        &mut self,
        child : &mut NeuralNetSettings,
        parent : &mut NeuralNetSettings
    )
    {
        for j in 0 .. child.layers.len() {
            match &mut child.layers[j].layer_weights {
                Some(weight_vecs) => {
                    self.match_p1_weights_sa(
                        weight_vecs, 
                        &mut parent.layers[j]
                    );
                } None => {
                    panic!(sa_missing_weights_error());
                }
            }
        }
        //child.config_id = self.current_config_id;
        self.current_config_id += 1;
    }

    fn match_p1_weights_sa(
        &mut self,
        weight_vecs : &mut Vec<Vec<f64>>, 
        parent1_layer : &mut LayerSettings
    )
    {
        let mut rng = rand::thread_rng();
        let mut rand_float = rand::thread_rng();
        match &mut parent1_layer.layer_weights {
            Some(parent_weight_vecs) => {
                for k in 0 .. weight_vecs.len() {
                    for l in 0 .. weight_vecs[k].len() {
                        let current_temp : f64 = self.current_annealing_temp;
                        let mutation : f64 = rand_float.gen_range(0.0, current_temp);
                        let rand : usize = rng.gen_range(0, 1);
                        let mut sign : f64 = 1.0;
                        if rand == 0 {
                            sign = -1.0;
                        }
                        weight_vecs[k][l] = parent_weight_vecs[k][l] + sign * mutation;
                    }
                }
            } None => {
                panic!(sa_missing_weights_error());
            }
        }
    }

    pub fn child_w_ga_weights_swap(
        &mut self,
        child : &mut NeuralNetSettings,
        parent1 : &NeuralNetSettings, 
        parent2 : &NeuralNetSettings
    )
    {
        self.swap_weights = true;
        self.child_w_ga_weights(child, parent1, parent2);
        self.swap_weights = false;
    }

    pub fn child_w_ga_weights(
        &mut self,
        child : &mut NeuralNetSettings,
        parent1 : &NeuralNetSettings, 
        parent2 : &NeuralNetSettings
    )
    {
        for j in 0 .. child.layers.len() {
            match &mut child.layers[j].layer_weights {
                Some(weight_vecs) => {
                    self.match_p1_weights_ga(
                        weight_vecs, 
                        &parent1.layers[j], 
                        &parent2.layers[j]
                    );
                } None => {
                    panic!(ga_missing_weights_error());
                }
            }
        }
        //child.config_id = self.current_config_id;
        self.current_config_id += 1;
    }

    fn match_p1_weights_ga(
        &mut self, 
        weight_vecs : &mut Vec<Vec<f64>>, 
        parent1_layer : &LayerSettings, 
        parent2_layer : &LayerSettings
    )
    {
        match &parent1_layer.layer_weights {
            Some(p1_weight_vecs) => {
                self.match_p2_weights(
                    weight_vecs, 
                    p1_weight_vecs, 
                    parent2_layer
                );
            } None => {
                panic!(ga_missing_weights_error());
            }
        }
    }

    fn match_p2_weights(
        &mut self,
        weight_vecs : &mut Vec<Vec<f64>>,
        parent1_weights: &Vec<Vec<f64>>,
        parent2_layer : &LayerSettings
    )
    {
        let mut rand_float = rand::thread_rng();
        match &parent2_layer.layer_weights {
            Some(p2_weight_vecs) => {
                for k in 0 .. weight_vecs.len() {
                    let favourable : f64 = self.current_p1_favourability;
                    if self.swap_weights{
                        let p2_preference : f64 = rand_float.gen_range(0.0, 1.0);
                        for l in 0 .. weight_vecs[k].len() {
                            if p2_preference < favourable {
                                weight_vecs[k][l] = parent1_weights[k][l];
                            } else {
                                weight_vecs[k][l] = p2_weight_vecs[k][l];
                            }
                        }
                    } else {
                        for l in 0 .. weight_vecs[k].len() {
                            let p1_weight : f64 = parent1_weights[k][l] * favourable;
                            let p2_weight : f64 = p2_weight_vecs[k][l] * (1.0 - favourable);
                            weight_vecs[k][l] = p1_weight + p2_weight;
                        }
                    }
                }
            } None => {
                panic!(ga_missing_weights_error());
            }
        }
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

    pub fn min_ga_mutations_reached(&mut self) -> bool {
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

    pub fn min_sa_mutations_reached(&mut self) -> bool {
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

