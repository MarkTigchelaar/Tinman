use super::super::data_and_config::neural_net_config_parts::NeuralNetSettings;
use super::super::data_and_config::dataset::DataSet;
use super::super::classifier_parts::breeder::Breeder;
use super::super::classifier_parts::nnet_trainer::NNetTrainer;

extern crate rayon;

use rayon::prelude::*;
use std::str;

pub struct Classifier {
    breeder : Breeder,
    test_train_cutoff_idx : usize,
    training_rounds_per_train_epoch : usize,
    max_train_epochs_per_config_epoch : usize,
    max_config_epochs : usize,
    final_number_of_nnets : usize,
    winners_per_round : usize,
    ancestor : NeuralNetSettings
}

impl Classifier {
    pub fn new(ancestor : NeuralNetSettings) -> Classifier {
        let temp_drop : f64 = 0.05; 
        let fave_drop : f64 = 0.05;
        Classifier {
            breeder : Breeder::new(temp_drop, fave_drop),
            test_train_cutoff_idx : 75,
            training_rounds_per_train_epoch : 1,
            max_train_epochs_per_config_epoch : 1,
            max_config_epochs : 1,
            final_number_of_nnets : 1,
            winners_per_round : 2,
            ancestor : ancestor
        }
    }

    pub fn set_train_cutoff(&mut self, train_cutoff: usize) {
        self.test_train_cutoff_idx = train_cutoff;
    }

    pub fn set_train_rounds(&mut self, train_rounds: usize) {
        self.training_rounds_per_train_epoch = train_rounds;
    }

    pub fn fit(&mut self, data : &DataSet) {
        for _ in 0 .. self.max_config_epochs {
            for _ in 0 .. self.max_train_epochs_per_config_epoch {
                let tuned_settings : Vec<NeuralNetSettings> = self.train_for_hyper_parameters(data);
                self.breeder.total_reset();
                for i in 0 .. tuned_settings.len() {
                    let serialized = serde_json::to_string(&tuned_settings[i]).unwrap();
                    println!("{}",str::from_utf8(&serialized.as_bytes()).unwrap());
                }
                
            }
        }
    }

    pub fn train_for_hyper_parameters(&mut self, data : &DataSet) -> Vec<NeuralNetSettings> {
        let mut winners : Vec<NeuralNetSettings> = Vec::new();
        let mut trainee_configs : Vec<NeuralNetSettings> = Vec::new();
        winners.push(self.ancestor.clone());
        while !
        (
            self.breeder.min_ga_mutations_reached() || 
            self.breeder.min_sa_mutations_reached()
        ) 
        {
            self.train_hyper_parameters_w_sa(
                &mut trainee_configs, 
                &mut winners, 
                data
            );
            self.train_hyper_parameters_w_ga(
                &mut trainee_configs, 
                &mut winners, 
                data
            );
        }
        winners
    }

    pub fn train_batch(&mut self, settings : &Vec<NeuralNetSettings>, data : &DataSet) -> Vec<NNetTrainer> {
        let tc : usize = self.test_train_cutoff_idx;
        let tr : usize = self.training_rounds_per_train_epoch;
        let trainers : Vec<NNetTrainer> = settings
            .par_iter()
            .map( |x| train_single(x, data, tc, tr) )
            .collect();
        println!("Results for current batch:");
        for i in 0 .. trainers.len() {
            println!("Correct: {}, total: {}", trainers[i].get_test_result(), data.data.len() - trainers[i].get_train_test_boundary());
        }
        trainers
    }

    fn get_winners(
        &mut self, 
        trainers : &mut Vec<NNetTrainer>, 
        trainee_configs : &mut Vec<NeuralNetSettings>,
        winners : &mut Vec<NeuralNetSettings>
    ) {
        winners.clear();
        let winner_ids : Vec<usize> = self.get_winner_ids(trainers);
        for i in 0 .. trainee_configs.len() {
            for j in 0 .. winner_ids.len() {
                if trainee_configs[i].config_id == winner_ids[j] {
                    winners.push(trainee_configs[i].clone());
                    break;
                }
            }
        }
    }

    fn get_winner_ids(&mut self, trainers : &mut Vec<NNetTrainer>) -> Vec<usize> {
        let mut winner_ids : Vec<usize> = Vec::new();
        let mut winner_idx : Vec<usize> = Vec::new();
        for i in 0 .. trainers.len() {
            if winner_idx.len() < self.winners_per_round {
                winner_idx.push(i);
            } else {
                let mut min : usize = 0;
                for j in 0 .. winner_idx.len() {
                    if trainers[winner_idx[j]].get_test_result() < trainers[winner_idx[min]].get_test_result() {
                        min = j;
                    }
                }
                if trainers[i].get_test_result() > trainers[winner_idx[min]].get_test_result() {
                    winner_idx[min] = i;
                }
            }
        }

        for i in 0 .. winner_idx.len() {
            winner_ids.push(trainers[ winner_idx[i] ].get_trainee_id());
        }
        winner_ids
    }

    fn train_hyper_parameters_w_sa(
        &mut self, 
        trainee_configs : &mut Vec<NeuralNetSettings>, 
        winners : &mut Vec<NeuralNetSettings>,
        data : &DataSet
    ) {
        trainee_configs.clear();
        while ! self.breeder.min_temp_reached() {
            for i in 0 .. winners.len() {
                trainee_configs.push(
                    self.breeder.child_w_sa_hyper_parameters(&mut winners[i])
                );
            }
            self.breeder.inc_generation_count();
            self.breeder.drop_temp();
        }
        self.breeder.reset_temp_to_adjusted_max();
        let mut trainers : Vec<NNetTrainer> = self.train_batch(&trainee_configs, data);
        self.get_winners(&mut trainers, trainee_configs, winners);
    }

    fn train_hyper_parameters_w_ga(
        &mut self, 
        trainee_configs : &mut Vec<NeuralNetSettings>, 
        winners : &mut Vec<NeuralNetSettings>,
        data : &DataSet
    ) {
        trainee_configs.clear();
        while ! self.breeder.min_current_p1_fav_reached() {
            for i in 0 .. winners.len() {
                for j in 0 .. winners.len() {
                    if i == j {
                        continue;
                    }
                    let mut child : NeuralNetSettings = winners[i].clone();
                    self.breeder.child_w_ga_hyper_params(
                        &mut child,
                        &mut winners[j]
                    );
                    trainee_configs.push(child);
                }
            }
            self.breeder.inc_generation_count();
            self.breeder.drop_p1_favourability();
        }
        self.breeder.reset_p1_fav_to_adjusted_max();
        let mut trainers : Vec<NNetTrainer> = self.train_batch(&trainee_configs, data);
        self.get_winners(&mut trainers, trainee_configs, winners);
    }
}

fn train_single(
    settings : &NeuralNetSettings, 
    data : &DataSet, 
    train_cutoff : usize, 
    train_rounds : usize
) -> NNetTrainer 
{
    let mut trainer : NNetTrainer = NNetTrainer::new(
        &settings, 
        train_cutoff, 
        train_rounds
    );
    trainer.train(&data);
    trainer.test(&data);
    trainer
}