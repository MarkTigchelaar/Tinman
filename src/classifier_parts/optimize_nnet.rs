use super::super::data_and_config::neural_net_config_parts::{NeuralNetSettings, LayerSettings};
use super::super::data_and_config::dataset::DataSet;
use super::super::data_and_config::optimizer_parameters::OptimizerParameters;
use super::super::classifier_parts::breeder::Breeder;
use super::super::classifier_parts::nnet_trainer::NNetTrainer;

use std::thread::spawn;
use std::sync::Arc;
use rand::Rng;

extern crate num_cpus;
use num_cpus::get;

extern crate crossbeam;
use crossbeam::scope;

use std::time::Instant;

struct InternalSettings {
    optimizer_params : OptimizerParameters,
    most_accurate_settings : Vec<Box<NeuralNetSettings>>,
    candidate_settings : Vec<Box<NeuralNetSettings>>,
    recycled_settings : Vec<Box<NeuralNetSettings>>,
    candidate_trainers : Vec<Box<NNetTrainer>>,
    recycled_trainers : Vec<Box<NNetTrainer>>,
    breeder : Breeder,
    max_accuracy : f64,
    reset_weights : bool,
    id_sequence : usize
}

// Although this could be changed to a struct,
// the user should see this as a mini program to call, not a object type.
// Insert a request form, get a product as close to requested as possible.
pub fn optimize_nnet_settings(order_form : &mut OptimizerParameters, data : &DataSet) {
    let mut settings : InternalSettings = make_settings(order_form);
    'outer: for i in 0 .. settings.optimizer_params.max_config_changing_epochs {
        settings.reset_weights = true;
        let mut prev_accuracy : f64 = 0.0;
        let mut tries : u8 = 0;
        for j in 0 .. settings.optimizer_params.max_train_epochs {
            println!("config epoch: {}, training epoch: {}", i + 1, j + 1);
            alter_winners_per_round_if_final_round(i,j, &mut settings);

            prev_accuracy = tune_hyper_parameters(&mut settings, data, prev_accuracy);
            prev_accuracy = tune_weights(&mut settings, data, prev_accuracy);

            if acceptable_accuracy_reached(&settings) {
                break 'outer;
            } else if min_accuracy_inc_not_reached(prev_accuracy, &mut settings) {
                alter_activation_fns_and_optimizers(&mut settings);
                tries += 1;
            }
            if tries > 3 {
                break;
            }
        }
        if i < settings.optimizer_params.max_config_changing_epochs - 1 {
            increase_nodes(&mut settings);
        }
    }
    trim_nodes(&mut settings, data);
    collect_final_settings(&mut settings, order_form);
}

fn tune_hyper_parameters(settings : &mut InternalSettings, data : &DataSet, mut prev_accuracy : f64 ) -> f64 {
    settings.breeder.total_reset();
    println!("tune_hyper_parameters");
    while minimum_not_reached(settings) {
        prev_accuracy = tune_hyper_parameters_w_sa(settings, data, prev_accuracy);
        if acceptable_accuracy_reached(&settings) {
            return prev_accuracy;
        }
        prev_accuracy = tune_hyper_parameters_w_ga(settings, data, prev_accuracy);
        if min_accuracy_inc_not_reached(prev_accuracy, settings) {
            println!("Accuracy not improving enough from GA, leaving loop");
            break;
        } else if acceptable_accuracy_reached(&settings) {
            return prev_accuracy;
        }
    }
    prev_accuracy
}

fn print_off_info(settings : &mut InternalSettings) {
    println!("\n\n\n\nWinner vector");
    for i in 0 .. settings.most_accurate_settings.len() {
        print!("id: {} ", settings.most_accurate_settings[i].config_id);
    }
    println!("\nCandidate vector");
    for i in 0 .. settings.candidate_settings.len() {
        print!("id: {} ", settings.candidate_settings[i].config_id);
    }
    println!("recycled vector");
    for i in 0 .. settings.recycled_settings.len() {
        print!("id: {} ", settings.recycled_settings[i].config_id);
    }
    println!("\nCandidate trainers");
    for i in 0 .. settings.candidate_trainers.len() {
        print!("id: {} ", settings.candidate_trainers[i].get_trainee_id());
    }
    println!("\nRecycled trainers");
    for i in 0 .. settings.recycled_trainers.len() {
        print!("id: {} ", settings.recycled_trainers[i].get_trainee_id());
    }
    println!("\n\n\n\n");
}

fn min_accuracy_inc_not_reached(prev_accuracy : f64, settings : &mut InternalSettings) -> bool {
    let mut max_inc_threshold : f64 = 0.2;
    if settings.max_accuracy > 90.0 && settings.max_accuracy <= 92.0 {
        max_inc_threshold = 0.1;  
    } else if settings.max_accuracy > 92.0 && settings.max_accuracy <= 94.0 {
        max_inc_threshold = 0.08;
    } else if settings.max_accuracy > 94.0 && settings.max_accuracy <= 96.0 {
        max_inc_threshold = 0.04;
    } else if settings.max_accuracy >= 98.0 && settings.max_accuracy <= 99.0 {
        max_inc_threshold = 0.01;
    } else if settings.max_accuracy > 99.0 && settings.max_accuracy <= 99.4 {
        max_inc_threshold = 0.001;
    } else if settings.max_accuracy > 99.4 && settings.max_accuracy <= 99.6 {
        max_inc_threshold = 0.0005;
    } else if settings.max_accuracy > 99.6 && settings.max_accuracy <= 99.8 {
        max_inc_threshold = 0.0001;
    }  else if settings.max_accuracy > 99.8 {
        max_inc_threshold = 0.00005;
    }
    if settings.max_accuracy - prev_accuracy < max_inc_threshold {
        return true
    }
    false
}

fn minimum_not_reached(settings : &mut InternalSettings) -> bool {
    !(settings.breeder.min_ga_mutations_reached() && settings.breeder.min_sa_mutations_reached())
}

fn tune_hyper_parameters_w_sa(settings : &mut InternalSettings, data : &DataSet, mut prev_accuracy : f64) -> f64 {
    println!("tune_hyper_parameters_w_sa");
    while !settings.breeder.min_temp_reached() {
        for i in 0 .. settings.most_accurate_settings.len() {
            let mut child : NeuralNetSettings = *get_child_settings(settings);
            settings.breeder.child_w_sa_hyper_parameters(
                &mut child, 
                &mut settings.most_accurate_settings[i]
            );
            settings.candidate_settings.push(Box::new(child));
        }
        settings.breeder.drop_temp();
    }
    initiate_training_round(settings, data);
    prev_accuracy = settings.max_accuracy;
    settings.breeder.reset_temp_to_adjusted_max();
    prev_accuracy
}

fn tune_hyper_parameters_w_ga(settings : &mut InternalSettings, data : &DataSet, mut prev_accuracy : f64) -> f64 {
    println!("tune_hyper_parameters_w_ga");
    while !settings.breeder.min_current_p1_fav_reached() {
        let most_accurate_length : usize = settings.most_accurate_settings.len();
        for i in 0 .. most_accurate_length {
            for j in 0 .. most_accurate_length {
                if i == j {
                    continue;
                }
                let mut child : NeuralNetSettings = *get_child_settings(settings);
                settings.breeder.child_w_ga_hyper_params(
                    &mut child,
                    &settings.most_accurate_settings[i],
                    &settings.most_accurate_settings[j]
                );
                settings.candidate_settings.push(Box::new(child));
            }
        }
        settings.breeder.drop_p1_favourability();
        initiate_training_round(settings, data);
        if min_accuracy_inc_not_reached(prev_accuracy, settings) {
            println!("min accuracy inc. in h.p. ga not reached.");
            break;
        } else {
            prev_accuracy = settings.max_accuracy;
        }
        if acceptable_accuracy_reached(&settings) {
            return prev_accuracy;
        }
    }
    settings.breeder.reset_p1_fav_to_adjusted_max();
    prev_accuracy
}

fn initiate_training_round(settings : &mut InternalSettings, data : &DataSet) {
    settings.breeder.inc_generation_count();
    prepare_trainers(settings);
    train_parallel(settings, data);
    collect_winners(settings);
    reset_weights(settings);
}

fn get_recycled_trainer(settings : &mut InternalSettings) -> Box<NNetTrainer> {
    // don't check candidate_trainers,
    // since it should always be empty when this function is called.
    if settings.recycled_trainers.len() > 0 {
        let trainer_opt : Option<Box<NNetTrainer>> = settings.recycled_trainers.pop();
        match trainer_opt {
            Some(trainer) => {
                return trainer;
            } None => {
                panic!(no_recycled_trainer());
            }
        }
    } else {
        let current_default : &NeuralNetSettings = &*settings.optimizer_params.current_candidate_configuration;
        let train_cutoff_idx : usize = settings.optimizer_params.test_train_cutoff_idx;
        let training_rounds_per_epoch : usize = settings.optimizer_params.train_rounds_per_epoch;
        Box::new(NNetTrainer::new(current_default, train_cutoff_idx,training_rounds_per_epoch))
    }
}

fn get_child_settings(settings : &mut InternalSettings) -> Box<NeuralNetSettings> {
    if settings.recycled_settings.len() > 0 {
        let settings_opt : Option<Box<NeuralNetSettings>> = settings.recycled_settings.pop();
        match settings_opt {
            Some(child_settings) => {
                return child_settings;
            } None => {
                panic!(no_recycled_settings());
            }
        }
    } else {
        let current_settings : &NeuralNetSettings = &*settings.optimizer_params.current_candidate_configuration;
        let mut new_settings : NeuralNetSettings = current_settings.clone();
        new_settings.config_id = settings.id_sequence;
        settings.id_sequence += 1;
        println!("new settings id: {}", new_settings.config_id);
        Box::new(new_settings)
    }
}

fn collect_final_settings(settings : &mut InternalSettings, order_form : &mut OptimizerParameters) {
    while settings.most_accurate_settings.len() > 0 {
        let mut high_score : f64 = 0.0;
        let mut high_score_idx : usize = 0;
        for i in 0 .. settings.most_accurate_settings.len() {
            if settings.most_accurate_settings[i].accuracy > high_score {
                high_score = settings.most_accurate_settings[i].accuracy;
                high_score_idx = i;
            }
        }
        let winner = settings.most_accurate_settings.remove(high_score_idx);
        if winner.accuracy < settings.optimizer_params.min_acceptable_accuracy {
            continue;
        }
        order_form.tuned_settings.push(winner);
        if order_form.tuned_settings.len() >= order_form.final_number_of_nnet_settings {
            break;
        }
    }
}

fn train_parallel(settings : &mut InternalSettings, data : &DataSet) {
    process_parallel(settings, data, true);
}

fn train_single(trainer : &mut Box<NNetTrainer>, shared_data : &Arc<&DataSet>) {
    let data : &DataSet = &shared_data;
    trainer.train(data);
    trainer.test(data);
}

fn collect_winners(settings : &mut InternalSettings) {
    copy_trainer_accuracy_to_settings(settings);
    expand_most_accurate_settings(settings);
    promote_candidates_to_most_accurate(settings);
    recycle_settings_and_trainers(settings);
    set_top_score(settings);
}

fn copy_trainer_accuracy_to_settings(settings : &mut InternalSettings) {
    for i in 0 .. settings.candidate_trainers.len() {
        for j in 0 .. settings.candidate_settings.len() {
            if settings.candidate_trainers[i].get_trainee_id() ==
               settings.candidate_settings[j].config_id
            {
                settings.candidate_settings[j].accuracy =
                settings.candidate_trainers[i].get_test_result();
                break;
            }
        }
    }
}

fn expand_most_accurate_settings(settings : &mut InternalSettings) {
    while settings.most_accurate_settings.len() < settings.optimizer_params.winners_per_round {
        if settings.candidate_settings.len() <= 0 {
            break;
        }
        let last : usize = settings.candidate_settings.len() - 1;
        settings.most_accurate_settings.push(
            settings.candidate_settings.remove(last)
        );
    }
}

fn promote_candidates_to_most_accurate(settings : &mut InternalSettings) {
    for i in (0 .. settings.candidate_settings.len()).rev() {
        for j in 0 .. settings.most_accurate_settings.len() {
            if settings.candidate_settings[i].accuracy > settings.most_accurate_settings[j].accuracy {
                settings.recycled_settings.push(
                    settings.most_accurate_settings.remove(j)
                );
                settings.most_accurate_settings.push(
                    settings.candidate_settings.remove(i)
                );
                break;
            }
        }
    }
}

fn recycle_settings_and_trainers(settings : &mut InternalSettings) {
    while settings.candidate_settings.len() > 0 {
        let candidate_opt : Option<Box<NeuralNetSettings>> = settings.candidate_settings.pop();  
        match candidate_opt {
            Some(mut candidate) => {
                settings.recycled_settings.push(candidate);
            } None => {
                panic!("No candidate found");
            }
        }
    }
    while settings.candidate_trainers.len() > 0 {
        let trainer_opt : Option<Box<NNetTrainer>> = settings.candidate_trainers.pop();
        match trainer_opt {
            Some(trainer) => {
                settings.recycled_trainers.push(trainer);
            } None => {
                panic!("Not trainer found!");
            }
        }
    }
}

fn set_top_score(settings : &mut InternalSettings) {
    let mut max_accuracy : f64 = 0.0;
    for i in 0 .. settings.most_accurate_settings.len() {
        print!("score: {}, id: {} |> ", 
            settings.most_accurate_settings[i].accuracy,
            settings.most_accurate_settings[i].config_id
        );
        if settings.most_accurate_settings[i].accuracy > max_accuracy {
            max_accuracy = settings.most_accurate_settings[i].accuracy;
        }
    }
    if max_accuracy > settings.max_accuracy {
        settings.max_accuracy = max_accuracy;
    }
    println!("\nMax accuracy from training session: {}", max_accuracy);
}

// lines up trainers for training with the most up to date settings, including weights
fn prepare_trainers(settings : &mut InternalSettings) {
    for i in 0 .. settings.candidate_settings.len() {
        let mut recycled_trainer : NNetTrainer = *get_recycled_trainer(settings);
        recycled_trainer.update_trainee(
            &settings.candidate_settings[i],
            settings.optimizer_params.test_train_cutoff_idx
        );
        settings.candidate_trainers.push(
            Box::new(recycled_trainer)
        );
    }
}

fn alter_winners_per_round_if_final_round(i : usize, j : usize, settings : &mut InternalSettings) {
    if j >= settings.optimizer_params.max_train_epochs - 1 && 
       i >= settings.optimizer_params.max_config_changing_epochs - 1 {
        settings.optimizer_params.winners_per_round = 
        settings.optimizer_params.final_number_of_nnet_settings;
    } else if acceptable_accuracy_reached(settings) {
        settings.optimizer_params.winners_per_round = 
        settings.optimizer_params.final_number_of_nnet_settings;
    }
}

fn acceptable_accuracy_reached(settings : &InternalSettings) -> bool {
    settings.max_accuracy >= settings.optimizer_params.min_acceptable_accuracy
}



fn reset_weights(settings : &mut InternalSettings) {
    if !settings.reset_weights {
        return;
    }
    let mut rng = rand::thread_rng();
    for i in 0 .. settings.recycled_settings.len() {
        for j in 0 .. settings.recycled_settings[i].layers.len() {
            let low = settings.recycled_settings[i].layers[j].weight_range[0];
            let high = settings.recycled_settings[i].layers[j].weight_range[1];
            match &mut settings.recycled_settings[i].layers[j].layer_weights {
                Some(weight_vectors) => {
                    for l in 0 .. weight_vectors.len() {
                        for m in 0 .. weight_vectors[l]. len() {
                            weight_vectors[l][m] = rng.gen_range(low, high);
                        }
                    }
                } None => {
                    panic!("no weights to update");
                }
            }
        }
    }
    println!("Done resetting weights.");
}


fn tune_weights(settings : &mut InternalSettings, data : &DataSet, mut prev_accuracy : f64) -> f64 {
    settings.breeder.total_reset();
    println!("tune_weights");
    //settings.reset_weights = false;
    while minimum_not_reached(settings) {
        prev_accuracy = tune_weights_w_sa(settings, data, prev_accuracy);
        prev_accuracy = tune_weights_w_ga(settings, data, prev_accuracy);
        if min_accuracy_inc_not_reached(prev_accuracy, settings) {
            println!("Didn't reach increase in accuracy.");
            break;
        }
    }
    prev_accuracy
}

fn tune_weights_w_sa(settings : &mut InternalSettings, data : &DataSet, mut prev_accuracy : f64) -> f64 {
    while !settings.breeder.min_temp_reached() {
        for i in 0 .. settings.most_accurate_settings.len() {
            let mut child : NeuralNetSettings = *get_child_settings(settings);
            settings.breeder.child_w_sa_weights(
                &mut child, 
                &mut settings.most_accurate_settings[i]
            );
            settings.candidate_settings.push(Box::new(child));
        }
        settings.breeder.drop_temp();
    }
    initiate_searching_round(settings, data);
    prev_accuracy = settings.max_accuracy;
    settings.breeder.reset_temp_to_adjusted_max();
    prev_accuracy
}

fn tune_weights_w_ga(settings : &mut InternalSettings, data : &DataSet, mut prev_accuracy : f64) -> f64 {
    while !settings.breeder.min_current_p1_fav_reached() {
        let most_accurate_length : usize = settings.most_accurate_settings.len();
        for i in 0 .. most_accurate_length {
            for j in 0 .. most_accurate_length {
                if i == j {
                    continue;
                }
                let mut child : NeuralNetSettings = *get_child_settings(settings);
                settings.breeder.child_w_ga_weights(
                    &mut child,
                    &settings.most_accurate_settings[i],
                    &settings.most_accurate_settings[j]
                );
                settings.candidate_settings.push(Box::new(child));
                child = *get_child_settings(settings);
                settings.breeder.child_w_ga_weights_swap(
                    &mut child,
                    &settings.most_accurate_settings[i],
                    &settings.most_accurate_settings[j]
                );
                settings.candidate_settings.push(Box::new(child));
            }
        }
        settings.breeder.drop_p1_favourability();
        initiate_searching_round(settings, data);
        if min_accuracy_inc_not_reached(prev_accuracy, settings) {
            println!("Minimum accuracy inc. not reached in tune weights with ga. leaving loop");
            break;
        } else {
            prev_accuracy = settings.max_accuracy;
        }
    }
    settings.breeder.reset_p1_fav_to_adjusted_max();
    prev_accuracy
}

fn initiate_searching_round(settings : &mut InternalSettings, data : &DataSet) {
    settings.breeder.inc_generation_count();
    let cutoff : usize = settings.optimizer_params.test_train_cutoff_idx;
    settings.optimizer_params.test_train_cutoff_idx = 0;
    prepare_trainers(settings);
    search_parallel(settings, data);
    settings.optimizer_params.test_train_cutoff_idx = cutoff; 
    collect_winners(settings);
}

fn create_workload(settings : &mut InternalSettings, workload : &mut Vec<Vec<Box<NNetTrainer>>>) {
    let mut num_cpus : usize = get();
    if settings.optimizer_params.cpus_to_use < num_cpus {
        num_cpus = settings.optimizer_params.cpus_to_use;
        if num_cpus < 2 {
            num_cpus = 2;
        }
    }
    for i in 0 .. num_cpus {
        let mut thread_task : Vec<Box<NNetTrainer>> = Vec::new();
        workload.push(thread_task);
    }
    let mut i : usize = 0;
    while settings.candidate_trainers.len() > 0 {
        let trainer_opt = settings.candidate_trainers.pop();
        match trainer_opt {
            Some(trainer_box) => {
                workload[i].push(trainer_box);
            } None => {
                panic!("Could not aqquire trainer.");
            }
        }
        i += 1;
        i %= workload.len();
    }
}

fn search_parallel(settings : &mut InternalSettings, data : &DataSet) {
    process_parallel(settings, data, false);
}

fn process_parallel(settings : &mut InternalSettings, data : &DataSet, tune_params : bool) {
    let mut shared_data : Arc<&DataSet> = Arc::new(data);
    let mut t_handles = vec![];
    let mut workload : Vec<Vec<Box<NNetTrainer>>> = Vec::new();
    create_workload(settings, &mut workload);

    scope(
        |scope| {
            while workload.len() > 0 {
                let copy_data = shared_data.clone();
                let trainers_opt = workload.pop();
                match trainers_opt {
                    Some(trainer_box_vec) => {
                        t_handles.push(
                            scope.spawn(
                                move || {
                                    process_batch(
                                        trainer_box_vec, 
                                        copy_data,
                                        tune_params
                                    )
                                }
                            )
                        );
                    } None => {
                        panic!("Could not aqquire trainer.");
                    }
                }
            }
        }
    );
    for handler in t_handles {
        let mut trainers : Vec<Box<NNetTrainer>> = handler.join();
        while trainers.len() > 0 {
            let mut trainer_opt : Option<Box<NNetTrainer>> = trainers.pop();
            match trainer_opt {
                Some(trainer) => {
                    settings.candidate_trainers.push(trainer);
                } None => {
                    panic!("Could not find nnet trainer");
                }
            }
        }
    }
}

fn process_batch(
    mut trainer_box_vec : Vec<Box<NNetTrainer>>, 
    shared_data : Arc<&DataSet>, 
    tune_params : bool
) -> Vec<Box<NNetTrainer>> 
{
    if tune_params {
        for i in 0 .. trainer_box_vec.len() {
            train_single(&mut trainer_box_vec[i], &shared_data);
        }
    } else {
        for i in 0 .. trainer_box_vec.len() {
            search_single(&mut trainer_box_vec[i], &shared_data);
        }
    }
    trainer_box_vec
}

fn search_single(trainer_box : &mut Box<NNetTrainer>, shared_data : &Arc<&DataSet>) {
    let data : &DataSet = &shared_data;
    trainer_box.test(data);
}

fn make_settings(order_form : &mut OptimizerParameters) -> InternalSettings {
    let temp_drop_amt : f64 = 0.99 / order_form.temperature_drops as f64;
    let heritability_bias_drop : f64 = 0.4 / order_form.heritability_bias_drops as f64;
    let mut settings : InternalSettings = InternalSettings {
        optimizer_params : order_form.clone(),
        most_accurate_settings : Vec::new(),
        candidate_settings : Vec::new(),
        recycled_settings : Vec::new(),
        candidate_trainers : Vec::new(),
        recycled_trainers : Vec::new(),
        breeder : Breeder::new(temp_drop_amt, heritability_bias_drop),
        max_accuracy : 0.0,
        reset_weights : true,
        id_sequence : 2 // bc original settings is pushed onto vector below
    };
    settings.optimizer_params.current_candidate_configuration.config_id = 1;
    validate_and_adjust_parameters(&mut settings);
    settings.most_accurate_settings.push(
        settings.optimizer_params.current_candidate_configuration.clone()
    );
    settings
}

fn validate_and_adjust_parameters(settings : &mut InternalSettings) {
    // add weight vectors if they are missing
    // will need to complete code change to update neural network
    let mut rng = rand::thread_rng();
    let mut input_size : usize = settings.optimizer_params.current_candidate_configuration.input_size;
    let mut layers : &mut Vec<LayerSettings> = &mut settings.optimizer_params.current_candidate_configuration.layers;
    for i in 0 .. layers.len() {
        let low : f64 = layers[i].weight_range[0];
        let high : f64 = layers[i].weight_range[1];
        let mut new_layer_weights : Vec<Vec<f64>> = Vec::new();
        for j in 0 .. layers[i].output_units {
            new_layer_weights.push(vec![rng.gen_range(low, high); input_size]);
        }
        layers[i].layer_weights = Some(new_layer_weights);
        input_size = layers[i].output_units;
    }
    if settings.optimizer_params.winners_per_round < 2 {
        settings.optimizer_params.winners_per_round = 2;
    }
    let cores_available : usize = num_cpus::get();
    if settings.optimizer_params.cpus_to_use > cores_available {
        settings.optimizer_params.cpus_to_use = cores_available;
    } else if settings.optimizer_params.cpus_to_use < 1 {
        settings.optimizer_params.cpus_to_use = 1;
    }
}

fn alter_activation_fns_and_optimizers(settings : &mut InternalSettings) {
    println!("alter_activation_fns_and_optimizers");

    // layer is final layer, only use activation functions that work for final layer
}

fn increase_nodes(settings : &mut InternalSettings) {
    // trainers can be re - used, since they will make the nnet resize itself.


    // if layer[i] + 1 is final, new layer must match final layers outputs

    
}

fn trim_nodes(settings : &mut InternalSettings, data : &DataSet) {

}

#[allow(dead_code)]
fn no_recycled_settings() -> String {
    "Optimize_nnet_settings failed in function get_child_settings".to_string()
}

#[allow(dead_code)]
fn no_winners_to_add_to_candidate_list() -> String {
    "Optimize_nnet_settings failed in function add_prev_winners_to_candidates".to_string()
}

#[allow(dead_code)]
fn no_recycled_trainer() -> String {
    "No recycled trainer found".to_string()
}