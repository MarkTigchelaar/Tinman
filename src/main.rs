// exists
pub mod NeuralNetworkParts;
pub mod DataAndConfig;
pub mod ClassifierParts;
// want to use certain parts
use NeuralNetworkParts::neural_network::NeuralNetwork;
use DataAndConfig::neural_net_config_parts::NeuralNetSettings;
use DataAndConfig::dataset::DataSet;
use DataAndConfig::test_config::{TestConfig, Test};
use ClassifierParts::nnet_trainer::NNetTrainer;
use ClassifierParts::breeder::Breeder;
use ClassifierParts::classifier::Classifier;

use serde_json;
use std::fs::File;
use std::io::prelude::*;
use std::env;
use rand::Rng;
use std::io::Write;


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 2 {
        panic!("Test harness not passed correct number of arguments.");
    } else if args.len() < 2 {
        return;
    }
    let filename : &String = &args[1];
    let tests : TestConfig = get_tests(&filename);
    run_test(&tests);
}

fn get_tests(filename : &String) -> TestConfig {
    let mut file_handle = File::open(filename).expect("Can't open the test file.");
    let mut json_string = String::new();
    file_handle.read_to_string(&mut json_string).expect("Failed to read data into string.");
    let tests : TestConfig = serde_json::from_str(&json_string[..]).unwrap();
    tests
}

fn get_data(filename : &String) -> DataSet {
    let mut file_handle = File::open(filename).expect("Can't open the data file.");
    let mut json_string = String::new();
    file_handle.read_to_string(&mut json_string).expect("Failed to read data into string.");
    let data : DataSet = serde_json::from_str(&json_string[..]).unwrap();
    data
}

fn get_settings(filename : &String) -> NeuralNetSettings {
    let mut file_handle = File::open(filename).expect("Can't open the settings file.");
    let mut json_string = String::new();
    file_handle.read_to_string(&mut json_string).expect("Failed to read settings into string.");
    let settings : NeuralNetSettings = serde_json::from_str(&json_string[..]).unwrap();
    settings
}

fn run_test(test_run : &TestConfig) {
    if test_run.target_object == "NeuralNetwork".to_string() {
        test_neural_network(test_run);
    } else if test_run.target_object == "NNetTrainer".to_string() {
        test_neural_network_trainer(test_run);
    } else if test_run.target_object == "Breeder".to_string() {
        test_breeder(test_run);
    } else if test_run.target_object == "Classifier".to_string() {
        test_classifier(test_run);
    }
}


fn test_neural_network(test_run : &TestConfig) {
    let mut rng = rand::thread_rng();

    let tests : &Vec<Test> = &test_run.tests;
    for i in 0 .. tests.len() {
        let test : &Test = &tests[i];
        let settings_file : &String = &test.settings;
        let data_file : &String = &test.data;
        let data : DataSet = get_data(data_file);
        let settings : NeuralNetSettings = get_settings(settings_file);
        let mut nnet : NeuralNetwork = NeuralNetwork::new(&settings);

        let rounds : usize = test.rounds;

        let mut correct : usize = 0;
        let total : usize = data.data.len();
        let mut rand_idx : Vec<usize> = vec![0; total];
        for l in 0 .. rand_idx.len() {
            rand_idx[l] = l;
        }
        for _ in 0 .. rounds {
            for m in 0 .. rand_idx.len() {
                let rand : usize = rng.gen_range(0, rand_idx.len() - 1);
                let temp : usize = rand_idx[rand];
                rand_idx[rand] = rand_idx[m];
                rand_idx[m] = temp;
            }
            for j in 0 .. total {
                let index = rand_idx[j];
                let columns : &Vec<f64> = &data.data[index].columns;
                let answer : usize = data.data[index].label;
                nnet.forward(columns);
                nnet.set_error_delta(answer);
                nnet.backward(columns);
            }
        }
        for a in 0 .. total {
            let columns : &Vec<f64> = &data.data[a].columns;
            let answer : usize = data.data[a].label;
            let prediction : usize = nnet.predict(columns);
            if prediction == answer {
                correct += 1;
            }
        }
        println!("Test {} resulted in {} / {} correct.", i + 1, correct, total);
    }
}

fn test_neural_network_trainer(test_run : &TestConfig) {
    let tests : &Vec<Test> = &test_run.tests;
    for i in 0 .. tests.len() {
        let test : &Test = &tests[i];
        let settings_file : &String = &test.settings;
        let data_file : &String = &test.data;
        let data : DataSet = get_data(data_file);
        let settings : NeuralNetSettings = get_settings(settings_file);
        let stop_idx : usize = test.training_cutoff_index;
        let mut trainer : NNetTrainer = NNetTrainer::new(&settings, stop_idx, test.rounds);
        trainer.train(&data);
        trainer.test(&data);
        let correct : usize = trainer.get_test_result();
        let total : usize = data.data.len();
        let boundary : usize = trainer.get_train_test_boundary();
        println!("Test {} resulted in {} / {} correct.", i + 1, correct, total - boundary);
    }
}

fn test_breeder(test_run : &TestConfig) {
    let tests : &Vec<Test> = &test_run.tests;
    for i in 0 .. tests.len() {
        let test : &Test = &tests[i];
        let settings_file : &String = &test.settings;
        let mut settings : NeuralNetSettings = get_settings(settings_file);
        let temp_drop = 0.1;
        let fave_drop = 0.1;
        let mut breeder : Breeder = Breeder::new(temp_drop, fave_drop);
        let mut child : NeuralNetSettings = breeder.child_w_sa_hyper_parameters(&mut settings);
        let mut serialized = serde_json::to_string(&child).unwrap();
        let mut name : String = "SA_params_breeder_result.json".to_string();
        let mut file = std::fs::File::create(name).expect("create failed");
        file.write_all(serialized.as_bytes()).expect("write failed");

        child = breeder.child_w_sa_weights(&mut settings);
        serialized = serde_json::to_string(&child).unwrap();
        name = "SA_weights_breeder_result.json".to_string();
        file = std::fs::File::create(name).expect("create failed");
        file.write_all(serialized.as_bytes()).expect("write failed");

        breeder.drop_p1_favourability();
        breeder.drop_p1_favourability();
        let mut settings2 : NeuralNetSettings = get_settings(&".\\settings\\default2_with_weights_config.json".to_string());
        child = breeder.child_w_ga_weights(&mut settings, &mut settings2);
        serialized = serde_json::to_string(&child).unwrap();
        name = "GA_weights_breeder_result.json".to_string();
        file = std::fs::File::create(name).expect("create failed");
        file.write_all(serialized.as_bytes()).expect("write failed");
        let mut child : NeuralNetSettings = settings.clone();
        breeder.child_w_ga_hyper_params(&mut child, &mut settings2);
        serialized = serde_json::to_string(&child).unwrap();
        name = "GA_params_breeder_result.json".to_string();
        file = std::fs::File::create(name).expect("create failed");
        file.write_all(serialized.as_bytes()).expect("write failed");
    }
}

fn test_classifier(test_run : &TestConfig) {
    let tests : &Vec<Test> = &test_run.tests;
    for i in 0 .. tests.len() {
        let data_file : &String = &tests[i].data;
        let data : DataSet = get_data(data_file);
        for j in 0 .. tests[i].bulk_settings.len() {
            let settings_file : &String = &tests[i].bulk_settings[j];
            let mut classifier : Classifier = Classifier::new(
                get_settings(settings_file)
            );
            classifier.set_train_cutoff(tests[i].training_cutoff_index);
            classifier.set_train_rounds(tests[i].rounds);
            println!("fitting...");
            classifier.fit(&data);
            println!("tested from config file");
        }
    }
}