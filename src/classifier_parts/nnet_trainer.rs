use super::super::neural_network_parts::neural_network::NeuralNetwork;
use super::super::data_and_config::neural_net_config_parts::NeuralNetSettings;
use super::super::data_and_config::dataset::{DataSet, Row};
use super::index_manager::IndexManager;

pub struct NNetTrainer {
    rand_index : IndexManager,
    trainee : NeuralNetwork,
    rounds : usize,
    train_test_boundary : usize,
    correct : usize,
    dataset_length : usize
}

impl NNetTrainer {
    pub fn new(
        settings: &NeuralNetSettings, 
        train_cutoff : usize, 
        rounds : usize
    ) -> NNetTrainer 
    {
        NNetTrainer {
            rand_index : IndexManager::new(),
            trainee : NeuralNetwork::new(settings),
            rounds : rounds,
            train_test_boundary : train_cutoff,
            correct : 0,
            dataset_length : 0
        }
    }

    pub fn train(&mut self, dataset : &DataSet) {
        if self.train_test_boundary > dataset.data.len() {
            self.rand_index.update_random_path_len(dataset.data.len());
        } else {
            self.rand_index.update_random_path_len(self.train_test_boundary);
        }
        self.dataset_length = dataset.data.len();
        let mut prev_accuracy : f64 = 0.0;
        for _ in 0 .. self.rounds {
            self.rand_index.reset();
            while self.rand_index.has_next() {
                let index : usize = self.rand_index.next();
                self.train_on_row(index, dataset);
            }
        }
    }

    fn train_on_row(&mut self, index : usize, dataset : &DataSet) {
        let row : &Row = &dataset.data[index];
        let row_data : &Vec<f64> = &row.columns;
        let label : usize = row.label;
        self.trainee.forward(row_data);
        self.trainee.set_error_delta(label);
        self.trainee.backward(row_data);
    }

    pub fn test(&mut self, dataset : &DataSet) {
        self.correct = 0;
        if self.train_test_boundary > dataset.data.len() {
            self.train_test_boundary = 0;
        }
        self.dataset_length = dataset.data.len();
        for i in self.train_test_boundary .. dataset.data.len() {
            let row : &Row = &dataset.data[i];
            let row_data : &Vec<f64> = &row.columns;
            let label : usize = row.label;
            let prediction : usize = self.trainee.predict(row_data);
            if prediction == label {
                self.correct += 1;
            }
        }
    }

    pub fn get_test_result(&self) -> f64 {
        let test_count : usize = self.dataset_length - self.train_test_boundary;
        (self.correct as f64 / test_count as f64) * 100.0
    }

    pub fn get_train_test_boundary(&self) -> usize {
        self.train_test_boundary
    }

    pub fn get_trainee_id(&self) -> usize {
        self.trainee.get_id()
    }

    pub fn update_trainee(&mut self, current_default : &NeuralNetSettings, new_cutoff : usize) {
        self.train_test_boundary = new_cutoff;
        self.trainee.update_state(current_default);
    }
}