use serde_derive::{Deserialize};


#[derive(Deserialize)]
pub struct TestConfig {
    pub settings_path : String,
    pub data_path : String,
    pub target_object : String,
    pub tests : Vec<Test>
}

#[derive(Deserialize)]
pub struct Test {
    pub settings : String,
    pub bulk_settings : Vec<String>,
    pub data : String,
    pub rounds : usize,
    pub training_cutoff_index : usize
}