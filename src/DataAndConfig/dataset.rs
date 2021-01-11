use serde_derive::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct DataSet {
    pub table_info : TableInfo,
    pub result_map : Vec<String>,
    pub data : Vec<Row>
}

#[derive(Serialize, Deserialize)]
pub struct TableInfo {
    pub table_name : String,
    pub query_id : usize,
    pub column_names : Option<Vec<String>>
}

#[derive(Serialize, Deserialize)]
pub struct Row {
    pub label : usize,
    pub columns : Vec<f64>,
}