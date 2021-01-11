#[allow(dead_code)]
pub fn weights_input_len_mismatch() -> String {
    "Length of weight vector for for hidden unit does not match length of inputs for layer!".to_string()
}

#[allow(dead_code)]
pub fn weights_node_count_mismatch() -> String {
    "Number of weight vectors does not match number of hidden units in layer!".to_string()
}

#[allow(dead_code)]
pub fn failed_prediction() -> String {
    "Neural network failed to find correct prediction!".to_string()
}