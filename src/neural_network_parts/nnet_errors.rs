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

#[allow(dead_code)]
pub fn set_act_fn_code_out_of_bounds () -> String {
    "\n\nActivator panicked!\n\nAttempt to set activation function code outside listing of activation functions.".to_string()
}

#[allow(dead_code)]
pub fn fn_name_not_found () -> String {
    "\n\nActivator panicked!\n\nActivation function not found in list of functions.".to_string()
}