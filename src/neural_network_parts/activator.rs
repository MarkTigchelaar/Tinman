
use std::f64::consts::{E, PI};


use super::nnet_errors::{
    set_act_fn_code_out_of_bounds, 
    fn_name_not_found
};

const LAMBDA : f64 = 1.6732632423543772848170429916717;
const ALPHA : f64 = 1.0507009873554804934193349852946;

pub struct Activator {
    fn_names : [String; 12],
    activation_fn_code : usize,
    activation_fns : [fn(f64) -> f64; 12],
    activation_prime_fns : [fn(f64) -> f64; 12]
}

impl Activator {
    pub fn new () -> Activator {
        Activator {
            activation_fn_code : 0,
            fn_names : [
                "default".to_string(),
                "sigmoid".to_string(),
                "binary_step".to_string(),
                "tanh".to_string(),
                "sqnl".to_string(),
                "arctan".to_string(),
                "lrelu".to_string(),
                "elu".to_string(),
                "selu".to_string(),
                "gelu".to_string(),
                "softplus".to_string(),
                "swish".to_string()
            ],
            activation_fns : [
                default,
                sigmoid,
                binary_step,
                tanh,
                sqnl,
                arctan,
                lrelu,
                elu,
                selu,
                gelu,
                softplus,
                swish
            ],
            activation_prime_fns : [
                default_prime,
                sigmoid_prime,
                binary_step_prime,
                tanh_prime,
                sqnl_prime,
                arctan_prime,
                lrelu_prime,
                elu_prime,
                selu_prime,
                gelu_prime,
                softplus_prime,
                swish
            ]
        }
    }

    pub fn activate (&mut self, x : f64) -> f64 {
        (self.activation_fns)[self.activation_fn_code](x)
    }
 
    pub fn activate_prime (&mut self, x : f64) -> f64 {
        (self.activation_prime_fns)[self.activation_fn_code](x)
    }

    pub fn set_fn_code (&mut self, code : usize) {
        if code > self.activation_fns.len() {
            panic!(set_act_fn_code_out_of_bounds());
        }
        self.activation_fn_code = code;
    }

    pub fn get_fn_code_by_name (&mut self, fn_name : &String) -> usize {
        for (i, name) in self.fn_names.iter().enumerate() {
            if fn_name == name {
                return i
            }
        }
        panic!(fn_name_not_found());
    }
/*
    pub fn get_fn_name_from_code(&mut self, fn_code : usize) -> String {
        if fn_code > self.fn_names.len() {
           panic!("Activator cannot find activation function from code number {}!", fn_code);
        }
        let bytes = &self.fn_names[fn_code].into_bytes();
        String::from_utf8(bytes.to_vec()).expect("Found invalid UTF-8")
    }
*/
}

fn default(x : f64) -> f64 {
    x
}

fn default_prime(x : f64) -> f64 {
    x
}

fn sigmoid(x : f64) -> f64 {
    if x > 0.0 {
        return 1. / ( 1. + ( E.powf( - x ) ) )
    }
    let temp = E.powf( x );
    return temp / (1.0 + temp)
}

fn sigmoid_prime(x : f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1. - sig)
}

fn binary_step(x : f64) -> f64 {
    if x >= 0. {
        return 1.
    }
    return -1.
}

fn binary_step_prime(x : f64) -> f64 {
    x
}

fn tanh(x : f64) -> f64 {
    let a = E.powf(x);
    let b = E.powf(-x);
    (a - b) / (a + b)
}

fn tanh_prime(x : f64) -> f64 {
    let tanh = tanh(x);
    1. - tanh.powi(2)
}

fn sech(x : f64) -> f64 {
    2.0 / (E.powf(x) + E.powf(-x))
}

fn sqnl(x : f64) -> f64 {
    if x > 2. {
        return 0.
    } else if (0. <= x) & (x <= 2.) {
        return x - (x.powi(2) / 4.)
    } else if (-2. <= x) & (x < 0.) {
        return x + (x.powi(2) / 4.)
    } else {
        -1.
    }
}

fn sqnl_prime(x : f64) -> f64 {
    if x > 2. {
        return 0.
    } else if (0. <= x) & (x <= 2.) {
        return 1. - (x / 2.)
    } else if (-2. <= x) & (x < 0.) {
        return x + (x / 2.)
    } else {
        return 0.
    }
}

fn arctan(x : f64) -> f64 {
    x.atan()
}

fn arctan_prime(x : f64) -> f64 {
    1. / (x.powi(2) + 1.)
}

fn lrelu(x : f64) -> f64 {
    if x < 0. {
        return 0.01 * x
    }
    x
}

fn lrelu_prime(x : f64) -> f64 {
    if x < 0. {
        return 0.01
    }
    1.
}

fn elu(x : f64) -> f64 {
    if x <= 0.0 {
        ALPHA * (E.powf( x ) - 1.0)
    } else {
        x
    }
}

fn elu_prime(x : f64) -> f64 {
    if x <= 0.0 {
        elu(x) + ALPHA
    } else {
        1.0
    }
}

fn selu(x : f64) -> f64 {
    if x <= 0.0 {
        LAMBDA * (E.powf( x ) - 1.0)
    } else {
        ALPHA * x
    }
}

fn selu_prime(x : f64) -> f64 {
    if x <= 0.0 {
        elu(x) + LAMBDA
    } else {
        ALPHA
    }
}

fn gelu(x : f64) -> f64 {
    0.5 * x * (1.0 + tanh((2.0 / PI).sqrt() * (x + (0.044715 * ( x * x * x )))))
}

fn gelu_prime(x : f64) -> f64  {
    let term1 = 0.0356774 * x * x * x;
    let term2 = 0.398942 * x;
    let term3 = 0.797885 * x;
    let term4 = 0.0535161  * x * x * x;
    let mut hyp_secant = sech(term1 + term3);
    hyp_secant *= hyp_secant;
    0.5 * tanh(term1 + term3) + (term4 + term2) * hyp_secant + 0.5
}

fn softplus(x : f64) -> f64 {
    (1.0 + E.powf( x )).ln()
}

fn softplus_prime(x : f64) -> f64 {
    sigmoid(x)
}

fn swish(x : f64) -> f64 {
    x * sigmoid(x)
}

fn swish_prime(x : f64) -> f64 {
    x * sigmoid_prime(x) + sigmoid(x)
}