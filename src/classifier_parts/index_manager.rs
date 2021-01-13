use rand::Rng;

pub struct IndexManager {
    path : Vec<usize>,
    current_idx : usize,
    max_index : usize
}

impl IndexManager {
    pub fn new() -> IndexManager {
        IndexManager {
            path : Vec::new(),
            current_idx : 0,
            max_index : 0
        }
    }

    pub fn update_random_path_len(&mut self, path_len: usize) {
        let current_len : usize = self.path.len();
        if path_len > current_len {
            for _ in current_len .. path_len {
                self.path.push(0);
            }
            self.max_index = self.path.len()-1;
        } else if path_len < 1 {
            panic!("Invalid path length for Index Manager");
        } else if path_len < current_len {
            self.max_index = path_len - 1;
        }
        for i in 0 .. self.max_index {
            self.path[i] = i;
        }
    }

    pub fn has_next(&self) -> bool {
        self.current_idx <= self.max_index
    }

    pub fn next(&mut self) -> usize {
        let idx : usize = self.path[self.current_idx];
        self.current_idx += 1;
        idx
    }

    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0 .. self.max_index {
            let rand : usize = rng.gen_range(0, self.max_index);
            let temp : usize = self.path[rand];
            self.path[rand] = self.path[i];
            self.path[i] = temp;
        }
        self.current_idx = 0;
    }
}