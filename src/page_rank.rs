use crate::utils::{normalise, vec_add};

const NUM_NEIGHBOURS: usize = 5;
const NUM_ITER: usize = 50;
const DAMPENING_AMOUNT: f32 = 0.8;

fn run(
    mut am: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    seed: [f32; NUM_NEIGHBOURS],
) -> [f32; NUM_NEIGHBOURS] {
    for i in 0..NUM_NEIGHBOURS {
        am[i] = normalise(am[i], seed);
    }

    let mut s = seed.clone();
    let pre_trusted_scores = seed.map(|x| x * DAMPENING_AMOUNT);

    println!("start: [{}]", s.map(|v| format!("{:>9.4}", v)).join(", "));
    for _ in 0..NUM_ITER {
        let mut new_s = [0.; 5];

        // Compute sum of incoming weights
        for i in 0..NUM_NEIGHBOURS {
            for j in 0..NUM_NEIGHBOURS {
                new_s[i] += am[j][i] * s[j];
            }
        }

        let global_scores = new_s.map(|x| (1. - DAMPENING_AMOUNT) * x);
        let current_s = vec_add(pre_trusted_scores, global_scores);

        s = current_s;
    }
    println!("end: [{}]", s.map(|v| format!("{:>9.4}", v)).join(", "));

    s
}

pub fn run_job() {
    // From hubs to authorities
    let adjacency_matrix: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
      [0., 0., 1., 1., 1.],
      [0., 0., 0., 1., 0.],
      [1., 0., 0., 1., 1.],
      [0., 0., 0., 0., 1.],
      [0., 1., 1., 0., 0.],
    ];
    let seed = [1., 4., 3., 5., 1.];
    run(adjacency_matrix, seed);
}
