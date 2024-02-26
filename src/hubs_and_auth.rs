use crate::utils::{normalise_sqrt, transpose};

const NUM_NEIGHBOURS: usize = 5;
const NUM_ITER: usize = 50;

fn run(
    am: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    initial_state_hubs: [f32; NUM_NEIGHBOURS],
    initial_state_auth: [f32; NUM_NEIGHBOURS],
) -> ([f32; NUM_NEIGHBOURS], [f32; NUM_NEIGHBOURS]) {
    let mut s_hubs = initial_state_hubs.clone();
    let mut s_auth = initial_state_auth.clone();
    let transposed_am = transpose(am);

    println!("start hubs: [{}]", s_hubs.map(|v| format!("{:>9.4}", v)).join(", "));
    println!("start auth: [{}]", s_auth.map(|v| format!("{:>9.4}", v)).join(", "));
    for _ in 0..NUM_ITER {
        let mut new_s_hubs = [0.; 5];
        let mut new_s_auth = [0.; 5];

        // Hubs
        for i in 0..NUM_NEIGHBOURS {
            for j in 0..NUM_NEIGHBOURS {
                new_s_hubs[i] += am[j][i] * s_auth[j];
            }
        }
        // Authorities
        for i in 0..NUM_NEIGHBOURS {
            for j in 0..NUM_NEIGHBOURS {
                new_s_auth[i] += transposed_am[j][i] * s_hubs[j];
            }
        }

        let final_s_hubs = normalise_sqrt(new_s_hubs);
        let final_s_auth = normalise_sqrt(new_s_auth);

        s_hubs = final_s_hubs;
        s_auth = final_s_auth;
    }
    println!("end hubs: [{}]", s_hubs.map(|v| format!("{:>9.4}", v)).join(", "));
    println!("end auth: [{}]", s_auth.map(|v| format!("{:>9.4}", v)).join(", "));

    (s_hubs, s_auth)
}

pub fn run_job() {
    // From hubs to authorities
    let adjacency_matrix: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
      [0., 1., 0., 1., 0.],
      [0., 0., 0., 1., 0.],
      [1., 0., 0., 1., 1.],
      [0., 1., 0., 0., 0.],
      [0., 1., 0., 1., 0.],
    ];
    let initial_state_hubs = [0.32, 0.0, 0.22, 0.0, 0.66];
    let initial_state_auth = [0.32, 0.11, 0.14, 0.1, 0.33];
    run(adjacency_matrix, initial_state_hubs, initial_state_auth);
}
