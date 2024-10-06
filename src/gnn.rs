use std::array::from_fn;

use rand::{thread_rng, Rng};
use rustydiff::reverse::{Diff, ScalarOps, Tape, Var};

const NUM_NEIGHBOURS: usize = 5;

fn forward_run<'a, const NUM_ITER: usize>(
    tp: &'a Tape<f32, ScalarOps>,
    weights: &[[Option<Var<'a, f32, ScalarOps>>; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    biases: &[Var<'a, f32, ScalarOps>; NUM_NEIGHBOURS],
    seed: &[Var<'a, f32, ScalarOps>; NUM_NEIGHBOURS],
) -> [Var<'a, f32, ScalarOps>; NUM_NEIGHBOURS] {
    let mut s: [Var<f32, ScalarOps>; NUM_NEIGHBOURS] = from_fn(|_| tp.var(0.));
    for i in 0..seed.len() {
        s[i] = tp.var(seed[i].data);
    }

    for _ in 0..NUM_ITER {
        let mut new_s = from_fn(|_| tp.var(0.));

        // Compute sum of incoming weights
        for i in 0..NUM_NEIGHBOURS {
            // Aggregate - agg = (w_ji/W_j) * x_j
            for j in 0..NUM_NEIGHBOURS {
                if let Some(w) = &weights[i][j] {
                    new_s[i] += w * &s[j];
                }
            }
            // Update - x_i' = b_i * x_i + agg
            new_s[i] = (&(&biases[i] * &s[i]) + &new_s[i]).tanh();
        }

        s = new_s;
    }

    s
}

pub fn run_job() {
    let tp = Tape::<f32, ScalarOps>::new();

    let mut rng = thread_rng();

    // Labels - desired scores for peer 0, 3, and 4 - the rest will be learned by the network
    let labels = vec![(0, tp.var(0.0)), (3, tp.var(0.2)), (4, tp.var(0.3))];

    let mut err = 100.0;

    // Starting with random seed values at each training run
    let seed = [
        tp.var(rng.gen_range(0.0..1.0)),
        tp.var(rng.gen_range(0.0..1.0)),
        tp.var(rng.gen_range(0.0..1.0)),
        tp.var(rng.gen_range(0.0..1.0)),
        tp.var(rng.gen_range(0.0..1.0)),
    ];

    for _ in 0..100 {
        // Initial weights
        let mut weights: [[Option<Var<f32, ScalarOps>>; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
            [
                None,
                None,
                Some(tp.var(rng.gen_range(0.0..1.0))),
                Some(tp.var(rng.gen_range(0.0..1.0))),
                Some(tp.var(rng.gen_range(0.0..1.0))),
            ],
            [
                None,
                None,
                None,
                Some(tp.var(rng.gen_range(0.0..1.0))),
                None,
            ],
            [
                Some(tp.var(rng.gen_range(0.0..1.0))),
                None,
                None,
                Some(tp.var(rng.gen_range(0.0..1.0))),
                Some(tp.var(rng.gen_range(0.0..1.0))),
            ],
            [
                None,
                None,
                None,
                None,
                Some(tp.var(rng.gen_range(0.0..1.0))),
            ],
            [
                None,
                Some(tp.var(rng.gen_range(0.0..1.0))),
                Some(tp.var(rng.gen_range(0.0..1.0))),
                None,
                None,
            ],
        ];
        // Initial biases
        let mut biases = [tp.var(0.), tp.var(0.), tp.var(0.), tp.var(0.1), tp.var(0.1)];
        // Learning rate starting value
        let mut lr = 0.01;

        // ------------------------------------------------------------------

        for i in 0..100 {
            // Do the message passing
            let res = forward_run::<20>(&tp, &weights, &biases, &seed);

            // Calculate the error - only the labeled nodes are involved
            let mut error = tp.var(0.0);
            for (index, label) in &labels {
                // Approx of ln10(x):
                // ax^{\frac{1}{a}}-a\ +4
                // a=1000
                // let e = label - &res[*index];
                // let a = tp.var(1000.);
                // let pow = a.powf(&tp.var(-1.0));
                // error += &(&(&a * &e.powf(&pow)) - &a) + &tp.var(4.);

                let abs = tp.var((label - &res[*index]).data.abs());
                error += &(label - &res[*index]).powf(&tp.var(2.0)) * &abs.powf(&tp.var(-1.));
            }

            if error.data < err {
                err = error.data;
                println!("smallest error: {}", error.data);
                println!(
                    "end: [{}]",
                    res.iter()
                        .map(|v| format!("{:>9.4}", v.data.clone()))
                        .collect::<String>()
                );
            }

            // Print out the error every N iterations
            if i % 99 == 0 {
                println!("curr_err: {}", error.data);
            }

            // Calculate the gradient for all variables
            error.reverse();

            // Update weights and bisases based on the error
            for i in 0..NUM_NEIGHBOURS {
                for j in 0..NUM_NEIGHBOURS {
                    if let Some(w) = &weights[i][j] {
                        weights[i][j] = Some(tp.var(w.data - w.grad() * lr));
                    }
                }
                biases[i] = tp.var(biases[i].data - biases[i].grad() * lr);
            }

            // Decay learning rate at each step
            lr = 0.999 * lr;
        }
    }
}
