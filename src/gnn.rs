use std::array::from_fn;

use rand::{thread_rng, Rng};
use rustydiff::reverse::{Diff, ScalarOps, Tape, Var};

const NUM_NEIGHBOURS: usize = 5;

fn positive_run<'a, const NUM_ITER: usize>(
    tp: &'a Tape<f32, ScalarOps>,
    weights: &[[Var<'a, f32, ScalarOps>; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    biases: &[Var<'a, f32, ScalarOps>; NUM_NEIGHBOURS],
    seed: [Var<'a, f32, ScalarOps>; NUM_NEIGHBOURS],
) -> [Var<'a, f32, ScalarOps>; NUM_NEIGHBOURS] {
    let mut s: [Var<f32, ScalarOps>; NUM_NEIGHBOURS] = seed;

    let mut sums: [Var<f32, ScalarOps>; NUM_NEIGHBOURS] = from_fn(|_| tp.var(0.));
    for i in 0..NUM_NEIGHBOURS {
        let mut sum = tp.var(0.0);
        for j in 0..NUM_NEIGHBOURS {
            sum = &sum + &weights[i][j];
        }

        sums[i] = sum;
    }

    for _ in 0..NUM_ITER {
        let mut new_s = from_fn(|_| tp.var(0.));

        // Compute sum of incoming weights
        for i in 0..NUM_NEIGHBOURS {
            // Aggregate
            for j in 0..NUM_NEIGHBOURS {
                new_s[i] += &(&weights[j][i] * &sums[j].powf(&tp.var(-1.0))) * &s[j];
            }
            // Update
            new_s[i] = (&(&biases[i] * &s[i]) + &new_s[i]).tanh();
        }

        s = new_s;
    }

    println!(
        "end: [{}]",
        s.iter()
            .map(|v| format!("{:>9.4}", v.data.clone()))
            .collect::<String>()
    );
    println!("sum: {}", s.iter().map(|x| x.data).sum::<f32>());

    s
}

pub fn run_job() {
    let tp = Tape::<f32, ScalarOps>::new();
    // From hubs to authorities
    let mut weights: [[Var<f32, ScalarOps>; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [
            tp.var(0.),
            tp.var(0.),
            tp.var(0.1),
            tp.var(0.4),
            tp.var(0.1),
        ],
        [tp.var(0.), tp.var(0.), tp.var(0.), tp.var(0.4), tp.var(0.)],
        [
            tp.var(0.6),
            tp.var(0.),
            tp.var(0.),
            tp.var(0.6),
            tp.var(0.3),
        ],
        [tp.var(0.), tp.var(0.), tp.var(0.), tp.var(0.), tp.var(0.8)],
        [tp.var(0.), tp.var(0.2), tp.var(0.9), tp.var(0.), tp.var(0.)],
    ];
    let mut biases = [tp.var(0.), tp.var(0.), tp.var(0.), tp.var(0.1), tp.var(0.5)];
    let labels = vec![(0, tp.var(0.0)), (3, tp.var(0.2)), (4, tp.var(0.3))];
    let mut lr = 0.1;

    // ------------------------------------------------------------------

    let mut rng = thread_rng();

    for i in 0..10000 {
        let seed = [
            tp.var(rng.gen_range(0.0..1.0)),
            tp.var(rng.gen_range(0.0..1.0)),
            tp.var(rng.gen_range(0.0..1.0)),
            tp.var(rng.gen_range(0.0..1.0)),
            tp.var(rng.gen_range(0.0..1.0)),
        ];
        let res = positive_run::<10>(&tp, &weights, &biases, seed);

        let mut error = tp.var(0.0);
        for (index, label) in &labels {
            error += (label - &res[*index]).powf(&tp.var(2.0));
        }

        if i % 1000 == 0 {
            println!("curr_err: {}", error.data);
        }

        error.reverse();

        for i in 0..NUM_NEIGHBOURS {
            for j in 0..NUM_NEIGHBOURS {
                weights[i][j] = tp.var(weights[i][j].data - weights[i][j].grad() * lr);
            }
            biases[i] = tp.var(biases[i].data - biases[i].grad() * lr);
        }

        lr = 0.9 * lr;
    }
}
