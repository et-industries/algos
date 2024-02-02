const NUM_NEIGHBOURS: usize = 5;
const NUM_ITER: usize = 30;
const PRE_TRUST_WEIGHT: f32 = 0.3;

fn validate_lt(lt: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS]) {
    // Compute sum of incoming distrust
    for i in 0..NUM_NEIGHBOURS {
        for j in 0..NUM_NEIGHBOURS {
            // Make sure we are not giving score to ourselves
            if i == j {
                assert_eq!(lt[i][j], 0.);
            }
            assert!(lt[i][j] >= 0.);
        }
    }
}

fn normalise(
    lt_vec: [f32; NUM_NEIGHBOURS],
    pre_trust: [f32; NUM_NEIGHBOURS],
) -> [f32; NUM_NEIGHBOURS] {
    let sum: f32 = lt_vec.iter().sum();
    if sum == 0. {
        return pre_trust;
    }
    lt_vec.map(|x| x / sum)
}

fn vec_scalar_mul(s: [f32; NUM_NEIGHBOURS], y: f32) -> [f32; NUM_NEIGHBOURS] {
    s.map(|x| x * y)
}

fn vec_mul(s: [f32; NUM_NEIGHBOURS], y: [f32; NUM_NEIGHBOURS]) -> [f32; NUM_NEIGHBOURS] {
    let mut out: [f32; NUM_NEIGHBOURS] = [0.; NUM_NEIGHBOURS];
    for i in 0..NUM_NEIGHBOURS {
        out[i] = s[i] + y[i];
    }
    out
}

fn transpose(
    s: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
) -> [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] {
    let mut new_s: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [[0.; NUM_NEIGHBOURS]; NUM_NEIGHBOURS];
    for i in 0..NUM_NEIGHBOURS {
        for j in 0..NUM_NEIGHBOURS {
            new_s[i][j] = s[j][i];
        }
    }
    new_s
}

fn positive_run(
    domain: String,
    mut lt: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    pre_trust: [f32; NUM_NEIGHBOURS],
) -> [f32; NUM_NEIGHBOURS] {
    println!("");
    println!("{} - Trust:", domain);

    validate_lt(lt);
    for i in 0..NUM_NEIGHBOURS {
        lt[i] = normalise(lt[i], pre_trust);
    }

    let mut s = pre_trust.clone();
    let pre_trusted_scores = pre_trust.map(|x| x * PRE_TRUST_WEIGHT);

    println!("start: [{}]", s.map(|v| format!("{:>9.4}", v)).join(", "));
    for _ in 0..NUM_ITER {
        let mut new_s = [0.; 5];

        // Compute sum of incoming weights
        for i in 0..NUM_NEIGHBOURS {
            for j in 0..NUM_NEIGHBOURS {
                new_s[i] += lt[j][i] * s[j];
            }
        }

        let global_scores = new_s.map(|x| (1. - PRE_TRUST_WEIGHT) * x);
        let current_s = vec_mul(pre_trusted_scores, global_scores);

        s = current_s;
    }
    println!("end: [{}]", s.map(|v| format!("{:>9.4}", v)).join(", "));

    s
}

fn negative_run(
    domain: String,
    mut lt: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    s: [f32; NUM_NEIGHBOURS],
    pre_trust: [f32; NUM_NEIGHBOURS],
) -> [f32; NUM_NEIGHBOURS] {
    println!("");
    println!("{} - Distrust:", domain);

    validate_lt(lt);
    for i in 0..NUM_NEIGHBOURS {
        lt[i] = normalise(lt[i], pre_trust);
    }

    let mut new_s = [0.0; NUM_NEIGHBOURS];
    // Compute sum of incoming weights
    for i in 0..NUM_NEIGHBOURS {
        for j in 0..NUM_NEIGHBOURS {
            new_s[i] += lt[j][i] * s[j];
        }
    }

    println!("end: [{}]", new_s.map(|v| format!("{:>9.4}", v)).join(", "));
    new_s
}

fn functional_case() {
    let pre_trust: [f32; NUM_NEIGHBOURS] = [0.0, 0.0, 0.0, 0.7, 0.3];

    let lt_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 1.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [11., 0.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 10., 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];

    let ss_s = positive_run("Software Security".to_string(), lt_ss, pre_trust);

    let ld_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 10., 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [0.0, 1.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [10., 0.0, 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];

    negative_run("Software Security".to_string(), ld_ss, pre_trust, ss_s);

    let lt_sd: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 1.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [1.0, 0.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];

    let sd_s = positive_run("Software Development".to_string(), lt_sd, pre_trust);

    let ld_sd: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [0.0, 1.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];

    negative_run("Software Development".to_string(), ld_sd, pre_trust, sd_s);

    let snap1_trust: [f32; NUM_NEIGHBOURS] = [50., 0., 0., 50., 0.];
    let snap1_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 0., 0.];

    let snap2_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 50.];
    let snap2_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 50., 0.];

    let num1: f32 = snap1_trust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let den1: f32 = snap1_distrust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let snap1_score: f32 = num1 / num1 + den1;

    let num2: f32 = snap2_trust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let den2: f32 = snap2_distrust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let snap2_score: f32 = num2 / num2 + den2;

    println!("");
    println!("snap1 score: {}", snap1_score);
    println!("snap2 score: {}", snap2_score);
}

fn sybil_case() {
    let pre_trust: [f32; NUM_NEIGHBOURS] = [0.0, 0.0, 0.0, 0.7, 0.3];

    let lt_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 10.0, 10.0, 0.0, 0.0], // - Peer 0 opinions
        [10.0, 0.0, 10.0, 0.0, 0.0], // - Peer 1 opinions
        [10.0, 10.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 3 opinions
        [0.0, 10., 0.0, 0.0, 0.0],   // = Peer 4 opinions
    ];

    let ss_s = positive_run("Software Security".to_string(), lt_ss, pre_trust);

    let ld_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 10.0, 0.0],   // - Peer 0 opinions
        [0.0, 0.0, 0.0, 10.0, 0.0],   // - Peer 1 opinions
        [0.0, 0.0, 0.0, 10.0, 0.0],   // - Peer 2 opinions
        [10.0, 10.0, 10.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],    // = Peer 4 opinions
    ];

    negative_run("Software Security".to_string(), ld_ss, pre_trust, ss_s);

    let snap1_trust: [f32; NUM_NEIGHBOURS] = [50., 50., 50., 0., 0.];
    let snap1_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 50., 50.];

    let snap2_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 50., 50.];
    let snap2_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 0.];

    let num1: f32 = snap1_trust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let den1: f32 = snap1_distrust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let snap1_score: f32 = num1 / num1 + den1;

    let num2: f32 = snap2_trust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let den2: f32 = snap2_distrust
        .iter()
        .zip(ss_s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let snap2_score: f32 = num2 / num2 + den2;

    println!("");
    println!("snap1 score: {}", snap1_score);
    println!("snap2 score: {}", snap2_score);
}

fn main() {
    functional_case();
}
