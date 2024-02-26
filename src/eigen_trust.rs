const NUM_NEIGHBOURS: usize = 5;
const NUM_ITER: usize = 30;
const PRE_TRUST_WEIGHT: f32 = 0.5;

const CONFIDENCE_THRESHOLD: f32 = 0.3;
const SECURE_THRESHOLD: f32 = 0.7;

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

fn validate_lt_overlap(
    lt: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    ld: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
) {
    // Compute sum of incoming distrust
    for i in 0..NUM_NEIGHBOURS {
        for j in 0..NUM_NEIGHBOURS {
            let trust_zero = lt[i][j] == 0.0;
            let distrust_zero = ld[i][j] == 0.0;
            assert!(trust_zero || distrust_zero);
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

fn vec_add(s: [f32; NUM_NEIGHBOURS], y: [f32; NUM_NEIGHBOURS]) -> [f32; NUM_NEIGHBOURS] {
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

fn calculate_snap_score(
    st: [f32; NUM_NEIGHBOURS],
    sd: [f32; NUM_NEIGHBOURS],
    s: [f32; NUM_NEIGHBOURS],
) -> f32 {
    let num: f32 = st
        .iter()
        .zip(s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let den: f32 = sd
        .iter()
        .zip(s)
        .map(|(x, y)| if *x == 50. { y } else { 0. })
        .sum();
    let snap_score: f32 = num / (num + den);

    let score = if snap_score.is_nan() { 0. } else { snap_score };

    score
}

#[derive(Debug)]
enum State {
    Reported,
    Contested,
    Endorsed,
    Unverified,
}

fn calculate_snap_score_with_threshold(
    st: [f32; NUM_NEIGHBOURS],
    sd: [f32; NUM_NEIGHBOURS],
    s: [f32; NUM_NEIGHBOURS],
    threshold: f32,
) -> (f32, f32, State) {
    /* x: positive rating from peer to snap */
    /* y: the peer's score */
    let num: f32 = st
        .iter()
        .zip(s)
        .map(|(x, y)| {
            if y < 0.0 {
                return 0.0;
            }
            if *x == 50. {
                y
            } else {
                0.
            }
        })
        .sum();
    // num: sum of positive peer scores who said yes
    let den: f32 = sd
        .iter()
        .zip(s)
        .map(|(x, y)| {
            if y < 0.0 {
                return 0.0;
            }
            if *x == 50. {
                y
            } else {
                0.
            }
        })
        .sum();
    // num: sum of positive peer scores who said no
    let snap_score: f32 = num / (num + den);

    let score = if snap_score.is_nan() { 0. } else { snap_score };
    let confidence = num + den;

    let upper_threshold = 1. - threshold;
    let state = match (score, confidence) {
        (_, c) if c <= threshold => State::Unverified,
        (s, c) if c > threshold && s <= threshold => State::Reported,
        (s, c) if c > threshold && s > threshold && s <= upper_threshold => State::Contested,
        (s, c) if c > threshold && s > upper_threshold => State::Endorsed,
        (_, _) => State::Unverified,
    };

    (score, confidence, state)
}

// Calculate threshold
fn calculate_snap_score_threshold(pre_trust: [f32; NUM_NEIGHBOURS]) -> f32 {
    let non_zero = pre_trust
        .into_iter()
        .filter(|x| *x != 0.)
        .collect::<Vec<f32>>();

    let mut min = f32::MAX;
    non_zero.into_iter().for_each(|x| {
        if x < min {
            min = x;
        }
    });

    min
}

fn positive_run(
    domain: String,
    mut lt: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    pre_trust: [f32; NUM_NEIGHBOURS],
) -> [f32; NUM_NEIGHBOURS] {
    println!();
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
        let current_s = vec_add(pre_trusted_scores, global_scores);

        s = current_s;
    }
    println!("end: [{}]", s.map(|v| format!("{:>9.4}", v)).join(", "));

    s
}

fn negative_run(
    domain: String,
    mut lt: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS],
    s: [f32; NUM_NEIGHBOURS],
) -> [f32; NUM_NEIGHBOURS] {
    println!();
    println!("{} - Distrust:", domain);

    validate_lt(lt);
    for i in 0..NUM_NEIGHBOURS {
        lt[i] = normalise(lt[i], [0.; NUM_NEIGHBOURS]);
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

fn negative_adjustment(
    s: [f32; NUM_NEIGHBOURS],
    sd: [f32; NUM_NEIGHBOURS],
) -> [f32; NUM_NEIGHBOURS] {
    let mut adjusted = [0.0f32; NUM_NEIGHBOURS];
    for i in 0..NUM_NEIGHBOURS {
        adjusted[i] = s[i] - sd[i];
    }

    println!();
    println!("Adjusted");
    println!(
        "adjusted: [{}]",
        adjusted.map(|v| format!("{:>9.4}", v)).join(", ")
    );

    adjusted
}

fn functional_case() {
    let pre_trust: [f32; NUM_NEIGHBOURS] = [0.0, 0.0, 0.0, 0.7, 0.3];
    let snap_threshold = calculate_snap_score_threshold(pre_trust);
    println!("security_threshold: {}", snap_threshold);

    let lt_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 1.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [11., 0.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 10., 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];
    let ld_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 10., 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [0.0, 1.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [10., 0.0, 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];
    validate_lt_overlap(lt_ss, ld_ss);
    let lt_sd: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 1.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [1.0, 0.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];
    let ld_sd: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [0.0, 1.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];
    validate_lt_overlap(lt_sd, ld_sd);

    let ss_s = positive_run("Software Security".to_string(), lt_ss, pre_trust);
    let ss_ds = negative_run("Software Security".to_string(), ld_ss, ss_s);
    let ss_final = negative_adjustment(ss_s, ss_ds);

    let sd_s = positive_run("Software Development".to_string(), lt_sd, pre_trust);
    let sd_ds = negative_run("Software Development".to_string(), ld_sd, sd_s);
    let _sd_final = negative_adjustment(sd_s, sd_ds);

    let snap1_trust: [f32; NUM_NEIGHBOURS] = [50., 0., 0., 50., 0.];
    let snap1_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 0., 0.];

    let snap2_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 50.];
    let snap2_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 50., 0.];

    let (snap1_score, confidence1, state1) =
        calculate_snap_score_with_threshold(snap1_trust, snap1_distrust, ss_final, snap_threshold);
    let (snap2_score, confidence2, state2) =
        calculate_snap_score_with_threshold(snap2_trust, snap2_distrust, ss_final, snap_threshold);

    println!();
    println!(
        "snap1(malicious) score: {}, confidence: {}, state: {:?}",
        snap1_score, confidence1, state1
    );
    println!(
        "snap2(secure) score: {}, confidence: {}, state: {:?}",
        snap2_score, confidence2, state2
    );
}

fn sybil_case() {
    let pre_trust: [f32; NUM_NEIGHBOURS] = [0.0, 0.0, 0.0, 0.7, 0.3];
    let snap_threshold = calculate_snap_score_threshold(pre_trust);
    println!("security_threshold: {}", snap_threshold);

    let lt_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 10.0, 10.0, 0.0, 0.0], // - Peer 0 opinions
        [10.0, 0.0, 10.0, 0.0, 0.0], // - Peer 1 opinions
        [10.0, 10.0, 0.0, 0.0, 0.0], // - Peer 2 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 3 opinions
        [0.0, 10., 0.0, 0.0, 0.0],   // = Peer 4 opinions
    ];
    let ld_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 10.0, 0.0],   // - Peer 0 opinions
        [0.0, 0.0, 0.0, 10.0, 0.0],   // - Peer 1 opinions
        [0.0, 0.0, 0.0, 10.0, 0.0],   // - Peer 2 opinions
        [10.0, 10.0, 10.0, 0.0, 0.0], // - Peer 3 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],    // = Peer 4 opinions
    ];
    validate_lt_overlap(lt_ss, ld_ss);

    let ss_s = positive_run("Software Security".to_string(), lt_ss, pre_trust);
    let ss_ds = negative_run("Software Security".to_string(), ld_ss, ss_s);
    let ss_final = negative_adjustment(ss_s, ss_ds);

    let snap1_trust: [f32; NUM_NEIGHBOURS] = [50., 50., 50., 0., 0.];
    let snap1_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 50., 50.];

    let snap2_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 50., 50.];
    let snap2_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 0.];

    let (snap1_score, confidence1, state1) =
        calculate_snap_score_with_threshold(snap1_trust, snap1_distrust, ss_final, snap_threshold);
    let (snap2_score, confidence2, state2) =
        calculate_snap_score_with_threshold(snap2_trust, snap2_distrust, ss_final, snap_threshold);

    println!();
    println!(
        "snap1(malicious) score: {}, confidence: {}, state: {:?}",
        snap1_score, confidence1, state1
    );
    println!(
        "snap2(secure) score: {}, confidence: {}, state: {:?}",
        snap2_score, confidence2, state2
    );
}

fn sleeping_agent_case() {
    let pre_trust: [f32; NUM_NEIGHBOURS] = [0.0, 0.0, 0.0, 0.7, 0.3];
    let snap_threshold = calculate_snap_score_threshold(pre_trust);
    println!("security_threshold: {}", snap_threshold);

    let lt_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 2 opinions
        [0.0, 0.0, 10.0, 0.0, 10.0], // - Peer 3 opinions
        [0.0, 0.0, 10.0, 10.0, 0.0], // = Peer 4 opinions
    ];
    let ld_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],   // - Peer 2 opinions
        [10.0, 10.0, 0.0, 0.0, 0.0], // - Peer 3 opinions
        [10.0, 10.0, 0.0, 0.0, 0.0], // = Peer 4 opinions
    ];
    validate_lt_overlap(lt_ss, ld_ss);

    let ss_s = positive_run("Software Security".to_string(), lt_ss, pre_trust);
    let ssd_s = negative_run("Software Security".to_string(), ld_ss, ss_s);
    let ssa_s = negative_adjustment(ss_s, ssd_s);

    let snap1_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 0.];
    let snap1_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 0.];

    let snap2_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 50., 50.];
    let snap2_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 0.];

    let (snap1_score, confidence1, state1) =
        calculate_snap_score_with_threshold(snap1_trust, snap1_distrust, ssa_s, snap_threshold);
    let (snap2_score, confidence2, state2) =
        calculate_snap_score_with_threshold(snap2_trust, snap2_distrust, ssa_s, snap_threshold);

    println!();
    println!("1st Round");
    println!(
        "snap1(malicious) score: {}, confidence: {}, state: {:?}",
        snap1_score, confidence1, state1
    );
    println!(
        "snap2(secure) score: {}, confidence: {}, state: {:?}",
        snap2_score, confidence2, state2
    );

    let snap1_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 0., 0.];
    let snap1_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 0., 0.];

    let snap2_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 50., 50.];
    let snap2_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 0., 0.];

    let (snap1_score, confidence1, state1) =
        calculate_snap_score_with_threshold(snap1_trust, snap1_distrust, ssa_s, snap_threshold);
    let (snap2_score, confidence2, state2) =
        calculate_snap_score_with_threshold(snap2_trust, snap2_distrust, ssa_s, snap_threshold);

    println!();
    println!("2nd Round");
    println!(
        "snap1(malicious) score: {}, confidence: {}, state: {:?}",
        snap1_score, confidence1, state1
    );
    println!(
        "snap2(secure) score: {}, confidence: {}, state: {:?}",
        snap2_score, confidence2, state2
    );

    let lt_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],  // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],  // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],  // - Peer 2 opinions
        [0.0, 0.0, 0.0, 0.0, 10.0], // - Peer 3 opinions
        [0.0, 0.0, 0.0, 10.0, 0.0], // = Peer 4 opinions
    ];
    let ld_ss: [[f32; NUM_NEIGHBOURS]; NUM_NEIGHBOURS] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],    // - Peer 0 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],    // - Peer 1 opinions
        [0.0, 0.0, 0.0, 0.0, 0.0],    // - Peer 2 opinions
        [10.0, 10.0, 10.0, 0.0, 0.0], // - Peer 3 opinions
        [10.0, 10.0, 10.0, 0.0, 0.0], // = Peer 4 opinions
    ];
    validate_lt_overlap(lt_ss, ld_ss);

    let ss_s = positive_run("Software Security".to_string(), lt_ss, pre_trust);
    let ssd_s = negative_run("Software Security".to_string(), ld_ss, ss_s);
    let ssa_s = negative_adjustment(ss_s, ssd_s);

    let snap1_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 0., 0.];
    let snap1_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 50., 50.];

    // TODO(ek): peer 2: what does he say?
    let snap2_trust: [f32; NUM_NEIGHBOURS] = [0., 0., 0., 50., 50.];
    let snap2_distrust: [f32; NUM_NEIGHBOURS] = [0., 0., 50., 0., 0.];

    let (snap1_score, confidence1, state1) =
        calculate_snap_score_with_threshold(snap1_trust, snap1_distrust, ssa_s, snap_threshold);
    let (snap2_score, confidence2, state2) =
        calculate_snap_score_with_threshold(snap2_trust, snap2_distrust, ssa_s, snap_threshold);

    println!();
    println!("3rd Round");
    println!(
        "snap1(malicious) score: {}, confidence: {}, state: {:?}",
        snap1_score, confidence1, state1
    );
    println!(
        "snap2(secure) score: {}, confidence: {}, state: {:?}",
        snap2_score, confidence2, state2
    );
}
