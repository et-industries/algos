use std::collections::HashMap;

fn trust_policy(weights: &HashMap<(u32, u32), f32>) -> bool {
    if weights.len() > 5 {
        return false;
    }
    for (_, v) in weights {
        if *v > 0.8 || *v <= 0.0 {
            return false;
        }
    }
    return true;
}

fn h(inspected: &HashMap<u32, bool>, scores: &HashMap<u32, f32>) -> HashMap<u32, Vec<u32>> {
    let mut sets = HashMap::new();
    for (i, v_i) in scores {
        let ins_i = inspected.get(&i).unwrap_or(&false);
        let mut set_i = Vec::new();
        for (j, v_j) in scores {
            let ins_j = inspected.get(&i).unwrap_or(&false);
            if *ins_i && *ins_j && v_i > v_j {
                set_i.push(*j);
            }
        }
        sets.insert(*i, set_i);
    }
    sets
}

fn all_inspected(inspected: &HashMap<u32, bool>) -> bool {
    for (_, ins) in inspected {
        if !ins {
            return false;
        }
    }

    return true;
}

fn run(
    mut inspected: HashMap<u32, bool>,
    mut scores: HashMap<u32, f32>,
    weights: HashMap<(u32, u32), f32>,
) -> HashMap<u32, f32> {
    assert!(trust_policy(&weights));
    while !all_inspected(&inspected) {
        let sets = h(&inspected, &scores);
        for (i, set_i) in sets {
            inspected.insert(i, true); // mark as inspected
            for j in set_i {
                if !inspected.get(&i).unwrap_or(&false) {
                    let weight_ij = weights.get(&(i, j)).unwrap_or(&0.0);
                    let score_j = scores.get(&j).unwrap_or(&0.0);
                    let score_i = scores.get(&i).unwrap_or(&0.0);
                    let new_score_j = score_j + (score_i - score_j) * weight_ij;
                    scores.insert(j, new_score_j.max(0.0));
                }
            }
        }
    }
    scores
}

pub fn run_job() {
    let nodes = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let inspected = nodes.clone().into_iter().map(|x| (x, false)).collect();
    let mut scores: HashMap<u32, f32> = nodes.into_iter().map(|x| (x, 0.0)).collect();
    scores.insert(1, 1.0);
    let mut weights = HashMap::new();
    weights.insert((1, 2), 0.2);
    weights.insert((3, 4), 0.6);
    weights.insert((4, 2), 0.3);
    weights.insert((6, 4), 0.2);
    weights.insert((10, 6), 0.6);
    let final_scores = run(inspected, scores, weights);
    for (i, score) in final_scores {
        println!("{}: {}", i, score);
    }
}
