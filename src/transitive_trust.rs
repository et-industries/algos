use std::{
    collections::{HashMap, HashSet},
    vec::IntoIter,
};

use priority_queue::PriorityQueue;

struct Result {
    node: String,
    p_score: i32,
    n_score: i32,
}

impl Result {
    pub fn new(node: String, p_score: i32, n_score: i32) -> Self {
        Self {
            node,
            p_score,
            n_score,
        }
    }

    pub fn net_score(&self) -> i32 {
        self.p_score - self.n_score
    }
}

#[derive(Debug, Clone)]
struct Node {
    positive_edges: HashMap<String, u16>,
    negative_edges: HashMap<String, u16>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            positive_edges: HashMap::new(),
            negative_edges: HashMap::new(),
        }
    }

    pub fn add_positive_edge(&mut self, target: String, weight: u16) {
        self.positive_edges.insert(target, weight);
    }

    pub fn add_negative_edge(&mut self, target: String, weight: u16) {
        self.negative_edges.insert(target, weight);
    }

    pub fn get_positive_weight(&self, target: String) -> u16 {
        self.positive_edges.get(&target).cloned().unwrap()
    }

    pub fn get_negative_weight(&self, target: String) -> u16 {
        self.negative_edges.get(&target).cloned().unwrap()
    }

    pub fn out_neighbours(&self) -> Vec<String> {
        let mut positive_keys: HashSet<&String> = self.positive_edges.keys().collect();
        let negative_keys: HashSet<&String> = self.negative_edges.keys().collect();
        positive_keys.extend(negative_keys);
        positive_keys.into_iter().cloned().collect()
    }
}

#[derive(Debug, Clone)]
struct Graph {
    nodes: HashMap<String, Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn add_positive_edge(&mut self, source: String, target: String, weight: u16) {
        if !self.nodes.contains_key(&source) {
            self.nodes.insert(source.clone(), Node::new());
        }
        if !self.nodes.contains_key(&target) {
            self.nodes.insert(target.clone(), Node::new());
        }

        let node = self.nodes.get_mut(&source).unwrap();
        node.add_positive_edge(target, weight);
    }

    pub fn add_negative_edge(&mut self, source: String, target: String, weight: u16) {
        if !self.nodes.contains_key(&source) {
            self.nodes.insert(source.clone(), Node::new());
        }
        if !self.nodes.contains_key(&target) {
            self.nodes.insert(target.clone(), Node::new());
        }

        let node = self.nodes.get_mut(&source).unwrap();
        node.add_negative_edge(target, weight);
    }

    pub fn get_positive_weight(&self, source: String, target: String) -> u16 {
        let node = self.nodes.get(&source).unwrap();
        node.get_positive_weight(target)
    }

    pub fn get_negative_weight(&self, source: String, target: String) -> u16 {
        let node = self.nodes.get(&source).unwrap();
        node.get_negative_weight(target)
    }

    pub fn for_each_node(&self) -> IntoIter<String> {
        let keys: Vec<String> = self.nodes.keys().cloned().collect();
        keys.into_iter()
    }

    pub fn for_each_neighbour(&self, node: String) -> IntoIter<String> {
        let node = self.nodes.get(&node).unwrap();
        node.out_neighbours().into_iter()
    }
}

pub fn compute_scores(graph: Graph, source: String) -> Vec<Result> {
    let mut p_scores = HashMap::<String, i32>::new();
    let mut n_scores = HashMap::<String, i32>::new();
    let mut inspected = HashSet::<String>::new();
    let mut pq = PriorityQueue::<String, i32>::new();

    for node in graph.for_each_node() {
        let p_score = if node == source { 1 } else { 0 };
        p_scores.insert(node.clone(), p_score);
        n_scores.insert(node.clone(), 0);
        pq.push(node, p_score);
    }

    while !pq.is_empty() {
        let (node_key, _) = pq.pop().unwrap();
        if inspected.contains(&node_key) {
            continue;
        }
        inspected.insert(node_key.clone());

        let node_score =
            (p_scores.get(&node_key).unwrap() - n_scores.get(&node_key).unwrap()).max(0);

        for neighbor_key in graph.for_each_neighbour(node_key.clone()) {
            let neighbor_score =
                p_scores.get(&neighbor_key).unwrap() - n_scores.get(&neighbor_key).unwrap();

            if inspected.contains(&neighbor_key) || neighbor_score < node_score {
                continue;
            }

            let positive_weight = graph.get_positive_weight(node_key.clone(), neighbor_key.clone());
            let negative_weight = graph.get_negative_weight(node_key.clone(), neighbor_key.clone());

            let neighbour_p_score = p_scores.get(&neighbor_key).unwrap();
            let neighbour_n_score = n_scores.get(&neighbor_key).unwrap();

            if node_score > *neighbour_p_score {
                let new_neighbour_p_score = neighbour_p_score
                    + (node_score - neighbour_p_score) * i32::from(positive_weight);
                p_scores.insert(neighbor_key.clone(), new_neighbour_p_score);
            };

            if node_score > *neighbour_n_score {
                let new_neighbour_n_score = neighbour_n_score
                    + (node_score - neighbour_n_score) * i32::from(negative_weight);
                n_scores.insert(neighbor_key.clone(), new_neighbour_n_score);
            };

            let neighbour_p_score = p_scores.get(&neighbor_key).unwrap();
            let neighbour_n_score = n_scores.get(&neighbor_key).unwrap();
            pq.push(neighbor_key, neighbour_p_score - neighbour_n_score);
        }
    }

    let mut results = Vec::new();
    for node in graph.for_each_node() {
        if node != source {
            let p_score = p_scores.get(&node).unwrap();
            let n_score = n_scores.get(&node).unwrap();
            let result = Result::new(node, *p_score, *n_score);
            results.push(result);
        }
    }

    results
}

pub fn run_job() {}
