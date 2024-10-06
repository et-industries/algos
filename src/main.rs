mod eigen_trust;
mod gnn;
mod hubs_and_auth;
mod page_rank;
mod transitive_trust;
mod utils;

fn main() {
    transitive_trust::run_job();
}
