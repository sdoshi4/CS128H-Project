use std::vec::Vec;

// pub mod Connector;
use super::Connector::*;


pub use crate::Connector::Connector;

struct Node {
    connectors: Vec<Connector>,
    value: f64
}



mod Connector;

