use std::vec::Vec;

use crate::NeuralNetwork::Connector;


// pub use crate::Connector::Connector;

pub struct Node {
    connectors: Vec<crate::NeuralNetwork::Connector::Connector>,
    total: f64
}

impl Node{
    fn Reset() {
        total = 0;
    }

    fn AddData(value: f64, weight: f64) {
        // default value passed into weight should be 1
        total += value * weight;
    }

    fn Activation() {
        total = Sigmoid(total);
    }

    fn FeedForward() {
        for connector in connectors {
            connector.Node.AddData(total, connector.weight);
        }
    }
}

pub fn Sigmoid(x: f64) -> f64 {
    1 / (1 + Math.Exp(-x))
}