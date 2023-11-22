use std::vec::Vec;

use crate::NeuralNetwork::Connector;


// pub use crate::Connector::Connector;

pub struct Node {
    connectors_: Vec<crate::NeuralNetwork::Connector::Connector>,
    total_: f64
}

impl Node{
    fn reset(&self) {
        self.total_ = 0.0;
    }

    fn add_data(&self, value: f64, weight: f64) {
        // default value passed into weight should be 1
        self.total_ += value * weight;
    }

    fn activation(&self) {
        self.total_ = sigmoid(self.total_);
    }

    fn feed_forward(&self) {
        for connector in self.connectors_ {
            connector.get_node().add_data(self.total_, connector.get_weight());
        }
    }
}


pub fn sigmoid(x: f64) -> f64 {
    1.0 / ((1.0 + f64::exp(-x)))
}