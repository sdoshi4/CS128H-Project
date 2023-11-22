use crate::NeuralNetwork::Node;
use crate::NeuralNetwork::Connector;
//use rand::prelude::*;
use rand::distributions::{Distribution};
use rand_distr::{Normal};
pub struct MultiLayerNeuralNet{
	input_nodes_: Vec<Node::Node>,
	output_nodes_: Vec<Node::Node>,
	connectors_: Vec<Connector::Connector>,
	num_layers_: i32,
	node_per_layer_: i32,
}


impl MultiLayerNeuralNet{


	fn get_random_gaussian(&self, mean : f64, std_dev : f64) -> f64{
		let mut rng = rand::thread_rng();
		let normal_value = Result::expect(Normal::new(mean, std_dev), "ok");
		return normal_value.sample(&mut rng);
	}


	fn change_weights(&self, std_dev : f64) -> f64{
		for connector in self.connectors_{
			connector.set_weight(connector.get_weight() * self.get_random_gaussian(0.0, std_dev));
		}
		todo!()
	}


	// trains the connector weights
	fn train_model(){
		todo!();
	}

	//initializes and connects perceptron layers to input and output
	fn generate_layers(&self, num_layers:i32, node_p_layer:i32){
		self.num_layers_ = num_layers;
		self.node_per_layer_ = node_p_layer;

		for i in 0..num_layers{
			
			for j in 0..self.node_per_layer_{


			}

		}

	}

	fn predict_data(){
		todo!();
	}













}