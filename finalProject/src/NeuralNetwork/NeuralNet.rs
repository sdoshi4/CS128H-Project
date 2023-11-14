use crate::NeuralNetwork::Node;
use crate::NeuralNetwork::Connector;
use rand::prelude::*;
use rand::distributions::{Distribution, Normal};

pub struct MultiLayerNeuralNet{
	input_nodes: Vec<crate::NeuralNetwork::Node::Node>,
	output_nodes: Vec<crate::NeuralNetwork::Node::Node>,
	connectors: Vec<crate::NeuralNetwork::Connector::Connector>,
	num_layers: i32,
	node_per_layer: i32,
}


impl MultiLayerNeuralNet{


	fn getRandomGaussian(mean : f64, std_dev : f64) -> f64{
		let mut rng = rand::thread_rng();
		let normal_value = Normal::new(mean, std_dev);
		return normal_value.sample(&mut rng);
	}


	fn ChangeWeights(std_dev : f64) -> f64{
		for connector in connectors{
			connector.weight_ = connector.weight_ *getRandomGaussian(0, std_dev);
		}
		todo!()
	}


	// trains the connector weights
	fn TrainModel(){
		todo!();
	}

	//initializes and connects perceptron layers to input and output
	fn generate_layers(num_layers:i32, node_p_layer:i32){
		num_layers = num_layers;
		node_per_layer = node_p_layer;

		for i in num_layers{
			
			for j in node_per_layer{


			}

		}

	}

	fn PredictData(){
		
	}













}