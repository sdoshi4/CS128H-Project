use crate::NeuralNetwork::Node;

pub struct Connector{
	weight_: f64,
	node_: crate::NeuralNetwork::Node::Node,
}

impl Connector{
	pub fn get_weight(&self) -> f64{
		self.weight_
	}
	pub fn set_weight(&self, weight : f64){
		self.weight_ = weight; 
	}

	pub fn get_node(&self) -> Node::Node{
		self.node_
	}

	pub fn set_node(&self, node : Node::Node){
		self.node_ = node;
	}


}