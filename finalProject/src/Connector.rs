mod Node;

mod Connector{
	struct Connector{
		weight_: f64,
		node_: Box<Node>,
	}
	
	impl Connector{
		fn get_weight() -> f64{
			weight_
		}
		fn set_weight(weight : f64){
			weight_ = weight; 
		}
	
		fn get_node() -> Box<Node>{
			node_
		}
	
		fn set_node(node : Box<Node>){
			node_ = node;
		}
	
	
	}
}
