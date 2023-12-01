use std::collections::HashSet;
use std::ops::Add;
use std::ops::Mul;
use petgraph::Graph;
use petgraph::dot::{Dot, Config};
use petgraph::visit;
use rand_distr::num_traits::Pow;


	pub struct Value{
		data_: f64,
		gradient_: f64,
		pub prev_: Vec<Value>,
		pub op_: char,
	}
	
	impl Value{
		pub fn new(data: f64) -> Value{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: Vec::new(),
				op_: '_',
			}
		}
		fn new_out(data: f64, prev: Vec<Value>, op: char) -> Value{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: prev,
				op_: op,
			}
		}

		pub fn add(v1: Value, v2: Value) -> Value{
			let d = v1.data_ + v2.data_;
			let mut set: Vec<Value> = Vec::new();
			set.push(v1);
			set.push(v2);
			let mut out=  Value::new_out(d, set,'+'); 
			return out;
		}

		pub fn  mul(v1: Value, v2: Value) -> Value{
			let d = v1.data_ * v2.data_;
			let mut set: Vec<Value> = Vec::new();
			set.push(v1);
			set.push(v2);

			let mut out =  Value::new_out(d, set,'*'); 

			return out;
		}

		pub fn tanh(v1: Value) -> Value{
			let x = v1.data_;
			let t=(f64::exp(2.0*x)-1.0)/(f64::exp(2.0*x)+1.0);
			let mut set: Vec<Value> = Vec::new();
			set.push(v1);
			let mut out =  Value::new_out(t, set,'t');
			return out;
		}

		pub fn backwards(&mut self){
			let topo_order_rev: &mut Vec<*mut Value> = &mut Vec::new();
			let mut visited: HashSet<*mut Value> = HashSet::new();
			let self_address:*mut Value = self;

			self.build_topo(&mut visited, self_address, topo_order_rev);
			self.gradient_ = 1.0;
			let topo_order: Vec<_> = topo_order_rev.iter().rev().cloned().collect();

			print!("topo_order size:{}", topo_order_rev.len());
			for v_address in topo_order{
				unsafe {
					(*(v_address)).single_back_prop();
				}
			}
		}

		fn build_topo(&mut self, visited: &mut HashSet<*mut Value>, self_address: *mut Value, topo:&mut Vec<*mut Value>){
			

			if !visited.contains(&self_address){
				visited.insert(self_address);
				for child in (&mut self.prev_){
					
					let mut child_adress:*mut Value = child;
					child.build_topo(visited, child_adress, topo);
				}
				topo.push(self_address);
				//println!("topo map length after{}", topo.len());

			}
		}

		fn single_back_prop(&mut self){
			print!("single_back_prop{}", self.op_);
			match self.op_ {
				'+' => self.add_back(),
				'*' => self.mul_back(),
				't' => self.tanh_back(), 
				_ => {},
			}
		}

		fn tanh_back(&mut self){
			let x = self.prev_.get_mut(0).unwrap().data_;
			let t=(f64::exp(2.0*x)-1.0)/(f64::exp(2.0*x)+1.0);
			if let Some(ref mut val) = self.prev_.get_mut(0){
				val.gradient_ =  (1.0 - t.pow(2.0)) * self.gradient_;
			}
		}
		fn mul_back(&mut self){
			print!("Mul_back");
			let mut v1_data = 0.0;
			let mut v2_data = 0.0;

			if let Some(ref mut val) = self.prev_.get_mut(0){
				v1_data =  val.data_;
			}
			if let Some(ref mut val) = self.prev_.get_mut(1){
				v2_data = val.data_;
			}
			println!("v1_grad:{}", v1_data);
			println!("v2_grad:{}", v2_data);
			if let Some(ref mut val) = self.prev_.get_mut(0){
				val.gradient_ =  v2_data * self.gradient_;
			}
			if let Some(ref mut val) = self.prev_.get_mut(1){
				val.gradient_ = v1_data * self.gradient_;
			}
		}
		fn add_back(&mut self){
			if let Some(ref mut val) = self.prev_.get_mut(0){
				val.gradient_ += self.gradient_;
			}
			if let Some(ref mut val) = self.prev_.get_mut(1){
				val.gradient_ += self.gradient_;
			}
		}

	}
	
	
	impl ToString for Value{
		fn to_string(&self) -> String{
			format!("Value(data:{}, gradient:{}, operation{})", self.data_, self.gradient_, self.op_)
		}
	}

	

	// pub fn trace(root: &Value) -> (HashSet<Value>, HashSet<Value>){
	// 	let mut nodes: HashSet<&Option<Value>> = HashSet::new();
	// 	let mut edges:HashSet<&Option<Value>> = HashSet::new();
		
		
	// 	todo!();


	// }

	// fn build(val: *const Value, nodes: &mut HashSet<*const Option<Value>>, edges: &mut HashSet<*const Option<Value>>) {
	// 	if !nodes.contains(&val){
	// 		nodes.insert(val);

	// 		unsafe{
	// 			for child    in &((*val).prev_){
	// 				edges.insert(child.clone());
	// 			}
	// 		}
			
	// 	}
	// }

