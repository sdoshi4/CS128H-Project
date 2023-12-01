use std::collections::HashSet;
use std::ops::Add;
use std::ops::Mul;
use petgraph::Graph;
use petgraph::dot::{Dot, Config};
use rand_distr::num_traits::Pow;


	pub struct Value<'a>{
		data_: f64,
		gradient_: f64,
		prev_: Vec<Option<&'a mut Value<'a>>>,
		op_: char,
		//backwards_: Box<dyn FnMut()>,
	}
	
	impl Value<'_>{
		pub fn new(data: f64) -> Value<'static>{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: Vec::new(),
				op_: '_',
				//backwards_: Box::new(|| {}),
			}
		}
		fn new_out<'a>(data: f64, prev: Vec<Option<&'a mut Value<'a>>>, op: char) -> Value<'a>{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: prev,
				op_: op,
				//backwards_: Box::new(|| {}),
			}
		}

		pub fn add(v1: &mut Value<'_>, v2: &mut Value<'_>) -> Value<'static>{
			let d = v1.data_ + v2.data_;
			let mut set: Vec<Option<&mut Value>> = Vec::new();
			set.push(Some(v1));
			set.push(Some(v2));
			

			let mut out: Value<'_> =  Value::new_out(d, set,'+'); 

			// let bkwd_closure = Box::new(|| {
			// 	v1.gradient_ = v1.gradient_ + out.gradient_;
			// 	v2.gradient_ = v2.gradient_ + out.gradient_;
			// });
			

			//out.backwards_ = bkwd_closure;

			return out;
		}

		pub fn mul<'a>(v1: &mut Value<'_>, v2: &mut Value<'_>) -> Value<'a>{
			let d = v1.data_ * v2.data_;
			let mut set: Vec<Option<&mut Value>> = Vec::new();
			set.push(Some(v1));
			set.push(Some(v2));

			let mut out:Value<'_> =  Value::new_out(d, set,'*'); 

			let bkwd_closure = Box::new(|| {
				v1.gradient_ = v1.gradient_ * out.gradient_;
				v2.gradient_ = v2.gradient_ * out.gradient_;
			});
			//out.backwards_ = bkwd_closure;

			return out;
		}

		pub fn tanh<'a>(v1: &mut Value<'_>) -> Value<'a>{
			let x = v1.data_;
			let t=(f64::exp(2.0*x)-1.0)/(f64::exp(2.0*x)+1.0);
			let mut set: Vec<Option<&mut Value>> = Vec::new();
			set.push(Some(v1));
			let mut out =  Value::new_out(t, set,'t');

			let bkwd_closure = Box::new(|| {
				v1.gradient_ = (1.0 - t.pow(2.0)) * out.gradient_;
			});
			//out.backwards_ = bkwd_closure;

			return out;
		}

		pub fn back_prop(self){
			match self.op_ {

				'+' => self.add_back(),
				'*' => self.mul_back(),
				't' => self.tanh_back(), 
				_ => {},
			}
			todo!();
		}

		fn tanh_back(self){

		}
		fn mul_back(self){

		}
		fn add_back(self){
			unsafe{
				// if let Some(prev) = self.prev_.get_mut(1) {
				// 	if let Some(value) = prev.as_mut() {
				// 		value.gradient_ += self.gradient_;
				// 	}
				// }
					
				//(**(self.prev_.get(1).unwrap())).unwrap().gradient_ += self.gradient_;
			}
		}


	}
	
	
	impl ToString for Value<'_>{
		fn to_string(&self) -> String{
			format!("Value(data:{}, gradient:{})", self.data_, self.gradient_)
		}
	}
	



	// impl Add for Value{
	// 	type Output = Value;

	// 	fn add(self, other: Value) -> Value{
	// 		let d = self.data_ + other.data_;
	// 		let mut set: Vec<*const Option<Value>> = Vec::new();
	// 		set.push(&Some(self));
	// 		set.push(&Some(other));
	// 		let mut out =  Value::new_out(d, set,'+'); 

	// 		let bkwd_closure = Box::new(|| {
	// 			self.gradient_ = self.gradient_ + out.gradient_;
	// 			other.gradient_ = self.gradient_ + out.gradient_;
	// 		});
	// 		out.backwards_ = bkwd_closure;

	// 		return out;
	// 	}
	// }
	
	// impl Mul for Value{
	// 	type Output = Value;
		
	// 	fn mul(self, other: Value) -> Value{
	// 		let d = self.data_ * other.data_;
	// 		let mut set: Vec<*const Option<Value>> = Vec::new();
	// 		set.push(&Some(self));
	// 		set.push(&Some(other));
	// 		Value{
	// 			data_: d,
	// 			gradient_: 0.0,
	// 			prev_:set,
	// 			op_:'*',
	// 		}
	// 	}
	// }

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

