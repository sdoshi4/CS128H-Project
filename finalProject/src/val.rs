use std::collections::HashSet;
use std::ops::Add;
use std::ops::Mul;
use petgraph::Graph;
use petgraph::dot::{Dot, Config};
use petgraph::visit;
use rand_distr::num_traits::Pow;
use std::hash::{Hash, Hasher};


	#[derive(Default)]
	pub struct Value<'a>{
		pub data_: f64,
		pub gradient_: f64,
		pub prev_: Vec<*mut Value<'a>>,
		pub op_: char,
	}
	
	impl<'a> PartialEq for Value<'a> {
		fn eq(&self, other: &Self) -> bool {
			(self.data_ - other.data_).abs() < std::f64::EPSILON &&
			self.gradient_ == other.gradient_ &&
			self.op_ == other.op_ &&
			self.prev_ == other.prev_
		}
	}
	impl<'a> Eq for Value<'a>{

	}



	impl<'a> Hash for Value<'a> {
		fn hash<H: Hasher>(&self, state: &mut H) {
			self.data_.to_bits().hash(state);
			self.gradient_.to_bits().hash(state);
			self.op_.hash(state);
		}
	}
	impl Value<'_>{
		pub fn new<'a>(data: f64) -> Value<'a>{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: Vec::new(),
				op_: '_',
			}
		}
		fn new_out<'a>(data: f64, prev: Vec<*mut Value<'a>>, op: char) -> Value<'a>{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: prev,
				op_: op,
			}
		}
	
		pub fn add<'a>(v1: *mut Value<'a>, v2: *mut Value<'a>) -> Value<'a>{
			unsafe{
				let d = (*v1).data_ + (*v2).data_;
				let mut set: Vec<*mut Value> = Vec::new();
				set.push(v1);
				set.push(v2);
				let out=  Value::new_out(d, set,'+'); 
				return out;
			}
			
		}

		pub fn  mul<'a>(v1: *mut Value<'a>, v2: &'a mut Value<'a>) -> Value<'a>{
			unsafe{
				let d = (*v1).data_ * (*v2).data_;
				let mut set: Vec<*mut Value> = Vec::new();
				set.push(v1);
				set.push(v2);
				let out=  Value::new_out(d, set,'*'); 
				return out;
			}
		}

		pub fn tanh<'a>(v1: *mut Value<'a>) -> Value<'a>{
			unsafe{
				let x = (*v1).data_;
				let t=(f64::exp(2.0*x)-1.0)/(f64::exp(2.0*x)+1.0);
				let mut set: Vec<*mut Value> = Vec::new();
				set.push(v1);
				let mut out =  Value::new_out(t, set,'t');
				return out;
			}
		}

		pub fn backwards(&mut self){
			let topo_order_rev: &mut Vec<&mut Value> = &mut Vec::new();
			let mut visited: HashSet<&mut Value> = HashSet::new();
			//let self_address:&mut Value = self;

			self.build_topo(&mut visited, topo_order_rev);
			self.gradient_ = 1.0;
			let topo_order: Vec<_> = topo_order_rev.iter().rev().collect();

			// print!("topo_order size:{}", topo_order_rev.len());
			for v_address in topo_order{
				unsafe {
					(*(v_address)).single_back_prop();
				}
			}
		}

		fn build_topo(&mut self, visited: &*mut HashSet<*mut Value>, topo:&mut Vec<*mut Value>){
			

			if !visited.contains(self){
				visited.insert(self);
				for child in (&mut self.prev_){
					
					let mut child_adress:&mut &mut Value = child;
					child.build_topo(visited, topo);
				}
				topo.push(self);
				//println!("topo map length after{}", topo.len());

			}
		}

		fn single_back_prop(&mut self){
			// print!("single_back_prop{}", self.op_);
			match self.op_ {
				'+' => self.add_back(),
				'*' => self.mul_back(),
				't' => self.tanh_back(), 
				_ => {},
			}
		}

		fn tanh_back(&mut self){
			let x = self.prev_get_mutt(0).unwrap().data_;
			let t=(f64::exp(2.0*x)-1.0)/(f64::exp(2.0*x)+1.0);
			if let Some(ref mut val) = self.prev_.get_mut(0){
				val.gradient_ =  (1.0 - t.pow(2.0)) * self.gradient_;
			}
		}
		fn mul_back(&mut self){
			// print!("Mul_back");
			let mut v1_data = 0.0;
			let mut v2_data = 0.0;

			if let Some(ref mut val) = self.prev_.get_mut(0){
				v1_data =  val.data_;
			}
			if let Some(ref mut val) = self.prev_.get_mut(1){
				v2_data = val.data_;
			}
			// println!("v1_grad:{}", v1_data);
			// println!("v2_grad:{}", v2_data);
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
	
	
	impl<'a> ToString for Value<'a>{
		fn to_string(&self) -> String{
			format!("Value(data:{}, gradient:{}, operation{})", self.data_, self.gradient_, self.op_)
		}
	}

	


	pub fn mul_vec<'a>(arr1: &'a mut Vec<Value>, arr2: &'a mut Vec<Value>) -> Vec<Value<'a>> {
		if arr1.len() != arr2.len(){
			panic!("length of vectors are not right for dot product");
		}

		let mut ret = Vec::new();
		let vec1: Vec<&mut Value> = arr1.iter_mut().collect();
		let vec2: Vec<&mut Value> = arr2.iter_mut().collect();

		for i in 0..vec1.len(){
			
			if let (Some(val1), Some(val2))  = (vec1.get_mut(i), arr2.get_mut(i)) {
				ret.push(Value::mul(val1, val2));
			}
		
		}
		ret
	}

	pub fn add_vec<'a>(mut arr1: &Vec<Value>, mut arr2: &Vec<Value>) -> Vec<Value<'a>> {
    
		if arr1.len() != arr2.len(){
			panic!("length of vectors are not right for adding product");
		}

		let mut ret = Vec::new();

		for i in 0..arr1.len(){
			let val1: &mut &Value<'_> = &mut arr1.get(i).unwrap();
			let val2: &mut &Value<'_> = &mut arr2.get(i).unwrap();
			ret.push(Value::add(val1, val2));
		}

		ret

	}

	pub fn vec_tot<'a>(mut arr1: &Vec<Value>) -> Value<'a> {
		if arr1.len() == 0{
			panic!("no contents to add in vector");
		}
		else if arr1.len() == 1{
			return Value::add(&mut arr1.get(0).unwrap(),&mut Value::new(0.0));
		}
		else{
            let mut ret = Value::new(0.0);
            for i in 0..arr1.len() {
                ret = Value::add(&mut arr1.get(i).unwrap(), &mut ret);
            }
            ret
		}
	}
