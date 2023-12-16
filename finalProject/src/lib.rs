
use ndarray::Axis;
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use num::abs;



#[derive(Debug, Eq, PartialEq)]
pub enum ActivationFunction {
    Relu,
    Sigmoid,
    Softmax,
	Tanh,
}


pub struct Neuron {
    pub weights: Vec<Rc<RefCell<Value>>>,
    pub bias: Rc<RefCell<Value>>,
}

impl Neuron {
    pub fn new(prev_layer_size: usize) -> Self {
		let rand_array_f64= Array::random((1, prev_layer_size), Normal::new(0.0, 1.0).unwrap());
		let mut rand_array_value:Vec::<Rc<RefCell<Value>>> = Vec::new();

		for row in 0..rand_array_f64.len(){
			rand_array_value.push(Rc::new(RefCell::new(Value::new(rand_array_f64.clone().remove_axis(Axis(0))[row]))));
		}

        let weights: Vec<Rc<RefCell<Value>>> = rand_array_value;
        let bias = Rc::new(RefCell::new(Value::new(Array::random((1, 1), Normal::new(0.0, 1.0).unwrap()).row(0).to_vec()[0])));
        Neuron {weights, bias }
    }
    pub fn print_nueron(self){
        println!("Neuron data:");
        print!("weights");
        for row in 0..self.weights.len() {
            print!("{} ", self.weights[row].borrow().to_string());
            println!();
        }
        println!("biases{}", self.bias.borrow().to_string())
    }
}


pub struct Layer {
    // pub weights: Vec<f64>,
    // pub biases: Vec<f64>,
    pub neurons: Vec<Neuron>,
    // pub outputs: Option<Vec<Value>>,
}

impl Layer {
    pub fn new(layer_size: usize, prev_layer_size: usize) -> Self {
        // let weights = Array::random((prev_layer_size, layer_size), Normal::new(0.0, 1.0).unwrap());
        // let biases = Array::random((1, layer_size), Normal::new(0.0, 1.0).unwrap());
        let mut neurons = Vec::new();
        
        for i in 0..layer_size {
            let new_neuron = Neuron::new(prev_layer_size);
            neurons.push(new_neuron);
        }

        // Layer {weights, biases, outputs}
        Layer { neurons }
    }

	


    pub fn forward(&mut self, inputs: &mut Vec<Rc<RefCell<Value>>>, afunc: ActivationFunction) -> Vec<Rc<RefCell<Value>>> {
        let mut outp: Vec<Rc<RefCell<Value>>>  = Vec::new();


        for i in 0..self.neurons.len() {
            // let my_speed_ptr: *mut i32 = &mut my_speed;
            // let r: f64 = inputs.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // let r: f64 = inputs.iter().zip(&self.neurons[i].weights.iter()).map(|(x, y)| x * y).sum();
            // println!("{:?}", outp);
            let z = Value::add(Util::vec_tot(&mut Util::mul_vec(inputs, &mut self.neurons[i].weights)), Rc::clone(&(self.neurons[i].bias))); //.remove_axis(ndarray::Axis(0)).to_vec()[0]
            // outp[i] = z;
           // let a = relu(array![[z]]);
           // self.outputs.unwrap().append(axis, array);

            if afunc == ActivationFunction::Relu {
                //outp[[0, i]] = z; // array![[&z]]
            } else if afunc == ActivationFunction::Sigmoid {
                //outp[[0, i]] = sigmoid(z);
            } else if afunc == ActivationFunction::Softmax {
                //self.outputs = Some(softmax(inputs.dot(&self.weights) + &self.biases));
            }else {
				outp.push(Value::tanh( z)); 
			}
            
        }
        // println!("{:?}", outp);
        // self.outputs = Some(outp);
        outp
        
    }
    pub fn print_layer(self){
        println!("layer data");
        for n in self.neurons{
            n.print_nueron();
        }
        println!("output:");
    }
}

pub struct Perceptron {
    pub layers: Vec<Layer>,
    pub outputs: Option<Vec<Rc<RefCell<Value>>>>,
    pub output: Option<usize>,
}

impl Perceptron {
    pub fn new(n_layers: usize, layer_size: usize) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        //layers.push(Layer::new(layer_size, n_inputs)); // starting layer
        for _i in 0..n_layers {
            layers.push(Layer::new(layer_size, layer_size));
        }
        //layers.push(Layer::new(n_outputs, layer_size)); // output layer

        // let weights = Array::random((n_inputs, layer_size), Normal::new(1.0, 0.0).unwrap());
        let outputs = None;
        let output = None;
        Perceptron { layers, outputs, output }
    }
    pub fn run(&mut self, inputs: &mut Vec<Rc<RefCell<Value>>>, n_outputs: usize) {
        let mlp_size = self.layers.len();

		let mut out_layer = Layer::new(n_outputs, self.layers[0].neurons.len());

        let mut next_layer = self.layers[0].forward(inputs, ActivationFunction::Tanh);

        for i in 1..mlp_size { // exclusive, so doesn't do output layer
            next_layer = self.layers[i].forward(&mut next_layer, ActivationFunction::Tanh);
        }
		
        self.outputs = Some(out_layer.forward(&mut next_layer, ActivationFunction::Tanh));

		self.find_output();

        // loss is just -log(ouput[correct_ind])
        // class_targets = [0, 1, 1]
        // for (targ_idx, distribution) in zip(class_targets, softmax_outputs):
        // print(distribution[targ_idx])
    }

    // pub fn calculate_loss_log(&mut self, correct_ind: usize) -> f64 {
    //     let outputs_clipped: Vec<f64> = self.outputs.clone().unwrap().map(|v| num::clamp(*v, 1e-7, 1.-1e-7));
    //     -outputs_clipped[[0, correct_ind]].ln()
    // }

    pub fn find_output(&mut self) {
        let mut best_ind: usize = 0;
        let mut shortest_distance: f64 = 10.0;
        let mut ind = 0;
        for i in self.outputs.clone().unwrap() {
            if abs(i.borrow().data_-1.) < shortest_distance {
                shortest_distance = abs(i.borrow().data_-1.);
                best_ind = ind;
            }
            ind+=1;
        }
        self.output = Some(best_ind);
    }

    // pub fn backpropogate(&mut self) {
    //     let alpha: f64 = 0.001;
    //     // outer layer
    //     let mut weight_derivs: Vec<f64> =
    // }
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

use rand_distr::num_traits::Pow;
//use std::cell::Ref;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::cell::RefCell;
//use std::io::WriterPanicked;
//use std::ops::BitOrAssign;
use std::rc::Rc;


	struct RcReffWrapper(Rc<RefCell<Value>>);
	impl Hash for RcReffWrapper{
		fn hash<H: Hasher>(&self, state: &mut H){
			Rc::as_ptr(&self.0).hash(state)
		}
	}
	impl PartialEq for RcReffWrapper {
		fn eq(&self, other: &RcReffWrapper) -> bool {
			Rc::as_ptr(&self.0) == Rc::as_ptr(&other.0)
		}
	}
	impl Eq for RcReffWrapper {}

	


	#[derive(Default)]
	pub struct Value{
		pub data_: f64,
		pub gradient_: f64,
		pub prev_: Vec<Rc<RefCell<Value>>>,
		pub op_: char,
	}
	
	impl PartialEq for Value{
		fn eq(&self, other: &Self) -> bool {
			(self.data_ - other.data_).abs() < std::f64::EPSILON &&
			self.gradient_ == other.gradient_ &&
			self.op_ == other.op_ &&
			self.prev_ == other.prev_
		}
	}
	impl Eq for Value{

	}



	impl Hash for Value{
		fn hash<H: Hasher>(&self, state: &mut H) {
			self.data_.to_bits().hash(state);
			self.gradient_.to_bits().hash(state);
			self.op_.hash(state);
		}
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
		fn new_out(data: f64, prev: Vec<Rc<RefCell<Value>>>, op: char) -> Value{
			Value{
				data_: data,
				gradient_: 0.0,
				prev_: prev,
				op_: op,
			}
		}
		pub fn change_grad(&self){

		}
	
		pub fn add(v1: Rc<RefCell<Value>>, v2:  Rc<RefCell<Value>>) -> Rc<RefCell<Value>>{
			let d = v1.borrow().data_ + v2.borrow().data_;
			let mut set: Vec<Rc<RefCell<Value>>> = Vec::new();
			set.push(v1);
			set.push(v2);
			let out=  Value::new_out(d, set,'+'); 
			return Rc::new(RefCell::new(out));
		}

		pub fn  mul(v1: Rc<RefCell<Value>>, v2: Rc<RefCell<Value>>) -> Rc<RefCell<Value>>{
			let d = v1.borrow().data_ * v2.borrow().data_;
			let mut set: Vec<Rc<RefCell<Value>>> = Vec::new();
			set.push(v1);
			set.push(v2);
			let out=  Value::new_out(d, set,'*'); 
			return Rc::new(RefCell::new(out));
		}

		pub fn tanh(v1: Rc<RefCell<Value>>) -> Rc<RefCell<Value>>{
			let x = v1.borrow().data_;
			let t=(f64::exp(2.0*x)-1.0)/(f64::exp(2.0*x)+1.0);
			let mut set: Vec<Rc<RefCell<Value>>> = Vec::new();
			set.push(v1);
			let out =  Value::new_out(t, set,'t');
			return Rc::new(RefCell::new(out));
		}


		pub fn backwards(val: Rc<RefCell<Value>>){

			let topo_order_rev: &mut Vec<Rc<RefCell<Value>>> = &mut Vec::new();
			let mut visited: HashSet<RcReffWrapper> = HashSet::new();

			Value::build_topo(Rc::clone(&val), topo_order_rev, &mut visited);
			val.borrow_mut().gradient_ = 1.0;

			let topo_order: Vec<_> = topo_order_rev.iter().rev().cloned().collect();

			for val in topo_order{
					val.borrow_mut().single_back_prop();
			}
		}
		fn build_topo(val: Rc<RefCell<Value>>, topo:&mut Vec<Rc<RefCell<Value>>>, visited: &mut HashSet<RcReffWrapper>){
			let wrapper = RcReffWrapper(Rc::clone(&val));
			if !visited.contains(&wrapper){
				visited.insert(wrapper);
				for child in &(val.borrow().prev_){
					Value::build_topo(Rc::clone(&child),topo, visited);
					//print!("line 311");
				}
				topo.push(Rc::clone(&val));
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
			let x = (self.prev_.get(0).unwrap()).borrow().data_;
			let t=(f64::exp(2.0*x)-1.0)/(f64::exp(2.0*x)+1.0);
			if let Some(pred) = self.prev_.get(0){
				pred.borrow_mut().gradient_ = (1.0 - t.pow(2.0)) * self.gradient_;
			}
		}
		fn mul_back(&mut self){
			// print!("Mul_back");
			let mut v1_data = 0.0;
			let mut v2_data = 0.0;
		
			if let Some(val) = self.prev_.get(0){
				v1_data =  val.borrow().data_;
			}
			if let Some(val) = self.prev_.get(1){
				v2_data = val.borrow().data_;
			}
			// println!("v1_grad:{}", v1_data);
			// println!("v2_grad:{}", v2_data);
			if let Some(pred) = self.prev_.get(0){
				pred.borrow_mut().gradient_ =  v2_data * self.gradient_;
			}
			if let Some(pred) = self.prev_.get(1){
				pred.borrow_mut().gradient_ = v1_data * self.gradient_;
			}
		}
		fn add_back(&mut self){

			if let Some(pred) = self.prev_.get(0){
				pred.borrow_mut().gradient_ += self.gradient_;
			}
			if let Some(pred) = self.prev_.get(1){
				pred.borrow_mut().gradient_ += self.gradient_;
			}
			
		}

	}
	
	
	impl ToString for Value{
		fn to_string(&self) -> String{
			format!("Value(data:{}, gradient:{}, operation{})", self.data_, self.gradient_, self.op_)
		}
	}

	pub struct Util{

	}
	impl Util{
		pub fn mul_vec(arr1: &mut Vec<Rc<RefCell<Value>>>, arr2: &mut Vec<Rc<RefCell<Value>>>) -> Vec<Rc<RefCell<Value>>> {
			if arr1.len() != arr2.len(){
				panic!("length of vectors are not right for multiplication");
			}
	
			let mut ret = Vec::new();
		
			for i in 0..arr1.len(){
				if let (Some(val1), Some(val2))  = (arr1.get(i), arr2.get(i)) {
					ret.push(Value::mul(Rc::clone(val1), Rc::clone(val2)));
				}
			}
			ret
		}
	
		pub fn add_vec(arr1: &mut Vec<Rc<RefCell<Value>>>, arr2: &mut Vec<Rc<RefCell<Value>>>) -> Vec<Rc<RefCell<Value>>> {
		
			if arr1.len() != arr2.len(){
				panic!("length of vectors are not right for addition");
			}
	
			let mut ret = Vec::new();
			
			for i in 0..arr1.len(){
				if let (Some(val1), Some(val2))  = (arr1.get(i), arr2.get(i)) {
					ret.push(Value::add(Rc::clone(val1), Rc::clone(val2)));
				}
			}
			ret
		}
	
		pub fn vec_tot(arr1: &mut Vec<Rc<RefCell<Value>>>) -> Rc<RefCell<Value>> {
	
			if arr1.len() == 0{
				panic!("no contents to add in vector");
			}
			else if arr1.len() == 1{
	
				if let Some(val1) = arr1.get(0) {
					return Value::add(Rc::clone(val1),  Rc::new(RefCell::new(Value::new(0.0))));
				}else{
					panic!();
				}
			}
			else{
				let mut ret = Rc::new(RefCell::new(Value::new(0.0)));
				for i in 0..arr1.len() {
					if let Some(val1) = arr1.get(i) {
						ret = Value::add(Rc::clone(val1), ret);
					}else{
						panic!();
					}
				}
				ret
			}
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

// pub fn sigmoid(x: f64) -> f64 {
//     // x.map(|elem| 1. / (1. + (-elem).exp()))
//     1. / (1. + (-x).exp())
// }


// pub fn relu(x: Vec<f64>) -> Vec<f64> {
//     x.map(|elem| elem.max(0.0))
// }

// pub fn print_arr2(arr: Vec<a>){
//     for row in Vec.outer_iter() {
//         // Iterate through the elements in the row
//         for &element in row {
//             // Print each element
//             print!("{} ", element);
//         }
//         // Move to the next line after printing a row
//         println!();
//     }
// }

// pub fn softmax(x: Vec<f64>) -> Vec<f64> {
//         let mut output = Vec::<f64>::zeros(x.raw_dim());
//         for (in_row, mut out_row) in x.axis_iter(Axis(0)).zip(output.axis_iter_mut(Axis(0))) {
//             let mut max = *in_row.iter().next().unwrap();
//             for col in in_row.iter() {
//                 if col > &max {
//                     max = *col;
//                 }
//             }
//             let exp = in_row.map(|x| (x-max).exp()); // used to be (x - max).exp()
//             let sum = exp.sum();
//             out_row.assign(&(exp / sum));
//         }
//         output
// }
	// fn build_topo(&mut self, visited: &mut HashSet<*mut Value>, topo:&mut Vec<*mut Value>){
			

		// 	if !visited.contains(self){
		// 		visited.insert(&mut *self);
		// 		for child in (&mut self.prev_){
					
		// 			let mut child_adress:&mut &mut Value = child;
		// 			child.build_topo(visited, topo);
		// 		}
		// 		topo.push(self);
		// 		//println!("topo map length after{}", topo.len());

		// 	}
		// }

				// pub fn backwards(&mut self){
		// 	let topo_order_rev: &mut Vec<*mut Value> = &mut Vec::new();
		// 	let mut visited: HashSet<*mut Value> = HashSet::new();
		// 	let self_address:&mut Value = self;

		// 	self.build_topo(&visited, topo_order_rev);
		// 	self.gradient_ = 1.0;
		// 	let topo_order: Vec<_> = topo_order_rev.iter().rev().collect();

		// 	// print!("topo_order size:{}", topo_order_rev.len());
		// 	for v_address in topo_order{
		// 		unsafe {
		// 			(*(*(v_address))).single_back_prop();
		// 		}
		// 	}
		// }
