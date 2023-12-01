
use ndarray::Axis;
use ndarray_rand::RandomExt;
use ndarray::ArrayView;
use num::complex::ComplexFloat;
use rand_distr::Distribution;
use rand_distr::Normal;
use num::clamp;
use ndarray::Array;
use num::abs;
// use libm::{exp, floorf, sin, sqrtf};
// struct Value {
//     data: f32,
// }

// impl Value {
//     fn add(&mut self, val: Value) {
//         self.data += val.data;
//     }
//     fn mult(&mut self, val: Value) {
//         self.data *= val.data;
//     }
// }
#[derive(Debug, Eq, PartialEq)]
pub enum ActivationFunction {
    Relu,
    Sigmoid,
    Softmax,
	Tanh,
}

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

pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
}

impl Neuron {
    pub fn new(prev_layer_size: usize) -> Self {

		let rand_array_f64= Array::random((1, prev_layer_size), Normal::new(0.0, 1.0).unwrap());
		let mut rand_array_value:Vec::<Value>= vec![Value::default(); prev_layer_size];

		for row in 0..rand_array_f64.len(){
			rand_array_value[row] = Value::new(rand_array_f64.clone()[[0,row]]);
		}

        let weights: Vec<Value> = rand_array_value;
        let bias: Value = Value::new(Array::random((1, 1), Normal::new(0.0, 1.0).unwrap()).row(0).to_vec()[0]);
        Neuron {weights, bias }
    }
    pub fn print_nueron(self){
        println!("Neuron data:");
        print!("weights");
        for row in 0..self.weights.len() {
            print!("{} ", self.weights[row].to_string());
            println!();
        }
        println!("biases{}", self.bias.to_string());

    }
}


pub struct Layer {
    // pub weights: Vec<f64>,
    // pub biases: Vec<f64>,
    pub neurons: Vec<Neuron>,
    pub outputs: Option<Vec<Value>>,
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

        let outputs = None;
        // Layer {weights, biases, outputs}
        Layer { neurons, outputs }
    }

	


    pub fn forward(&mut self, inputs: Vec<Value>, afunc: ActivationFunction) {
        let mut outp: Vec<Value>  = vec![Value::default(); self.neurons.len()];


        for i in 0..self.neurons.len() {
            // let my_speed_ptr: *mut i32 = &mut my_speed;
            // let r: f64 = inputs.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // let r: f64 = inputs.iter().zip(&self.neurons[i].weights.iter()).map(|(x, y)| x * y).sum();
            // println!("{:?}", outp);
            let z: Value = Value::add(vec_tot(mul_vec(inputs, (self.neurons[i].weights))), self.neurons[i].bias); //.remove_axis(ndarray::Axis(0)).to_vec()[0]
            outp[i] = z;
           // let a = relu(array![[z]]);
           // self.outputs.unwrap().append(axis, array);

            if afunc == ActivationFunction::Relu {
                //outp[[0, i]] = z; // array![[&z]]
            } else if afunc == ActivationFunction::Sigmoid {
                //outp[[0, i]] = sigmoid(z);
            } else if afunc == ActivationFunction::Softmax {
                //self.outputs = Some(softmax(inputs.dot(&self.weights) + &self.biases));
            }else {
				outp[i] = Value::tanh(z); 
			}
            
        }
        // println!("{:?}", outp);
        self.outputs = Some(outp);
        
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
    pub outputs: Option<Vec<Value>>,
    pub output: Option<usize>,
}

impl Perceptron {
    pub fn new(n_inputs: usize, n_outputs: usize, n_layers: usize, layer_size: usize) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        layers.push(Layer::new(layer_size, n_inputs)); // starting layer
        for i in (1..n_layers) {
            layers.push(Layer::new(layer_size, layer_size));
        }
        layers.push(Layer::new(n_outputs, layer_size)); // output layer

        // let weights = Array::random((n_inputs, layer_size), Normal::new(1.0, 0.0).unwrap());
        let outputs = None;
        let output = None;
        Perceptron { layers, outputs, output }
    }
    pub fn run(&mut self, inputs: Vec<Value>) {
        let output_layer_ind = self.layers.len()-1;
        // self.outputs = Some(inputs.dot(&self.weights));
        self.layers[0].forward(inputs, ActivationFunction::Tanh);
        for i in (1..output_layer_ind) { // exclusive, so doesn't do output layer
            let prev = self.layers[i-1].outputs.clone().unwrap();
            self.layers[i].forward(prev, ActivationFunction::Tanh);
        }
        let last_hidden = self.layers[output_layer_ind-1].outputs.clone().unwrap();
        self.layers[output_layer_ind].forward(last_hidden, ActivationFunction::Tanh);
        self.outputs = self.layers[output_layer_ind].outputs.clone();
       // self.find_output();

        // loss is just -log(ouput[correct_ind])
        // class_targets = [0, 1, 1]
        // for (targ_idx, distribution) in zip(class_targets, softmax_outputs):
        // print(distribution[targ_idx])
    }

    // pub fn calculate_loss_log(&mut self, correct_ind: usize) -> f64 {
    //     let outputs_clipped: Vec<f64> = self.outputs.clone().unwrap().map(|v| num::clamp(*v, 1e-7, 1.-1e-7));
    //     -outputs_clipped[[0, correct_ind]].ln()
    // }

    // pub fn find_output(&mut self) {
    //     let mut best_ind: usize = 0;
    //     let mut shortest_distance: f64 = 10.0;
    //     let mut ind = 0;
    //     for i in self.outputs.clone().unwrap() {
    //         if abs(i-1.) < shortest_distance {
    //             shortest_distance = abs(i-1.);
    //             best_ind = ind;
    //         }
    //         ind+=1;
    //     }
    //     self.output = Some(best_ind);
    // }

    // pub fn backpropogate(&mut self) {
    //     let alpha: f64 = 0.001;
    //     // outer layer
    //     let mut weight_derivs: Vec<f64> =
    // }
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

use std::collections::HashSet;
use std::ops::Add;
use std::ops::Mul;
use petgraph::Graph;
use petgraph::dot::{Dot, Config};
use petgraph::visit;
use rand_distr::num_traits::Pow;

	#[derive(Default, Clone)]
	pub struct Value{
		pub data_: f64,
		pub gradient_: f64,
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


	pub fn mul_vec(mut arr1: Vec<Value>, mut arr2: Vec<Value>) -> Vec<Value> {

		if arr1.len() != arr2.len(){
			panic!("length of vectors are not right for dot product");
		}

		let mut ret = vec![Value::default(); arr1.len()];

		for i in 0..arr1.len(){
			let val1 = arr1.pop().unwrap();
			let val2 = arr2.pop().unwrap();
			ret[i] = Value::mul(val1, val2);
		}

		ret

	}

	pub fn add_vec(mut arr1: Vec<Value>, mut arr2: Vec<Value>) -> Vec<Value> {

		if arr1.len() != arr2.len(){
			panic!("length of vectors are not right for adding product");
		}

		let mut ret = vec![Value::default(); arr1.len()];

		for i in 0..arr1.len(){
			let val1 = arr1.pop().unwrap();
			let val2 = arr2.pop().unwrap();
			ret[i] = Value::add(val1, val2);
		}

		ret

	}

	pub fn vec_tot(mut arr1: Vec<Value>) -> Value {
		if arr1.len() == 0{
			panic!("no contents to add in vector");
		}
		else if arr1.len() == 1{
            
			return Value::add(arr1[0].clone(),Value::new(0.0));
		}
		else{
            let mut ret: Value = Value::new(0.0);
            for i in 0..arr1.len() {
                ret = Value::add(ret, arr1.pop().unwrap());
            }
            ret
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

