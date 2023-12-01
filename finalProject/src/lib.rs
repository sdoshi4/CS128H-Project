use ndarray::Array;
use ndarray::array;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_rand::RandomExt;
use ndarray::ArrayView;
use num::complex::ComplexFloat;
use rand_distr::Distribution;
use rand_distr::Normal;
use num::clamp;
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
}

pub fn sigmoid(x: f64) -> f64 {
    // x.map(|elem| 1. / (1. + (-elem).exp()))
    1. / (1. + (-x).exp())
}

pub fn relu(x: Array2<f64>) -> Array2<f64> {
    x.map(|elem| elem.max(0.0))
}

// pub fn print_arr2(arr: Array2<a>){
//     for row in array2.outer_iter() {
//         // Iterate through the elements in the row
//         for &element in row {
//             // Print each element
//             print!("{} ", element);
//         }
//         // Move to the next line after printing a row
//         println!();
//     }
// }

pub fn softmax(x: Array2<f64>) -> Array2<f64> {
        let mut output = Array2::<f64>::zeros(x.raw_dim());
        for (in_row, mut out_row) in x.axis_iter(Axis(0)).zip(output.axis_iter_mut(Axis(0))) {
            let mut max = *in_row.iter().next().unwrap();
            for col in in_row.iter() {
                if col > &max {
                    max = *col;
                }
            }
            let exp = in_row.map(|x| (x-max).exp()); // used to be (x - max).exp()
            let sum = exp.sum();
            out_row.assign(&(exp / sum));
        }
        output
}

pub struct Neuron {
    pub weights: Array2<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(prev_layer_size: usize) -> Self {
        let weights = Array::random((1, prev_layer_size), Normal::new(1.0, 0.0).unwrap());
        let bias: f64 = Array::random((1, 1), Normal::new(1.0, 0.0).unwrap()).row(0).to_vec()[0];
        Neuron {weights, bias }
    }
    pub fn print_nueron(self){
        println!("Neuron data:");
        print!("weights");
        for row in self.weights.outer_iter() {
            for &element in row {
                print!("{} ", element);
            }
            println!();
        }
        println!("biases{}", self.bias);

    }
}


pub struct Layer {
    // pub weights: Array2<f64>,
    // pub biases: Array2<f64>,
    pub neurons: Vec<Neuron>,
    pub outputs: Option<Array2<f64>>,
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
    pub fn forward(&mut self, inputs: &Array2<f64>, afunc: ActivationFunction) {
        let mut outp: Array2<f64>  = Array2::zeros((1, self.neurons.len()));
        for i in 0..self.neurons.len() {
            // let my_speed_ptr: *mut i32 = &mut my_speed;
            // let r: f64 = inputs.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // let r: f64 = inputs.iter().zip(&self.neurons[i].weights.iter()).map(|(x, y)| x * y).sum();
            // println!("{:?}", outp);
            let z: f64 = (inputs * (&self.neurons[i].weights)).sum() + &self.neurons[i].bias; //.remove_axis(ndarray::Axis(0)).to_vec()[0]
            outp[[0, i]] = z;
            // let a = relu(array![[z]]);
            // self.outputs.unwrap().append(axis, array)
            // if (afunc == ActivationFunction::Relu) {
            //     outp[[0, i]] = z; // array![[&z]]
            // } else if (afunc == ActivationFunction::Sigmoid) {
            //     outp[[0, i]] = sigmoid(z);
            // }
            // } else if (afunc == ActivationFunction::Softmax) {
            //     self.outputs = Some(softmax(inputs.dot(&self.weights) + &self.biases));
            // }
            // }
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
    pub outputs: Option<Array2<f64>>,
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
    pub fn run(&mut self, inputs: &Array2<f64>) {
        let output_layer_ind = self.layers.len()-1;
        // self.outputs = Some(inputs.dot(&self.weights));
        self.layers[0].forward(inputs, ActivationFunction::Relu);
        for i in (1..output_layer_ind) { // exclusive, so doesn't do output layer
            let prev = self.layers[i-1].outputs.clone().unwrap();
            self.layers[i].forward(&prev, ActivationFunction::Relu);
        }
        let last_hidden = self.layers[output_layer_ind-1].outputs.clone().unwrap();
        self.layers[output_layer_ind].forward(&last_hidden, ActivationFunction::Relu);
        self.outputs = self.layers[output_layer_ind].outputs.clone();
        self.find_output();

        // loss is just -log(ouput[correct_ind])
        // class_targets = [0, 1, 1]
        // for (targ_idx, distribution) in zip(class_targets, softmax_outputs):
        // print(distribution[targ_idx])
    }

    pub fn calculate_loss_log(&mut self, correct_ind: usize) -> f64 {
        let outputs_clipped: Array2<f64> = self.outputs.clone().unwrap().map(|v| num::clamp(*v, 1e-7, 1.-1e-7));
        -outputs_clipped[[0, correct_ind]].ln()
    }

    pub fn find_output(&mut self) {
        let mut best_ind: usize = 0;
        let mut shortest_distance: f64 = 10.0;
        let mut ind = 0;
        for i in self.outputs.clone().unwrap() {
            if abs(i-1.) < shortest_distance {
                shortest_distance = abs(i-1.);
                best_ind = ind;
            }
            ind+=1;
        }
        self.output = Some(best_ind);
    }

    // pub fn backpropogate(&mut self) {
    //     let alpha: f64 = 0.001;
    //     // outer layer
    //     let mut weight_derivs: Array2<f64> =
    // }
}