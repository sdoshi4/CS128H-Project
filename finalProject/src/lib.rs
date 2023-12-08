use std::ops::Mul;

use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Zip;
use ndarray_rand::RandomExt;
use num::complex::ComplexFloat;
use rand_distr::Distribution;
use rand_distr::Normal;
use num::clamp;
// use ndarray_einsum_beta::einsum;
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


pub fn sigmoid(x: Array2<f64>) -> Array2<f64> {
    x.map(|elem| 1. / (1. + (-elem).exp()))
}  


pub fn relu(x: Array2<f64>) -> Array2<f64> {
    x.map(|elem| elem.max(0.0))
}


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


// pub struct Neuron {
//     pub weights: Array2<f64>,
//     pub bias: f64,
// }


// impl Neuron {
//     pub fn new(prev_layer_size: usize) -> Self {
//         let weights = Array::random((1, prev_layer_size), Normal::new(0.0, 1.0).unwrap());
//         let bias: f64 = Array::random((1, 1), Normal::new(0.0, 1.0).unwrap()).row(0).to_vec()[0];
//         Neuron {weights, bias }
//     }
// }

pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub outputs: Option<Array2<f64>>,
}


impl Layer {
    pub fn new(layer_size: usize, prev_layer_size: usize) -> Self {
        let weights = Array::random((prev_layer_size, layer_size), Normal::new(0.0, 1.0).unwrap());
        let biases = Array::random((1, layer_size), Normal::new(0.0, 1.0).unwrap());
        let outputs = None;
        Layer {weights, biases, outputs}
    }
    pub fn forward(&mut self, inputs: &Array2<f64>, afunc: ActivationFunction) {
        if (afunc == ActivationFunction::Relu) {
            self.outputs = Some(relu(inputs.dot(&self.weights) + &self.biases));
        } else if (afunc == ActivationFunction::Sigmoid) {
            self.outputs = Some(sigmoid(inputs.dot(&self.weights) + &self.biases));
        } else if (afunc == ActivationFunction::Softmax) {
            self.outputs = Some(softmax(inputs.dot(&self.weights) + &self.biases));
        }
    }

    pub fn randomClone(&mut self, dev: f64) -> Layer {
        // for every weight and bias, multiply it by a normal distribution.
        let weight_distros = Array::random(self.weights.shape(), Normal::new(0.0, dev).unwrap());
        // let new_weights = einsum("ij,ij->i+j", &[&weight_distros, &self.weights]).unwrap().into_dimensionality().unwrap();
        let new_weights = (self.weights.clone() + weight_distros).into_dimensionality().unwrap();
        // let new_weights = self.weights.clone() * weight_distros;
        // let new_weights = einsum!("ij,ij->ij", self.weights.clone(), weights_distros).unwrap();
        let bias_distros = Array::random(self.biases.shape(), Normal::new(0.0, dev).unwrap());
        // let new_biases= self.biases.clone() * bias_distros;
        let new_biases = (self.biases.clone() + bias_distros).into_dimensionality().unwrap();
        // let new_biases = einsum("ij,ij->i+j", &[&self.biases, &bias_distros]).unwrap().into_dimensionality().unwrap();
        // let new_biases = einsum!("ij,ij->ij", biases, bias_distros).unwrap();
        // np.einsum('ij,ij->ij', A, B)
        // einsum!("ijk,ji->ik", tensor, another_tensor).unwrap();
        Layer { weights: new_weights, biases: new_biases, outputs: None }
    }

    pub fn clone(&mut self) -> Layer {
        // for every weight and bias, multiply it by a normal distribution.
        let new_biases = self.biases.clone();
        let new_weights = self.weights.clone();
        Layer { weights: new_weights, biases: new_biases, outputs: None }
    }
}


pub struct Perceptron {
    pub layers: Vec<Layer>,
    pub outputs: Option<Array2<f64>>,  
    pub output: Option<usize>,
    pub loss: Option<f64>,
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
        let loss = None;
        Perceptron { layers, outputs, output, loss }
    }
    pub fn run(&mut self, inputs: &Array2<f64>) {
        let output_layer_ind = self.layers.len()-1;
        // self.outputs = Some(inputs.dot(&self.weights));
        self.layers[0].forward(inputs, ActivationFunction::Sigmoid);
        for i in (1..output_layer_ind) { // exclusive, so doesn't do output layer
            let prev = self.layers[i-1].outputs.clone().unwrap();
            self.layers[i].forward(&prev, ActivationFunction::Sigmoid);
        }
        let last_hidden = self.layers[output_layer_ind-1].outputs.clone().unwrap();
        self.layers[output_layer_ind].forward(&last_hidden, ActivationFunction::Softmax);
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

    pub fn training_loss(&mut self, train_data: &Array2<f32>, train_labels: &Array2<usize>, num_images: usize) {
        // let mut loss: f64 = 0.0;
        // for i in 0..num_images {
        //     let img = train_data.row(i).clone().into_shape((1, 784)).unwrap().map(|x| *x as f64);
        //     // let img2: Array2<f64> = img.map(|x| *x as f64);
        //     self.run(&img);

        //     let mut encoded_labels: Vec<f64> = Vec::new();
        //     for j in 0..10 {
        //         if j == train_labels.row(i).to_vec()[0] {
        //             encoded_labels.push(1.);
        //         } else {
        //             encoded_labels.push(0.);
        //         }
        //     }
        //     let po = self.outputs.clone().unwrap();
        //     loss += po.iter().zip(encoded_labels).map(|(x, y)| (x-y).powi(2)).sum::<f64>();
        // }
        // self.loss = Some(loss);
        let mut num_wrong = 0.0;
        for i in 0..num_images {
            let img = train_data.row(i).clone().into_shape((1, 784)).unwrap().map(|x| *x as f64);
            self.run(&img);
            if (self.output.unwrap() != train_labels.row(i).to_vec()[0]) {
                num_wrong += 1.0;
            }
        }
        self.loss = Some(num_wrong);
    }

    pub fn randomClone(&mut self, dev: f64) -> Perceptron {

        let mut new_layers = Vec::new();
        for i in 0..self.layers.len() {
            new_layers.push(self.layers[i].randomClone(dev));
        }
        let mut new_perceptron = Perceptron {layers: new_layers, outputs: None, output: None, loss: None};
        new_perceptron
    }

    pub fn clone(&mut self) -> Perceptron {

        let mut new_layers = Vec::new();
        for i in 0..self.layers.len() {
            new_layers.push(self.layers[i].clone());
        }
        let mut new_perceptron = Perceptron {layers: new_layers, outputs: None, output: None, loss: None};
        new_perceptron
    }


    // pub fn backpropogate(&mut self) {
    //     let alpha: f64 = 0.001;
    //     // outer layer
    //     let mut weight_derivs: Array2<f64> =
    // }
}

// pub fn Hadamard(array1: &Array2<f64>, array2: &Array2<f64>) -> Array2<f64> {
//     let result = Zip::from(array1)
//         .and(array2)
//         .for_each(|&x, &y| x * y);
//     result
// }