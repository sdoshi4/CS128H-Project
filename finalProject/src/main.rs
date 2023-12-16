extern crate finalProject;
use std::cell::{RefCell, Ref};
use std::rc::Rc;

use ndarray::Array2;
use finalProject::Perceptron;
// use finalProject::sigmoid;
// use finalProject::softmax;
use mnist::*;
//use petgraph::data;
//use std::thread;
//use std::time::Duration;
use finalProject::Value;
//use finalProject::Util;

 
fn main() {

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let test_data = Array2::from_shape_vec((10_000, 784), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels: Array2<usize> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as usize);

    let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let train_labels: Array2<usize> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as usize);

    let mut ptest = finalProject::Perceptron::new(784, 10, 1, 1);
    train(train_data, train_labels, &mut ptest, 300);
    run(test_data, test_labels, &mut ptest, 5000);
}

pub fn run(test_data: Array2<f32>, test_labels: Array2<usize>, perceptron: &mut Perceptron, data_count: usize) {

    let mut correct: i32 = 0;

    // let perceptron_outputs: Vec<usize> = Vec::new();
    for i in 0..data_count {
        let img = test_data.row(i).clone().into_shape((1, 784)).unwrap();
        let img2:&mut  Vec<Rc<RefCell<Value>>> = &mut img.map(|x| Rc::new(RefCell::new(Value::new(*x as f64)))).into_raw_vec();
        perceptron.run(img2);
        if perceptron.output.unwrap() == test_labels.row(i).to_vec()[0] {
            correct += 1;
        }
        // perceptron_outputs.push(perceptron)
        // let output: usize = perceptron.outputs.unwrap().iter().copied().max().unwrap();
    }
    println!("Accuracy: {:?}%", correct as f64 * 100. / data_count as f64);
    
}

pub fn train(train_data: Array2<f32>, train_labels: Array2<usize>, perceptron: &mut Perceptron, data_count:usize) {

	
    for i in 0..data_count {
        let img = train_data.row(i).clone().into_shape((1, 784)).unwrap();
        let img2: &mut Vec<Rc<RefCell<Value>>> = &mut img.map(|x| Rc::new(RefCell::new(Value::new(*x as f64)))).into_raw_vec();
        perceptron.run( img2);
        let mut encoded_labels: Vec<Rc<RefCell<Value>>> = Vec::new();
        for j in 0..10 {
            if j == train_labels.row(i).to_vec()[0] {
                encoded_labels.push(Rc::new(RefCell::new(Value::new(1.))));
            } else {
                encoded_labels.push(Rc::new(RefCell::new(Value::new(0.))));
            }
        }
        let po: &Vec<Rc<RefCell<Value>>>= perceptron.outputs.as_ref().unwrap();
        let losses = po.iter().zip(encoded_labels).map(|(x, y)| Value::mul(Value::add(Rc::clone(x), Value::mul(Rc::clone(&y), Rc::new(RefCell::new(Value::new(-1.))))), Value::add(Rc::clone(x), Value::mul(Rc::clone(&y), Rc::new(RefCell::new(Value::new(-1.)))))));
        let mut loss = Rc::new(RefCell::new(Value::new(0.)));
        for j in losses {
            loss = Value::add(loss, j);
        }

		//print!("perceptron check data before:{}", perceptron.layers.get(0).unwrap().neurons.get(0).unwrap().weights.get(0).unwrap().borrow().gradient_);

       // println!("Loss: {:?}", loss.borrow().data_);
        Value::backwards(Rc::clone(&loss));

		perceptron.layers.get(0).unwrap().neurons.get(0).unwrap().weights.get(0).unwrap().borrow_mut().data_ = 1000.0;

        let learning_rate = -(1.0-(0.9*(i as f64)/100.0));
        for layer in perceptron.layers.iter() {
            for neuron in layer.neurons.iter() {
                for val in neuron.weights.iter() {

					let val_data = val.borrow().data_;
					let val_grad = val.borrow().gradient_;
				
                    val.borrow_mut().data_ =  val_data + (learning_rate * val_grad);
                }

				let val_data = neuron.bias.borrow().data_ ;
				let val_grad = neuron.bias.borrow().gradient_;
                neuron.bias.borrow_mut().data_ = val_data + (learning_rate * val_grad);
            }
        }

		if i % 1 == 0{
			println!("step {} loss {}", i, loss.borrow().data_)
		}
        
		println!("data:{}", perceptron.layers.get(1).unwrap().neurons.get(0).unwrap().weights.get(0).unwrap().borrow().data_);
		println!("grad:{}", perceptron.layers.get(1).unwrap().neurons.get(0).unwrap().weights.get(0).unwrap().borrow().gradient_);
		println!("op:{}", perceptron.layers.get(0).unwrap().neurons.get(0).unwrap().weights.get(0).unwrap().borrow().op_);




        // if (perceptron.output.unwrap() == test_labels.row(i).to_vec()[0]) {
        //     correct += 1;
        // }
        // perceptron_outputs.push(perceptron)
        // let output: usize = perceptron.outputs.unwrap().iter().copied().max().unwrap();
    }
}
// Value::mul(x, y)




// fn main() {
//     println!("Hello, world!");

// 	let mut vec1: Vec<Rc<RefCell<Value>>> = Vec::new();
// 	let mut vec2: Vec<Rc<RefCell<Value>>> = Vec::new();

//     let mut a = Rc::new(RefCell::new(Value::new(6.0)));
// 	let mut b = Rc::new(RefCell::new(Value::new(5.0)));
// 	//let mut c = Value::mul(a, b);

// 	let p = Rc::clone(&b);
// 	let mut d = Rc::new(RefCell::new(Value::new(7.0)));
// 	let mut a1 = Rc::new(RefCell::new(Value::new(4.0)));

// 	vec1.push(a); vec1.push(b);
// 	vec2.push(d); vec2.push(a1);
	
// 	//let t = 
// 	 let mut mul = Util::mul_vec(&mut vec1, &mut vec2);
// 	//let tot = t.get(0).unwrap();
// 	let mut add = Util::add_vec(&mut vec1, &mut vec2);

// 	let tot = Value::tanh(Util::vec_tot(&mut add));
// 	//let tot = Value::tanh(Value::add(Util::vec_tot(&mut add), Util::vec_tot(&mut mul)));

// 	//let mut e = Value::add(d, c);

// 	//let mut f = Value::mul(e, a1);
// 	//let mut g = Value::tanh(f);
// 	print_vals(Rc::clone(&tot));

// 	Value::backwards(Rc::clone(&tot));
// 	//println!("line 169 b grad:{}", p.borrow().gradient_);
// 	println!();
// 	println!();
// 	print_vals(Rc::clone(&tot));

// 	//print!("e:{}", e.borrow.to_string());
// }

// fn print_vals(v: Rc<RefCell<Value>>){

// 	if v.borrow().prev_.len() == 1{
// 		print!("tanh()");
// 		print_vals(Rc::clone(v.borrow().prev_.get(0).unwrap()));
// 		println!();
// 	}
// 	else if v.borrow().prev_.len() == 2{
// 		print!("  EQ-");
// 		print_vals(Rc::clone(v.borrow().prev_.get(0).unwrap()));
// 		//print!("op:{}", v.borrow().op_);
// 		print_vals(Rc::clone(v.borrow().prev_.get(1).unwrap()));
// 		print!("-EQ  ");
// 		println!();
// 	}

// 	print!("{}", v.borrow().to_string());
// }
