extern crate finalProject;
use finalProject::Neuron;
use ndarray::Array2;
use ndarray::array;
use ndarray::prelude::*;
use finalProject::Perceptron;
// use finalProject::sigmoid;
// use finalProject::softmax;
use mnist::*;



use finalProject::Value;
 
fn main() {
    // let mut p = nn::Perceptron::new(3, 1, 2);
    // let inputs: Array2::<f64> = array![[1.,2.,3.]];
    // p.run(inputs);
    // let mut p = nn::Perceptron::new(3, 1, 1, 2);
    // let inputs: Array2::<f64> = array![[1.,2.,3.]];
    // println!("{:?}", p.layers[p.layers.len()-1].neurons.len());
    // p.run(&inputs);

    // println!("{:?}", p.outputs.unwrap());
    
    // println!("{:?}", sigmoid(2.));

    // let inps: Array2::<f64> = array![[3.2,1.3,0.2,0.8]];
    // println!("{:?}", softmax(inps));

    // let mut p2 = nn::Perceptron::new(2, 2, 1, 2);
    // p2.outputs = Some(array![[0.7, 0.1, 0.2]]);
    // println!("{:?}", p2.calculate_loss(0));


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

    let mut ptest = finalProject::Perceptron::new(784, 10, 2, 16);
    // run(test_data, test_labels, &mut ptest);
    train(train_data, train_labels, &mut ptest);
    run(test_data, test_labels, &mut ptest);
    // println!("{:?}", ptest.outputs.unwrap());


 
    // let image_num = 5;
    // // Can use an Array2 or Array3 here (Array3 for visualization)
    // let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
    //     .expect("Error converting images to Array3 struct")
    //     .map(|x| *x as f32 / 256.0);
    // println!("{:#.1?}\n",train_data.slice(s![image_num, ..]));
 
    // // Convert the returned Mnist struct to Array2 format
    // let train_labels: Array2<usize> = Array2::from_shape_vec((50_000, 1), trn_lbl)
    //     .expect("Error converting training labels to Array2 struct")
    //     .map(|x| *x as usize);
    // println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );
 
    // let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
    //     .expect("Error converting images to Array3 struct")
    //     .map(|x| *x as f32 / 256.);
 
    // let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
    //     .expect("Error converting testing labels to Array2 struct")
    //     .map(|x| *x as f32);
}

pub fn run(test_data: Array2<f32>, test_labels: Array2<usize>, perceptron: &mut Perceptron) {

    let mut correct: i32 = 0;

    // let perceptron_outputs: Vec<usize> = Vec::new();
    for i in (0..10_000) {
        let img = test_data.row(i).clone().into_shape((1, 784)).unwrap();
        let img2: Vec<Value> = img.map(|x| Value::new(*x as f64)).into_raw_vec();
        perceptron.run(&img2);
        if (perceptron.output.unwrap() == test_labels.row(i).to_vec()[0]) {
            correct += 1;
        }
        // perceptron_outputs.push(perceptron)
        // let output: usize = perceptron.outputs.unwrap().iter().copied().max().unwrap();
    }
    println!("Accuracy: {:?}%", correct as f64 * 100. / 10_000.0);
    
}

pub fn train(train_data: Array2<f32>, train_labels: Array2<usize>, perceptron: &mut Perceptron) {
    for i in (0..5_000) {
        let img = train_data.row(i).clone().into_shape((1, 784)).unwrap();
        let img2: Vec<Value> = img.map(|x| Value::new(*x as f64)).into_raw_vec();
        perceptron.run(&img2);
        let mut encoded_labels: Vec<Value> = Vec::new();
        for j in 0..10 {
            if j == train_labels.row(i).to_vec()[0] {
                encoded_labels.push(Value::new(1.));
            } else {
                encoded_labels.push(Value::new(0.));
            }
        }
        let po = perceptron.outputs.clone().unwrap();
        let losses = po.iter().zip(encoded_labels).map(|(x, y)| Value::mul(Value::add(x.clone(), Value::mul(y.clone(), Value::new(-1.))), Value::add(x.clone(), Value::mul(y, Value::new(-1.)))));
        let mut loss: Value = Value::new(0.);
        for j in losses {
            loss = Value::add(loss, j);
        }
        println!("Loss: {:?}", loss.data_);
        loss.backwards();

        let learning_rate = -0.01;
        for mut layer in perceptron.layers.iter_mut() {
            for mut neuron in layer.neurons.iter_mut() {
                for mut val in neuron.weights.iter_mut() {
                    val.data_ = val.data_ + (learning_rate * val.gradient_);
                }
                neuron.bias.data_ = neuron.bias.data_ + (learning_rate * neuron.bias.gradient_);
            }
        }

        // if (perceptron.output.unwrap() == test_labels.row(i).to_vec()[0]) {
        //     correct += 1;
        // }
        // perceptron_outputs.push(perceptron)
        // let output: usize = perceptron.outputs.unwrap().iter().copied().max().unwrap();
    }
}
// Value::mul(x, y)