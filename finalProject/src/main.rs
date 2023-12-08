extern crate finalProject;
use ndarray::Array2;
use ndarray::array;
use ndarray::prelude::*;
use finalProject::Perceptron;
// use finalProject::sigmoid;
// use finalProject::softmax;
use mnist::*;
// use itertools::Itertools;


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

    let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let train_labels: Array2<usize> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as usize);

    let test_data = Array2::from_shape_vec((10_000, 784), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels: Array2<usize> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as usize);

    // let mut ptest = nngenetic::Perceptron::new(784, 10, 2, 16);
    // run(test_data, test_labels, &mut ptest);
    let mut best_perceptron = train(train_data, train_labels);
    run(test_data, test_labels, &mut best_perceptron);
}

pub fn train(train_data: Array2<f32>, train_labels: Array2<usize>) -> Perceptron {
    let generation_size = 100;
    let num_best_perceptrons = 10;
    let num_clones = generation_size / num_best_perceptrons;
    let generations = 15;
    let mut dev = 1.;

    let mut perceptrons: Vec<Perceptron> = Vec::new();
    for _ in 0..generation_size {
        let mut p = Perceptron::new(784, 10, 2, 16);
        p.training_loss(&train_data, &train_labels, 50_000);
        perceptrons.push(p);
    }
    perceptrons.sort_by(|a, b| a.loss.unwrap().partial_cmp(&b.loss.unwrap()).unwrap());
    println!("{}", perceptrons[0].loss.unwrap());

    for g in 1..generations {
        let mut best_loss = 999999999.;
        let mut perceptrons_new: Vec<Perceptron> = Vec::new();
        // perceptrons_new.push(perceptrons[0]);
        for n in 0..num_best_perceptrons {
            for m in 0..num_clones {
                let mut clone = perceptrons[n].randomClone(dev);
                clone.training_loss(&train_data, &train_labels, 50_000);
                if (clone.loss.unwrap() < best_loss) {
                    best_loss = clone.loss.unwrap();
                }
                perceptrons_new.push(clone);
            }
        }
        println!("{}", best_loss);
        perceptrons_new.sort_by(|a, b| a.loss.unwrap().partial_cmp(&b.loss.unwrap()).unwrap());
        perceptrons = perceptrons_new;
        dev *= 0.98;
    }
    return perceptrons[0].clone();

    // let mut best_perceptron = &perceptrons[0];

    // for x in 0..perceptrons.len() {
    //     println!("{}", perceptrons[x].loss.unwrap());
    // }

    // perceptrons.sort_by_key(|d| d.loss); // This is fine


    // let mut best_perceptron = Perceptron::new(1, 1, 1, 1);
    // let mut best_loss = perceptrons[0].loss.unwrap().clone();



    // for b in 0..num_best_perceptrons{

    // }

    


    // for g in 1..generations {

    // }

    // best_perceptron
    // let mut correct: i32 = 0;

    // // let perceptron_outputs: Vec<usize> = Vec::new();
    // for i in (0..10_000) {
    //     let img = test_data.row(i).clone().into_shape((1, 784)).unwrap();
    //     let img2: Array2<f64> = img.map(|x| *x as f64);
    //     perceptron.run(&img2);
    //     if (perceptron.output.unwrap() == test_labels.row(i).to_vec()[0]) {
    //         correct += 1;
    //     }
    //     // perceptron_outputs.push(perceptron)
    //     // let output: usize = perceptron.outputs.unwrap().iter().copied().max().unwrap();
    // }
    // println!("Accuracy: {:?}%", correct as f64 * 100. / 10_000.0);
    
}

pub fn run(test_data: Array2<f32>, test_labels: Array2<usize>, perceptron: &mut Perceptron) {

    let mut correct: i32 = 0;

    // let perceptron_outputs: Vec<usize> = Vec::new();
    for i in (0..10_000) {
        let img = test_data.row(i).clone().into_shape((1, 784)).unwrap();
        let img2: Array2<f64> = img.map(|x| *x as f64);
        perceptron.run(&img2);
        if (perceptron.output.unwrap() == test_labels.row(i).to_vec()[0]) {
            correct += 1;
        }
        // perceptron_outputs.push(perceptron)
        // let output: usize = perceptron.outputs.unwrap().iter().copied().max().unwrap();
    }
    println!("Accuracy: {:?}%", correct as f64 * 100. / 10_000.0);
    
}