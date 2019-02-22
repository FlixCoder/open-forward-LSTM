extern crate oflstm;
extern crate esopt;
extern crate ofnn;
extern crate rand;

use oflstm::*;
use esopt::*;
use ofnn::losses;
use rand::Rng;
use std::io::prelude::*;
use std::fs::File;

const BATCHSIZE:usize = 64; //number of items to form a batch inside evaluation
const STEPS:usize = 1; //number of LSTM runs/steps to decide the class


fn main()
{
    //NN model
    let loaded = LSTM::load("cifar10.lstm");
    let mut model = if loaded.is_ok()
        { //try loaded model first
            loaded.unwrap()
        }
        else
        { //else construct it
            LSTM::new_ex(3072, 384, 10, Activation::SoftMax)
        };
    
    //create the evaluator
    let eval = CIFAR10Evaluator::new("cifar-10-binary/data_batch_1.bin", model.clone());
    
    //evolutionary optimizer (for more details about it, see the git repository of it)
    let mut opt = ES::new_with_adam(eval, 0.001, 0.0); //learning rate, weight decay
    opt.set_params(model.get_params())
        .set_std(0.02)
        .set_samples(50);
    
    //training: track the optimizer's results
    for i in 0..10
    { //10 times
        //optimize for n steps
        let n = 10;
        let res = opt.optimize_par(n);
        
        //save results
        model.set_params(opt.get_params());
        model.reset();
        model.save("cifar10.lstm").ok();
        
        //display progress
        println!("After {} iteratios:", (i+1) * n);
        println!("Score: {}", res.0);
        println!("Gradnorm: {}", res.1);
        println!("");
    }
    
    //display and save results
    model.set_params(opt.get_params());
    model.reset();
    model.save("cifar10.lstm").ok();
    
    //clean up
    //std::fs::remove_file("cifar10.lstm").ok();
}


fn load_cifar10(filename:&str) -> std::io::Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)>
{
    let mut file = File::open(filename)?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut buffer = [0u8; 3073];
    
    for _i in 0..10000
    {
        file.read_exact(&mut buffer)?;
        y.push(to_categorical(10, buffer[0]));
        let data:Vec<f64> = buffer[1..].iter().map(|val| *val as f64 / 128.0 - 1.0).collect();
        x.push(data);
    }
    
    Ok((x, y))
}

fn to_categorical(classes:u8, label:u8) -> Vec<f64>
{
    let mut vec = vec![0.0; classes as usize];
    vec[label as usize] = 1.0;
    vec
}


#[derive(Clone)]
struct CIFAR10Evaluator
{
    model:LSTM,
    data:(Vec<Vec<f64>>, Vec<Vec<f64>>),
}

impl CIFAR10Evaluator
{
    pub fn new(filename:&str, model:LSTM) -> CIFAR10Evaluator
    {
        let data = load_cifar10(filename).unwrap();
        CIFAR10Evaluator { model: model, data: data }
    }
}

impl Evaluator for CIFAR10Evaluator
{
    //make the model repeat numbers from two iterations ago
    fn eval(&self, params:&[f64]) -> f64
    {
        let mut local = self.model.clone();
        local.set_params(params);
        
        let mut rng = rand::thread_rng();
        let start = rng.gen::<usize>() % (self.data.0.len() - BATCHSIZE);
        let end = start + BATCHSIZE;
        
        let mut pred = Vec::new();
        for x in &self.data.0[start..end]
        {
            local.reset();
            for _ in 0..STEPS-1
            {
                local.run(x);
            }
            pred.push(local.run(x)); //last step results are saved as prediction
        }
        
        -losses::categorical_crossentropy(&pred, &self.data.1[start..end])
    }
}
