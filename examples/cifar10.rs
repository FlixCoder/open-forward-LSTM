extern crate oflstm;
extern crate esopt;
extern crate ofnn;
extern crate rand;

use oflstm::*;
use esopt::*;
use ofnn::losses;
use rand::prelude::*;
use std::io::prelude::*;
use std::fs::File;
use std::time::Instant;

const BATCHSIZE:usize = 64; //number of items to form a batch inside evaluation
const STEPS:usize = 1; //number of LSTM runs/steps to decide the class

const LR:f64 = 0.005; //learning rate for the optimizer
const LAMBDA:f64 = 0.0001; //weight decay factor
const NOISE_STD:f64 = 0.05; //standard deviation of noise to mutate parameters and generate meta population
const POPULATION:usize = 250; //number of double-sided samples forming the meta population


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
    let mut opt = ES::new_with_adam(eval, LR, LAMBDA); //learning rate, weight decay
    opt.set_params(model.get_params())
        .set_std(NOISE_STD)
        .set_samples(POPULATION);
    
    //show initial scores
    println!("Initial results on test set:");
    let mut tester = CIFAR10Evaluator::new("cifar-10-binary/test_batch.bin", model.clone());
    tester.print_metrics();
    
    //training: track the optimizer's results
    println!("Beginning training..");
    let time = Instant::now();
    for i in 0..10
    { //10 times
        //optimize for n steps
        let n = 10;
        let res = opt.optimize_std_par(n); //use ranked or not?
        
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
    let elapsed = time.elapsed();
    let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
    println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
    println!("");
    
    //save trained model and estimate and display results
    model.set_params(opt.get_params());
    model.reset();
    model.save("cifar10.lstm").ok();
    
    println!("Final results on test data:");
    tester.set_model(model.clone());
    tester.print_metrics();
    
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

fn argmax(vec:&[f64]) -> usize
{
    let mut argmax = 0;
    let mut max = std::f64::MIN;
    for (i, val) in vec.iter().enumerate()
    {
        if *val >= max
        {
            max = *val;
            argmax = i;
        }
    }
    argmax
}


#[derive(Clone)]
struct CIFAR10Evaluator
{
    model:LSTM,
    data:(Vec<Vec<f64>>, Vec<Vec<f64>>),
    seed:u64,
}

impl CIFAR10Evaluator
{
    pub fn new(filename:&str, model:LSTM) -> CIFAR10Evaluator
    {
        let data = load_cifar10(filename).unwrap();
        let seed = thread_rng().next_u64() % (std::u64::MAX - 10000); //prevent overflow when adding the index
        CIFAR10Evaluator { model: model, data: data, seed: seed }
    }
    
    pub fn set_model(&mut self, model:LSTM)
    {
        self.model = model;
    }
    
    pub fn print_metrics(&mut self)
    {
        //compute predicitions for whole data
        let mut pred = Vec::new();
        for x in &self.data.0
        {
            self.model.reset();
            for _ in 0..STEPS-1
            {
                self.model.run(x);
            }
            pred.push(self.model.run(x)); //last step results are saved as prediction
        }
        
        //calculate metrics
        let loss = losses::categorical_crossentropy(&pred, &self.data.1);
        let mut acc = 0.0;
        for (p, t) in pred.iter().zip(self.data.1.iter())
        {
            let select = argmax(p);
            if t[select] == 1.0
            {
                acc += 1.0;
            }
        }
        acc *= 100.0 / pred.len() as f64;
        
        //display results
        println!("Loss: {}", loss);
        println!("Accuracy: {:6.3}%", acc);
    }
}

impl Evaluator for CIFAR10Evaluator
{
    //make the model repeat numbers from two iterations ago
    fn eval_train(&self, params:&[f64], index:usize) -> f64
    {
        let mut local = self.model.clone();
        local.set_params(params);
        
        //every parameter pertubation uses the same training data, but every iteration uses different
        let mut rng = SmallRng::seed_from_u64(self.seed + index as u64);
        let start = rng.gen::<usize>() % (self.data.0.len() - BATCHSIZE); //not really uniform, but suffices
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
    
    fn eval_test(&self, params:&[f64]) -> f64
    {
        self.eval_train(params, 9999) //use index greater than can be used during training to possibly yield seperate test data (constant)
    }
}
