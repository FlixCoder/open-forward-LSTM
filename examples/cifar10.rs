extern crate oflstm;
extern crate esopt;
extern crate ofnn;
extern crate rand;

use oflstm::*;
use esopt::*;
use oflstm::Float;
use ofnn::losses;
use rand::prelude::*;
use std::io::prelude::*;
use std::fs::File;
use std::time::Instant;

const BATCHSIZE:usize = 32; //number of items to form a batch inside evaluation
const STEPS:usize = 1; //number of LSTM runs/steps to decide the class

const LR:Float = 0.002; //learning rate for the optimizer
const LAMBDA:Float = 0.001; //weight decay factor
const ADABOUND:bool = true; //use AdaBound variant?
const FINAL_LR:Float = 0.1; //final AdaBound learning rate (SGD)
const GAMMA:Float = 0.001; //gamma value for AdaBound

const DROPOUT_IN:Float = 0.1; //dropout for input layers factor (percentage to be dropped)
const DROPOUT_OUT:Float = 0.1; //dropout factor (percentage to be dropped)

const NOISE_STD:Float = 0.02; //standard deviation of noise to mutate parameters and generate meta population
const POPULATION:usize = 250; //number of double-sided samples forming the meta population

//TODO:
//try adamax/SGD?
//try normal opt.optimize_par or _std_par?
//try dropout further, without weight decay?
//try L0.5 regularization


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
    //activate dropout for training
    model.set_input_dropout(DROPOUT_IN)
        .set_extract_dropout(DROPOUT_OUT);
    
    //create the evaluator
    let eval = CIFAR10Evaluator::new("cifar-10-binary/data_batch_1.bin", model.clone());
    
    //create or load optimizer
    let loaded = Adam::load("optimizer.json");
    let mut opt = if loaded.is_ok()
        {
            loaded.unwrap()
        }
        else
        {
            Adam::new()
        };
    opt.set_lr(LR)
        .set_lambda(LAMBDA)
        .set_adabound(ADABOUND)
        .set_gamma(GAMMA)
        .set_final_lr(FINAL_LR);
    
    //evolutionary optimizer (for more details about it, see the git repository of it)
    let mut opt = ES::new(opt, eval);
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
        let res = opt.optimize_par(n); //use ranked or not?
        
        //save results
        model.set_params(opt.get_params());
        model.reset();
        model.save("cifar10.lstm").ok();
        opt.get_opt().save("optimizer.json").ok();
        
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
    opt.get_opt().save("optimizer.json").ok();
    
    println!("Final results on test data:");
    tester.set_model(model.clone());
    tester.print_metrics();
    
    //clean up
    //std::fs::remove_file("cifar10.lstm").ok();
    //std::fs::remove_file("optimizer.json").ok();
}


fn load_cifar10(filename:&str) -> std::io::Result<(Vec<Vec<Float>>, Vec<Vec<Float>>)>
{
    let mut file = File::open(filename)?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut buffer = [0u8; 3073];
    
    for _i in 0..10000
    {
        file.read_exact(&mut buffer)?;
        y.push(to_categorical(10, buffer[0]));
        let data:Vec<Float> = buffer[1..].iter().map(|val| *val as Float / 128.0 - 1.0).collect();
        //let data:Vec<Float> = buffer[1..].iter().map(|val| *val as Float / 255.0).collect();
        x.push(data);
    }
    
    Ok((x, y))
}

fn to_categorical(classes:u8, label:u8) -> Vec<Float>
{
    let mut vec = vec![0.0; classes as usize];
    vec[label as usize] = 1.0;
    vec
}

fn argmax(vec:&[Float]) -> usize
{
    let mut argmax = 0;
    let mut max = std::f64::NEG_INFINITY as Float;
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
    data:(Vec<Vec<Float>>, Vec<Vec<Float>>),
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
        //disable dropout for testing
        self.model.set_input_dropout(0.0)
            .set_extract_dropout(0.0);
        
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
        acc *= 100.0 / pred.len() as Float;
        
        //display results
        println!("Loss: {}", loss);
        println!("Accuracy: {:6.3}%", acc);
        
        //reenable dropout after testing
        self.model.set_input_dropout(DROPOUT_IN)
            .set_extract_dropout(DROPOUT_OUT);
    }
}

impl Evaluator for CIFAR10Evaluator
{
    //make the model repeat numbers from two iterations ago
    fn eval_train(&self, params:&[Float], index:usize) -> Float
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
    
    fn eval_test(&self, params:&[Float]) -> Float
    {
        self.eval_train(params, 9999) //use index greater than can be used during training to possibly yield seperate test data (constant)
    }
}
