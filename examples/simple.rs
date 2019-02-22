extern crate oflstm;
extern crate esopt;
extern crate ofnn;
extern crate rand;

use oflstm::*;
use esopt::*;
use ofnn::losses;
use rand::Rng;


fn main()
{
    //NN model
    let loaded = LSTM::load("test.nn");
    let mut model = if loaded.is_ok()
        { //try loaded model first
            loaded.unwrap()
        }
        else
        { //else construct it
            LSTM::new_ex(1, 5, 1, Activation::Linear) //input size = 1, memory size = 5, output size = 1
        };
    
    //create the evaluator
    let eval = LSTMEvaluator::new(model.clone());
    
    //evolutionary optimizer (for more details about it, see the git repository of it)
    let mut opt = ES::new_with_adam(eval, 0.025, 0.0); //learning rate, weight decay
    opt.set_params(model.get_params())
        .set_std(0.025)
        .set_samples(50);
    
    //training: track the optimizer's results
    for i in 0..5
    {
        let n = 200;
        let res = opt.optimize_par(n); //optimize for n steps
        println!("After {} iteratios:", (i+1) * n);
        println!("Score: {}", res.0);
        println!("Gradnorm: {}", res.1);
        println!("");
    }
    
    //display and save results
    model.set_params(opt.get_params());
    model.reset();
    model.save("test.nn").ok();
    
    let mut rng = rand::thread_rng();
    let mut target = Vec::new();
    let mut pred = Vec::new();
    
    let n = 7;
    for i in 0..n
    {
        let input = vec![rng.gen::<f64>()];
        let out = model.run(&input);
        if i < n-2 { target.push(input); }
        if i >= 2 { pred.push(out); }
    }
    
    for (p, t) in pred.iter().zip(target.iter())
    {
        println!("Target {:.3} <=> {:.3} Prediction", t[0], p[0]);
    }
    
    //clean up
    std::fs::remove_file("test.nn").ok();
}


#[derive(Clone)]
struct LSTMEvaluator
{
    model:LSTM,
}

impl LSTMEvaluator
{
    pub fn new(model:LSTM) -> LSTMEvaluator
    {
        LSTMEvaluator { model: model }
    }
}

impl Evaluator for LSTMEvaluator
{
    //make the model repeat numbers from two iterations ago
    fn eval(&self, params:&Vec<f64>) -> f64
    {
        let mut local = self.model.clone();
        local.set_params(params);
        local.reset();
        
        let mut rng = rand::thread_rng();
        let mut target = Vec::new();
        let mut pred = Vec::new();
        
        let n = 52;
        for i in 0..n
        {
            let input = vec![rng.gen::<f64>()];
            let out = local.run(&input);
            if i < n-2 { target.push(input); }
            if i >= 2 { pred.push(out); }
        }
        
        -losses::mae(&pred, &target)
    }
}
