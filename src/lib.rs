//! @author = FlixCoder
//! open forward implementation of an LSTM

#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate ofnn;

use std::io::prelude::*;
use std::fs::File;
use ofnn::*;


/// Activations possible to use for the output
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum Activation
{
    Linear,
    ReLU,
    ELU,
    Sigmoid,
    Tanh,
    SoftMax,
}

/// Implementation of the LSTM-like architecture
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LSTM
{
    nn_gate:Sequential, //NN to gate forgetting and, after inversion, information gain
    nn_extract:Sequential, //NN to process input, memory and previous output into new information
    nn_output:Sequential, //NN to process input, memory and previous output into new output
    cur_mem: Vec<f64>, //current memory vector
    cur_out: Vec<f64>, //current output vector
    num_inputs: usize, //input dimension
    num_mem: usize, //memory dimension
    num_outputs: usize, //output dimension
}

impl LSTM
{
    /// Creates a new LSTM
    /// inputs = dimension of input data
    /// mem = dimension of memory vector
    /// outputs = dimension of output data
    pub fn new(inputs:usize, mem:usize, outputs:usize) -> LSTM
    {
        LSTM::new_ex(inputs, mem, outputs, Activation::Tanh)
    }
    
    /// Creates a new LSTM
    /// inputs = dimension of input data
    /// mem = dimension of memory vector
    /// outputs = dimension of output data
    /// out_act = output layer activation function
    pub fn new_ex(inputs:usize, mem:usize, outputs:usize, out_act:Activation) -> LSTM
    {
        //init variables
        let nn_in_size = inputs + mem + outputs;
        let mut gate = Sequential::new(nn_in_size);
        let mut extract = Sequential::new(nn_in_size);
        let mut output = Sequential::new(nn_in_size);
        let memvec = vec![0.0; mem as usize];
        let outvec = vec![0.0; outputs as usize];
        
        //translate activation into the corresponding sequential layer
        let mut out_init = Initializer::Glorot;
        let out_act = match out_act
        {
            Activation::Linear => Layer::Linear,
            Activation::ReLU => { out_init = Initializer::He; Layer::ReLU },
            Activation::ELU => { out_init = Initializer::He; Layer::ELU },
            Activation::Sigmoid => Layer::Sigmoid,
            Activation::Tanh => Layer::Tanh,
            Activation::SoftMax => Layer::SoftMax,
        };
        
        //create neural nets
        gate.add_layer_dense(mem, Initializer::Glorot)
            .add_layer(Layer::Sigmoid);
        extract.add_layer_dense(mem, Initializer::Glorot)
            .add_layer(Layer::Tanh);
        output.add_layer_dense(outputs, out_init)
            .add_layer(out_act);
        
        //create and return object
        LSTM { nn_gate: gate, nn_extract: extract, nn_output: output,
            cur_mem: memvec, cur_out: outvec,
            num_inputs: inputs, num_mem: mem, num_outputs: outputs }
    }
    
    /// Returns the requested input dimension
    pub fn get_num_inputs(&self) -> usize
    {
        self.num_inputs
    }
    
    /// Get the current memory
    pub fn get_mem(&self) -> &Vec<f64>
    {
        &self.cur_mem
    }
    
    /// Get the current memory (as mut). Possible to initialize the memory as needed
    pub fn get_mem_mut(&mut self) -> &mut Vec<f64>
    {
        &mut self.cur_mem
    }
    
    /// Get the current previous output
    pub fn get_out(&self) -> &Vec<f64>
    {
        &self.cur_out
    }
    
    /// Get the current previous output (as mut). Possible to initialize the previous output as needed
    pub fn get_out_mut(&mut self) -> &mut Vec<f64>
    {
        &mut self.cur_out
    }
    
    /// Return the flat parameters of the LSTM.
    /// Used for evolution-strategies
    pub fn get_params(&self) -> Vec<f64>
    {
        let mut params = self.nn_gate.get_params();
        params.append(&mut self.nn_extract.get_params());
        params.append(&mut self.nn_output.get_params());
        params
    }
    
    /// Set the LSTM's parameters by a flat input.
    /// Used for evolution-strategies.
    /// Panics if params' size does not fit the LSTM
    pub fn set_params(&mut self, params:&[f64]) -> &mut Self
    {
        //calculate the number of parameters for every NN
        let insize = self.num_inputs + self.num_mem + self.num_outputs;
        let len = insize * self.num_mem + self.num_mem; //weights + bias
        
        //set parameters of each of the NNs
        self.nn_gate.set_params(&params[0..len]); //output size = mem
        self.nn_extract.set_params(&params[len..2*len]); //output size = mem
        self.nn_output.set_params(&params[2*len..]); //rest of the parameters
        
        self
    }
    
    /// Reset current memory and output
    pub fn reset(&mut self)
    {
        self.cur_mem = vec![0.0; self.num_mem as usize];
        self.cur_out = vec![0.0; self.num_outputs as usize];
    }
    
    /// Runs an LSTM iteration
    pub fn run(&mut self, input:&Vec<f64>) -> Vec<f64>
    {
        if input.len() != self.num_inputs
        {
            panic!("Input dimension incorrect for this LSTM!");
        }
        
        //create vectors for NN input: one using the old memory, one using the updated
        let mut old_vec = input.clone();
        old_vec.extend_from_slice(&self.cur_out);
        let mut new_vec = old_vec.clone();
        old_vec.extend_from_slice(&self.cur_mem);
        
        //calculate memory update
        let mut gate = self.nn_gate.run(&old_vec);
        let mut info = self.nn_extract.run(&old_vec);
        elem_multiply(&mut self.cur_mem, &gate);
        elem_invert(&mut gate);
        elem_multiply(&mut info, &gate);
        elem_add(&mut self.cur_mem, &info);
        
        //compute and return output using new input vector
        new_vec.extend_from_slice(&self.cur_mem);
        self.cur_out = self.nn_output.run(&new_vec);
        self.cur_out.clone()
    }
    
    /// Encodes the model as a JSON string.
    pub fn to_json(&self) -> String
    {
        serde_json::to_string(self).expect("Encoding JSON failed!")
    }

    /// Builds a new model from a JSON string.
    pub fn from_json(encoded:&str) -> LSTM
    {
        serde_json::from_str(encoded).expect("Decoding JSON failed!")
    }
    
    /// Saves the model to a file
    pub fn save(&self, file:&str) -> Result<(), std::io::Error>
    {
        let mut file = File::create(file)?;
        let json = self.to_json();
        file.write_all(json.as_bytes())?;
        Ok(())
    }
    
    /// Creates a model from a previously saved file
    pub fn load(file:&str) -> Result<LSTM, std::io::Error>
    {
        let mut file = File::open(file)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        Ok(LSTM::from_json(&json))
    }
}


/// Elementwise vector multiplication
fn elem_multiply(a:&mut[f64], b:&[f64])
{
    for (x, y) in a.iter_mut().zip(b.iter())
    {
        *x *= *y;
    }
}

/// Elementwise vector "inversion"
fn elem_invert(vec:&mut[f64])
{
    vec.iter_mut().for_each(|val| *val = 1.0 - *val);
}

/// Elementwise vector addition
fn elem_add(a:&mut[f64], b:&[f64])
{
    for (x, y) in a.iter_mut().zip(b.iter())
    {
        *x += *y;
    }
}
