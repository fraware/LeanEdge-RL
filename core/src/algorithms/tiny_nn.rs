use crate::{
    error::{Error, Result},
    obs::Obs,
    action::Action,
    algorithms::{Policy, utils},
};

/// Tiny Neural Network implementation (≤ 3 hidden layers)
pub struct TinyNN<const OBS_DIM: usize, const ACTION_DIM: usize> {
    /// Layer configurations: [input_size, hidden1_size, hidden2_size, ..., output_size]
    layer_sizes: Vec<usize>,
    /// Weights for each layer: [layer][output][input]
    weights: Vec<Vec<Vec<f32>>>,
    /// Biases for each layer: [layer][output]
    biases: Vec<Vec<f32>>,
    /// Activation functions for each layer
    activations: Vec<ActivationFunction>,
}

/// Supported activation functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    Linear,
}

impl ActivationFunction {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => utils::relu(x),
            Self::Tanh => utils::tanh(x),
            Self::Sigmoid => utils::sigmoid(x),
            Self::Linear => x,
        }
    }
    
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::ReLU),
            1 => Ok(Self::Tanh),
            2 => Ok(Self::Sigmoid),
            3 => Ok(Self::Linear),
            _ => Err(Error::InvalidWeights(format!("Unknown activation function: {}", value))),
        }
    }
    
    pub fn to_u8(self) -> u8 {
        match self {
            Self::ReLU => 0,
            Self::Tanh => 1,
            Self::Sigmoid => 2,
            Self::Linear => 3,
        }
    }
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> TinyNN<OBS_DIM, ACTION_DIM> {
    /// Create new TinyNN with default architecture
    pub fn new() -> Self {
        // Default architecture: input -> 64 -> 32 -> output
        let layer_sizes = vec![OBS_DIM, 64, 32, ACTION_DIM];
        let activations = vec![
            ActivationFunction::ReLU,
            ActivationFunction::ReLU,
            ActivationFunction::Tanh, // Output layer uses tanh for bounded actions
        ];
        
        Self::with_architecture(layer_sizes, activations)
    }
    
    /// Create TinyNN with custom architecture
    pub fn with_architecture(layer_sizes: Vec<usize>, activations: Vec<ActivationFunction>) -> Self {
        assert!(layer_sizes.len() >= 2, "At least input and output layers required");
        assert!(layer_sizes.len() <= 5, "Maximum 3 hidden layers allowed"); // input + 3 hidden + output
        assert_eq!(activations.len(), layer_sizes.len() - 1, "One activation per layer");
        
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        // Initialize weights and biases for each layer
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            
            // Initialize weights with Xavier/Glorot initialization
            let mut layer_weights = vec![vec![0.0; input_size]; output_size];
            let scale = (2.0 / input_size as f32).sqrt();
            
            for out_idx in 0..output_size {
                for in_idx in 0..input_size {
                    // Simple initialization for deterministic behavior
                    layer_weights[out_idx][in_idx] = (out_idx as f32 + in_idx as f32) * scale * 0.01;
                }
            }
            
            weights.push(layer_weights);
            biases.push(vec![0.0; output_size]);
        }
        
        Self {
            layer_sizes,
            weights,
            biases,
            activations,
        }
    }
    
    /// Create from weights
    pub fn from_weights(weights: &[u8]) -> Result<Self> {
        if weights.len() < 8 {
            return Err(Error::InvalidWeights("Insufficient weights for TinyNN".to_string()));
        }
        
        // Parse header: [num_layers, activation1, activation2, ...] (2 bytes each)
        let num_layers = u16::from_le_bytes([weights[0], weights[1]]) as usize;
        if num_layers < 2 || num_layers > 5 {
            return Err(Error::InvalidWeights("Invalid number of layers".to_string()));
        }
        
        let header_size = 2 + num_layers; // num_layers + activation functions
        if weights.len() < header_size {
            return Err(Error::InvalidWeights("Insufficient header for TinyNN".to_string()));
        }
        
        let mut activations = Vec::new();
        for i in 0..num_layers - 1 {
            let activation = ActivationFunction::from_u8(weights[2 + i])?;
            activations.push(activation);
        }
        
        // Parse layer sizes (assuming fixed architecture for now)
        let layer_sizes = vec![OBS_DIM, 64, 32, ACTION_DIM];
        if layer_sizes.len() != num_layers {
            return Err(Error::InvalidWeights("Layer size mismatch".to_string()));
        }
        
        let mut nn = Self::with_architecture(layer_sizes, activations);
        
        // Load weights and biases
        let weights_data = &weights[header_size..];
        nn.load_weights_and_biases(weights_data)?;
        
        Ok(nn)
    }
    
    /// Load weights and biases from bytes
    fn load_weights_and_biases(&mut self, data: &[u8]) -> Result<()> {
        let mut offset = 0;
        
        for layer_idx in 0..self.weights.len() {
            let input_size = self.layer_sizes[layer_idx];
            let output_size = self.layer_sizes[layer_idx + 1];
            
            // Load weights
            let weights_size = input_size * output_size * 4;
            if offset + weights_size > data.len() {
                return Err(Error::InvalidWeights("Insufficient data for weights".to_string()));
            }
            
            let weights_data = &data[offset..offset + weights_size];
            for (i, chunk) in weights_data.chunks(4).enumerate() {
                let out_idx = i / input_size;
                let in_idx = i % input_size;
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                self.weights[layer_idx][out_idx][in_idx] = value;
            }
            offset += weights_size;
            
            // Load biases
            let biases_size = output_size * 4;
            if offset + biases_size > data.len() {
                return Err(Error::InvalidWeights("Insufficient data for biases".to_string()));
            }
            
            let biases_data = &data[offset..offset + biases_size];
            for (i, chunk) in biases_data.chunks(4).enumerate() {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                self.biases[layer_idx][i] = value;
            }
            offset += biases_size;
        }
        
        Ok(())
    }
    
    /// Forward pass through the network
    fn forward(&self, input: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        let mut current_layer = input.as_slice().to_vec();
        
        for layer_idx in 0..self.weights.len() {
            let mut next_layer = vec![0.0; self.layer_sizes[layer_idx + 1]];
            
            // Matrix multiplication: next_layer = weights * current_layer + bias
            for out_idx in 0..self.layer_sizes[layer_idx + 1] {
                let mut sum = self.biases[layer_idx][out_idx];
                for in_idx in 0..self.layer_sizes[layer_idx] {
                    sum += self.weights[layer_idx][out_idx][in_idx] * current_layer[in_idx];
                }
                next_layer[out_idx] = self.activations[layer_idx].apply(sum);
            }
            
            current_layer = next_layer;
        }
        
        // Convert to Action
        let mut action_values = [0.0; ACTION_DIM];
        for (i, &val) in current_layer.iter().enumerate() {
            if i < ACTION_DIM {
                action_values[i] = val;
            }
        }
        
        Action::new(action_values)
    }
    
    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layer_sizes.len()
    }
    
    /// Get layer size at index
    pub fn layer_size(&self, layer_idx: usize) -> usize {
        self.layer_sizes[layer_idx]
    }
    
    /// Get weight at specific position
    pub fn get_weight(&self, layer_idx: usize, out_idx: usize, in_idx: usize) -> f32 {
        self.weights[layer_idx][out_idx][in_idx]
    }
    
    /// Get bias at specific position
    pub fn get_bias(&self, layer_idx: usize, out_idx: usize) -> f32 {
        self.biases[layer_idx][out_idx]
    }
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> Policy<OBS_DIM, ACTION_DIM> 
    for TinyNN<OBS_DIM, ACTION_DIM> 
{
    fn act(&self, obs: &Obs<OBS_DIM>) -> Action<ACTION_DIM> {
        self.forward(obs)
    }
    
    fn update_weights(&mut self, weights: &[u8]) -> Result<()> {
        if weights.len() < 8 {
            return Err(Error::InvalidWeights("Insufficient weights for TinyNN".to_string()));
        }
        
        let num_layers = u16::from_le_bytes([weights[0], weights[1]]) as usize;
        let header_size = 2 + num_layers;
        
        if weights.len() >= header_size {
            let weights_data = &weights[header_size..];
            self.load_weights_and_biases(weights_data)?;
        }
        
        Ok(())
    }
    
    fn get_weights(&self) -> Result<Vec<u8>> {
        let mut weights = Vec::new();
        
        // Header: [num_layers, activation1, activation2, ...]
        weights.extend((self.layer_sizes.len() as u16).to_le_bytes());
        for activation in &self.activations {
            weights.push(activation.to_u8());
        }
        
        // Weights and biases for each layer
        for layer_idx in 0..self.weights.len() {
            // Weights
            for out_idx in 0..self.weights[layer_idx].len() {
                for in_idx in 0..self.weights[layer_idx][out_idx].len() {
                    weights.extend(self.weights[layer_idx][out_idx][in_idx].to_le_bytes());
                }
            }
            
            // Biases
            for &bias_val in &self.biases[layer_idx] {
                weights.extend(bias_val.to_le_bytes());
            }
        }
        
        Ok(weights)
    }
    
    fn algorithm_name(&self) -> &'static str {
        "TinyNN"
    }
}

impl<const OBS_DIM: usize, const ACTION_DIM: usize> Default for TinyNN<OBS_DIM, ACTION_DIM> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tiny_nn_creation() {
        let nn = TinyNN::<4, 2>::new();
        assert_eq!(nn.num_layers(), 4); // input + 2 hidden + output
        assert_eq!(nn.layer_size(0), 4); // input
        assert_eq!(nn.layer_size(3), 2); // output
    }
    
    #[test]
    fn test_tiny_nn_custom_architecture() {
        let layer_sizes = vec![4, 8, 2];
        let activations = vec![ActivationFunction::ReLU, ActivationFunction::Tanh];
        
        let nn = TinyNN::<4, 2>::with_architecture(layer_sizes, activations);
        assert_eq!(nn.num_layers(), 3);
        assert_eq!(nn.layer_size(1), 8);
    }
    
    #[test]
    fn test_tiny_nn_from_weights() {
        let mut weights = Vec::new();
        weights.extend((4u16).to_le_bytes()); // num_layers
        weights.push(0); // ReLU
        weights.push(0); // ReLU
        weights.push(1); // Tanh
        
        // Add some dummy weights (simplified)
        for i in 0..100 {
            weights.extend((i as f32 * 0.01).to_le_bytes());
        }
        
        let nn = TinyNN::<4, 2>::from_weights(&weights);
        assert!(nn.is_ok());
    }
    
    #[test]
    fn test_tiny_nn_forward() {
        let nn = TinyNN::<4, 2>::new();
        let obs = Obs::new([1.0, 2.0, 3.0, 4.0]);
        let action = nn.act(&obs);
        
        // Should return valid action
        assert_eq!(action.as_slice().len(), 2);
        for &val in action.as_slice() {
            assert!(val.is_finite());
        }
    }
    
    #[test]
    fn test_activation_functions() {
        assert_eq!(ActivationFunction::ReLU.apply(1.0), 1.0);
        assert_eq!(ActivationFunction::ReLU.apply(-1.0), 0.0);
        assert_eq!(ActivationFunction::Linear.apply(0.5), 0.5);
    }
} 