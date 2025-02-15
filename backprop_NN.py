"""\
------------------------------------------------------------
NeuralNetwork class that implements all the necessary
functions to have a working back-propagating Neural Network
------------------------------------------------------------\
"""
import numpy as np

class NeuralNetwork:

    def __init__(self,lr,inputs,outputs,w1,w2,num_samples,bias1,bias2):
        self.lr = lr
        self.inputs = inputs
        self.outputs = outputs
        self.w1 = w1
        self.w2 = w2
        self.num_samples = num_samples
        self.num_neurons = w1.shape[0]
        self.bias1 = bias1
        self.bias2 = bias2

    
    ## Signmoid function 
    def sigmoid(self,x):
        sig = 1 /( (1 + np.exp(-x)) + 1e-8 ) # Adding a small number to avoid division by zero
        return sig

    ## Hyperbolic Tangent sigmoid function
    def tanh(self,x):
        tanh = np.tanh(x)
        return tanh

    ## Loss function using mean squared error
    def mse(self, pred):
        # pred = self.scale_coords(pred)
        error = np.mean((self.outputs - pred) ** 2)
        return error

    ## Loss function for angle outputs using trigonometric functions
    #  where the error is calculated as sin(theta/2)^2, being 1 the maximum error
    #  when the angles are opposite (pi) and 0 when they are equal
    def mean_angle_error(self, pred):
        # pred = self.scale_angles(pred)
        error = np.mean(np.sin((self.outputs - pred) / 2) ** 2)
        return error

    ## Propagate the inputs through the neural network
    def propagate_forward(self):
    
        w1 = self.w1
        w2 = self.w2
        x = self.inputs
        

        z1 = np.dot(w1,x) + self.bias1
        a1 = self.sigmoid(z1) # Using sigmoid as activation function
        z2 = np.dot(w2,a1) + self.bias2
        a2 = z2 # Using linear activation function
        # a2 = self.sigmoid(z2) # Using sigmoid as activation function
        return z1,a1,z2,a2

    ## Back-propagate the error 
    def propagate_backwards(self,z1,a1,z2,a2):
        
        w1 = self.w1
        w2 = self.w2
        x = self.inputs
        y = self.outputs
        m = self.num_samples
        bias1 = self.bias1
        bias2 = self.bias2

        ## delta member for angular loss function ##
        dz2 = - np.sin((y-a2)/2) * np.cos((y-a2)/2)
        
        ## delta member for mse loss functioon ##
        # dz2 = - ( y - a2 )
        dz1 = np.dot(w2.T,dz2) * a1*(1-a1)

        w1 -= self.lr * np.dot(dz1,x.T) / m
        w2 -= self.lr * np.dot(dz2,a1.T) / m
        bias1 -= self.lr * np.mean(dz1, axis=1, keepdims=True)
        bias2 -= self.lr * np.mean(dz2, axis=1, keepdims=True)

        return w1,w2,bias1,bias2
