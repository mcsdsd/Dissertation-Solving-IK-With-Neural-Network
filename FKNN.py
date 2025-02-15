from backprop_NN import NeuralNetwork
from FK_Model import ForwardKinematics
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns

## Stick Robot joint coordinates
u = np.array([0,1,0])
line1_end = np.array([2,1,1])
v = np.array([2,1,1])
line2_end = np.array([4,1,0])
w = np.array([4,1,0])
line3_end = np.array([6,1,3])

# Number of samples,inputs and outputs
num_samples = 2500
nInputs = 3
nOutputs = 3

# Number of iterations
num_iterations = 10000

# Learning rate
lr = 0.15
# Best learning rate so far: 0.03595

# Initialising an empty list to store the loss values
loss_values = []

# Neurons in hidden layer
num_neurons = 16

# Initialising weights randomly for three inputs, four neurons and three outputs
w1 = np.random.rand(num_neurons,nInputs) # Initialize weights for hidden layer
w2 = np.random.rand(nOutputs,num_neurons) # Initialize weights for output layer

## Initialising biases randomly between -2 and 2
bias1 = np.random.uniform(-2,2,(num_neurons, 1))  # Initialize bias for hidden layer
bias2 = np.random.uniform(-2,2,(nOutputs, 1)) 


######### Generating inputs #########

# Generate random angles in radians for each coordinate (x, y, z) for each sample
x_angles = np.random.uniform(-(2 * np.pi), 2 * np.pi, num_samples) 
y_angles = np.random.uniform(-(2 * np.pi), 2 * np.pi, num_samples)  
z_angles = np.random.uniform(-(2 * np.pi), 2 * np.pi, num_samples)  

# Combine the angles into a single numpy array
inputs = np.array([x_angles, y_angles, z_angles])

######### Generating outputs #########
outputs = np.array([])
x_outputs = np.array([])
y_outputs = np.array([])
z_outputs = np.array([])
for i in range (num_samples):
    # Create a forward kinematics model
    fk = ForwardKinematics([u],[line1_end],"xyz")
    
    # Creating transformation matrices
    T1 = fk.rotation_matrix(inputs[0][i], u)
    # T2 = fk.rotation_matrix(inputs[1][i], v)
    # T3 = fk.rotation_matrix(inputs[2][i], w)
    
    # Transforming the end effector sequentially
    line3_end_transformed = T1 @  fk.line_ends[0]
    # Appending the transformed end effector position to the outputs
    x_outputs = np.append(x_outputs, line3_end_transformed[0])
    y_outputs = np.append(y_outputs, line3_end_transformed[1])
    z_outputs = np.append(z_outputs, line3_end_transformed[2])

# Combine the outputs into a single numpy array
outputs = np.array([x_outputs, y_outputs, z_outputs])


# Training the neural network
for i in range(num_iterations):
    nn = NeuralNetwork(lr,inputs,outputs,w1,w2,num_samples,bias1,bias2)
    z1,a1,z2,a2 = nn.propagate_forward()
    error = nn.mse(a2)
    w1,w2,bias1,bias2 = nn.propagate_backwards(z1,a1,z2,a2)

    loss_values.append(error)
    print("Iteration: ", i, " Loss: ", error)

# Plotting the loss values over time
plt.plot(loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Time')
plt.show()

# Passing the first 10 samples through the neural network to see the predicted points
test_input = np.array([inputs[0,:10],inputs[1,:10],inputs[2,:10]])
test_output = np.array([outputs[0,:10],outputs[1,:10],outputs[2,:10]])
nn = NeuralNetwork(lr,test_input,test_output,w1,w2,1,bias1,bias2)
_,_,_,a2 = nn.propagate_forward()

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_zlim([-3,3])

# Plotting the actual points
ax.scatter3D(outputs[0,:10],outputs[1,:10],outputs[2,:10],color='blue',label='Actual Points')
# Plotting the predicted points
ax.scatter3D(a2[0,:10],a2[1,:10],a2[2,:10],color='red',label='Predicted Points')
plt.title('Actual vs Predicted Points')
plt.legend()
plt.show()


print("The loss after training is: ", loss_values[-1])
