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
a = np.array([6,1,3])
line4_end = np.array([8,1,4])
b = np.array([8,1,4])
line5_end = np.array([10,1,0])
c = np.array([10,1,0])
line6_end = np.array([12,1,1])
d = np.array([12,1,1])
line7_end = np.array([14,1,0])
e = np.array([14,1,0])
line8_end = np.array([16,1,3])
f = np.array([16,1,3])
line9_end = np.array([18,1,4])
g = np.array([18,1,4])
line10_end = np.array([20,1,0])
h = np.array([20,1,0])
line11_end = np.array([22,1,1])

# Number of samples,inputs and outputs
num_samples = 2500
nInputs = 3
nOutputs = 1
num_dof = 1

# Number of iterations
num_iterations = 10000

# Learning rate
lr = 0.4

# Initialising an empty list to store the loss values
loss_values = []
final_loss_after_training = []
num_neurons_list = []
num_dof_list = []

# Neurons in hidden layer
num_neurons = 22

#£ Initialising weights randomly for three inputs, four neurons and three outputs
w1 = np.random.rand(num_neurons,nInputs) # Initialize weights for hidden layer
w2 = np.random.rand(nOutputs,num_neurons) # Initialize weights for output layer

## Initialising biases randomly between -0.5 and0.5
bias1 = np.random.uniform(-0.5,0.5,(num_neurons, 1))  # Initialize bias for hidden layer
bias2 = np.random.uniform(-0.5,0.5,(nOutputs, 1)) 

######### Generating desired outputs #########

# Generate random angles in radians for each axis 
u_angles = np.random.uniform(-np.pi, np.pi, num_samples) 
v_angles = np.random.uniform(-np.pi, np.pi, num_samples)  
w_angles = np.random.uniform(-np.pi, np.pi, num_samples)
a_angles = np.random.uniform(-np.pi, np.pi, num_samples)  
b_angles = np.random.uniform(-np.pi, np.pi, num_samples)  
c_angles = np.random.uniform(-np.pi, np.pi, num_samples)  
d_angles = np.random.uniform(-np.pi, np.pi, num_samples)  
e_angles = np.random.uniform(-np.pi, np.pi, num_samples)
f_angles = np.random.uniform(-np.pi, np.pi, num_samples) 
g_angles = np.random.uniform(-np.pi, np.pi, num_samples)
h_angles = np.random.uniform(-np.pi, np.pi, num_samples)

# Combine the angles into a single numpy array
outputs = np.array([u_angles,v_angles,w_angles,a_angles,b_angles,c_angles,d_angles,e_angles,f_angles,g_angles,h_angles])

######### Generating inputs #########
inputs1 = np.array([])
inputs2 = np.array([])
inputs3 = np.array([])
inputs4 = np.array([])
inputs5 = np.array([])
inputs6 = np.array([])
inputs7 = np.array([])
inputs8 = np.array([])
inputs9 = np.array([])
inputs10 = np.array([])
inputs11 = np.array([])
x_inputs1 = np.array([])
y_inputs1 = np.array([])
z_inputs1 = np.array([])
x_inputs2 = np.array([])
y_inputs2 = np.array([])
z_inputs2 = np.array([])
x_inputs3 = np.array([])
y_inputs3 = np.array([])
z_inputs3 = np.array([])
x_inputs4 = np.array([])
y_inputs4 = np.array([])
z_inputs4 = np.array([])
x_inputs5 = np.array([])
y_inputs5 = np.array([])
z_inputs5 = np.array([])
x_inputs6 = np.array([])
y_inputs6 = np.array([])
z_inputs6 = np.array([])
x_inputs7 = np.array([])
y_inputs7 = np.array([])
z_inputs7 = np.array([])
x_inputs8 = np.array([])
y_inputs8 = np.array([])
z_inputs8 = np.array([])
x_inputs9 = np.array([])
y_inputs9 = np.array([])
z_inputs9 = np.array([])
x_inputs10 = np.array([])
y_inputs10 = np.array([])
z_inputs10 = np.array([])
x_inputs11 = np.array([])
y_inputs11 = np.array([])
z_inputs11 = np.array([])

for i in range (num_samples):
    # Create a forward kinematics model
    fk = ForwardKinematics([u,v,w,a,b,c,d,e,f,g,h],[line1_end,line2_end,line3_end,line4_end,line5_end,line6_end,line7_end,line8_end,line9_end,line10_end,line11_end],"xyz")
    
    # Creating transformation matrices
    T1 = fk.rotation_matrix(outputs[0][i], u)
    T2 = fk.rotation_matrix(outputs[1][i], v)
    T3 = fk.rotation_matrix(outputs[2][i], w)
    T4 = fk.rotation_matrix(outputs[3][i], a)
    T5 = fk.rotation_matrix(outputs[4][i], b)
    T6 = fk.rotation_matrix(outputs[5][i], c)
    T7 = fk.rotation_matrix(outputs[6][i], d)
    T8 = fk.rotation_matrix(outputs[7][i], e)
    T9 = fk.rotation_matrix(outputs[8][i], f)
    T10 = fk.rotation_matrix(outputs[9][i], g)
    T11 = fk.rotation_matrix(outputs[10][i], h)

    
    # Transforming the end effector sequentially
    endeffpos1 = T1 @ fk.line_ends[0]
    endeffpos2 = T1 @ T2 @ fk.line_ends[0]
    endeffpos3 = T1 @ T2 @ T3 @ fk.line_ends[0]
    endeffpos4 = T1 @ T2 @ T3 @ T4 @ fk.line_ends[0]
    endeffpos5 = T1 @ T2 @ T3 @ T4 @ T5 @ fk.line_ends[0]
    endeffpos6 = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ fk.line_ends[0]
    endeffpos7 = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ fk.line_ends[0]
    endeffpos8 = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T8 @ fk.line_ends[0]
    endeffpos9 = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T8 @ T9 @ fk.line_ends[0]
    endeffpos10 = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T8 @ T9 @ T10 @ fk.line_ends[0]
    end_effector_pos = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T8 @ T9 @ T10 @ T11 @ fk.line_ends[0]


    # Appending the transformed end effector position to the inputs
    x_inputs1 = np.append(x_inputs1, endeffpos1[0])
    y_inputs1 = np.append(y_inputs1, endeffpos1[1])
    z_inputs1 = np.append(z_inputs1, endeffpos1[2])

    x_inputs2 = np.append(x_inputs2, endeffpos2[0])
    y_inputs2 = np.append(y_inputs2, endeffpos2[1])
    z_inputs2 = np.append(z_inputs2, endeffpos2[2])

    x_inputs3 = np.append(x_inputs3, endeffpos3[0])
    y_inputs3 = np.append(y_inputs3, endeffpos3[1])
    z_inputs3 = np.append(z_inputs3, endeffpos3[2])

    x_inputs4 = np.append(x_inputs4, endeffpos4[0])
    y_inputs4 = np.append(y_inputs4, endeffpos4[1])
    z_inputs4 = np.append(z_inputs4, endeffpos4[2])

    x_inputs5 = np.append(x_inputs5, endeffpos5[0])
    y_inputs5 = np.append(y_inputs5, endeffpos5[1])
    z_inputs5 = np.append(z_inputs5, endeffpos5[2])

    x_inputs6 = np.append(x_inputs6, endeffpos6[0])
    y_inputs6 = np.append(y_inputs6, endeffpos6[1])
    z_inputs6 = np.append(z_inputs6, endeffpos6[2])

    x_inputs7 = np.append(x_inputs7, endeffpos7[0])
    y_inputs7 = np.append(y_inputs7, endeffpos7[1])
    z_inputs7 = np.append(z_inputs7, endeffpos7[2])

    x_inputs8 = np.append(x_inputs8, endeffpos8[0])
    y_inputs8 = np.append(y_inputs8, endeffpos8[1])
    z_inputs8 = np.append(z_inputs8, endeffpos8[2])

    x_inputs9 = np.append(x_inputs9, endeffpos9[0])
    y_inputs9 = np.append(y_inputs9, endeffpos9[1])
    z_inputs9 = np.append(z_inputs9, endeffpos9[2])

    x_inputs10 = np.append(x_inputs10, endeffpos10[0])
    y_inputs10 = np.append(y_inputs10, endeffpos10[1])
    z_inputs10 = np.append(z_inputs10, endeffpos10[2])

    x_inputs11 = np.append(x_inputs11, end_effector_pos[0])
    y_inputs11 = np.append(y_inputs11, end_effector_pos[1])
    z_inputs11 = np.append(z_inputs11, end_effector_pos[2])

# Creating the inputs arrays
inputs1 = np.array([x_inputs1, y_inputs1, z_inputs1])
inputs2 = np.array([x_inputs2, y_inputs2, z_inputs2])
inputs3 = np.array([x_inputs3, y_inputs3, z_inputs3])
inputs4 = np.array([x_inputs4, y_inputs4, z_inputs4])
inputs5 = np.array([x_inputs5, y_inputs5, z_inputs5])
inputs6 = np.array([x_inputs6, y_inputs6, z_inputs6])
inputs7 = np.array([x_inputs7, y_inputs7, z_inputs7])
inputs8 = np.array([x_inputs8, y_inputs8, z_inputs8])
inputs9 = np.array([x_inputs9, y_inputs9, z_inputs9])
inputs10 = np.array([x_inputs10, y_inputs10, z_inputs10])
inputs11 = np.array([x_inputs11, y_inputs11, z_inputs11])

#Combining all the inputs into one array
dof = np.stack((inputs1,inputs2,inputs3,inputs4,inputs5,inputs6,inputs7,inputs8,inputs9,inputs10,inputs11), axis=0)

###############################################################################################################
#  The following block of code was used to plot the the two graphs in the results section of the dissertation #
#  and was left here for reference.                                                                           #
###############################################################################################################
# Running the neural network for 10000 iterations
# for i in range(num_iterations):
# dof = np.stack((inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11), axis=0)
# num_dof=1
# nOutputs = 1
# for inp in dof:
#     outp = outputs[:num_dof]
#     for i in range(4,50):
#         num_neurons = i
#         # Initialising weights randomly for three inputs, four neurons and three outputs
#         w1 = np.random.rand(num_neurons,nInputs) # Initialize weights for hidden layer
#         w2 = np.random.rand(nOutputs,num_neurons) # Initialize weights for output layer

#         ## Initialising biases randomly between -3 and 3
#         bias1 = np.random.uniform(-0.5,0.5,(num_neurons, 1))  # Initialize bias for hidden layer
#         bias2 = np.random.uniform(-0.5,0.5,(nOutputs, 1)) 
#         error=1
#         for j in range(10000):
#             nn = NeuralNetwork(lr,inp,outp,w1,w2,num_samples,bias1,bias2)
#             z1,a1,z2,a2 = nn.propagate_forward()
#             error = nn.mean_angle_error(a2)
#             w1,w2,bias1,bias2 = nn.propagate_backwards(z1,a1,z2,a2)

#             loss_values.append(error)
#             print("Iteration: ", j, " Loss: ", error)
#         if error<0.43:
#             break
#     # final_loss_after_training.append(loss_values[-1])
#     num_dof_list.append(num_dof)
#     print("###############################################################################################################")
#     num_neurons_list.append(num_neurons) 
#     num_dof+=1
#     nOutputs+=1

# print(num_neurons_list, num_dof_list)
# # Plotting the loss values
# plt.plot(num_dof_list,num_neurons_list, marker='o', linestyle='-')
# plt.xlabel("DOF")
# plt.ylabel("Neurons")
# plt.title("Neurons Needed to Train the Neural Network for Different DOF")
# plt.show()


# # for i in range(num_iterations):
# num_dof=1
# nOutputs = 1
# for inp in dof:
#     outp = outputs[:num_dof]
#     num_neurons = 16
#     # Initialising weights randomly for three inputs, four neurons and three outputs
#     w1 = np.random.rand(num_neurons,nInputs) # Initialize weights for hidden layer
#     w2 = np.random.rand(nOutputs,num_neurons) # Initialize weights for output layer

#     ## Initialising biases randomly between -3 and 3
#     bias1 = np.random.uniform(-0.5,0.5,(num_neurons, 1))  # Initialize bias for hidden layer
#     bias2 = np.random.uniform(-0.5,0.5,(nOutputs, 1))
#     for j in range(100000):
#         nn = NeuralNetwork(lr,inp,outp,w1,w2,num_samples,bias1,bias2)
#         z1,a1,z2,a2 = nn.propagate_forward()
#         error = nn.mean_angle_error(a2)
#         w1,w2,bias1,bias2 = nn.propagate_backwards(z1,a1,z2,a2)

#         loss_values.append(error)
#         print("Iteration: ", j, " Loss: ", error)
#     final_loss_after_training.append(loss_values[-1])
#     num_dof_list.append(num_dof)
#     print("################################################################################################")
#     # num_neurons_list.append(num_neurons) 
#     num_dof+=1
#     nOutputs+=1

# # Plotting the loss values
# plt.plot(num_dof_list,final_loss_after_training)
# plt.xlabel("Robotic Arm DOF")
# plt.ylabel("Loss")
# plt.title("Loss after Training the Neural Network for Different DOF")
# plt.show()

###############################################################################################################


# Training the Neural Netqork for 10000 iterations

inp = inputs1 # Choosing inputs for the first degree of freedom
outp = outputs[:num_dof] # Choosing outputs for the first degree of freedom

for i in range(num_iterations):
    nn = NeuralNetwork(lr,inp,outp,w1,w2,num_samples,bias1,bias2)
    z1,a1,z2,a2 = nn.propagate_forward()
    error = nn.mean_angle_error(a2)
    w1,w2,bias1,bias2 = nn.propagate_backwards(z1,a1,z2,a2)

    loss_values.append(error)
    print("Iteration: ", i, " Loss: ", error)


# Plotting the loss values
plt.plot(loss_values)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.show()

# Passing the first 10 samples through the neural network to see the predicted joint angles
test_input = np.array([inputs1[0,:10],inputs1[1,:10],inputs1[2,:10]])
test_output = np.array([outputs[0,:10]])
nn = NeuralNetwork(lr,test_input,test_output,w1,w2,1,bias1,bias2)
_,_,_,a2 = nn.propagate_forward()

# Putting the preddicted joint angles through my forward kinematics model to get the end effector position
predicted_outputs_x = []
predicted_outputs_y = []
predicted_outputs_z = []
for i in range(10):
    plot_fk = ForwardKinematics([u],[line1_end],"xyz")
    T1 = plot_fk.rotation_matrix(a2[0,i],u)

    plot_endeffpos = T1 @ fk.line_ends[0]
    predicted_outputs_x.append(plot_endeffpos[0])
    predicted_outputs_y.append(plot_endeffpos[1])
    predicted_outputs_z.append(plot_endeffpos[2])

predicted_outputs = np.array([predicted_outputs_x, predicted_outputs_y, predicted_outputs_z])

#creating the axis for the plot
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_zlim([-3,3])

# Plotting the actual points
ax.scatter3D(inputs1[0,:10],inputs1[1,:10],inputs1[2,:10],color='blue',label='Actual Points')
# Plotting the predicted points
ax.scatter3D(predicted_outputs[0,:10],predicted_outputs[1,:10],predicted_outputs[2,:10],color='red',label='Predicted Points')
plt.title('Actual vs Predicted Points')
plt.legend()
plt.show()