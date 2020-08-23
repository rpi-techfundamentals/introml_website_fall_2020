## Neural Networks 
- This was adopted from the PyTorch Tutorials. 
- http://pytorch.org/tutorials/beginner/pytorch_with_examples.html

## Neural Networks 
- Neural networks are the foundation of deep learning, which has revolutionized the 

```In the mathematical theory of artificial neural networks, the universal approximation theorem states[1] that a feed-forward network with a single hidden layer containing a finite number of neurons (i.e., a multilayer perceptron), can approximate continuous functions on compact subsets of Rn, under mild assumptions on the activation function.```

### Generate Fake Data
- `D_in` is the number of dimensions of an input varaible.
- `D_out` is the number of dimentions of an output variable.
- Here we are learning some special "fake" data that represents the xor problem. 
- Here, the dv is 1 if either the first or second variable is 


# -*- coding: utf-8 -*-
import numpy as np

#This is our independent and dependent variables. 
x = np.array([ [0,0,0],[1,0,0],[0,1,0],[0,0,0] ])
y = np.array([[0,1,1,0]]).T
print("Input data:\n",x,"\n Output data:\n",y)

### A Simple Neural Network 
- Here we are going to build a neural network with 2 hidden layers. 
-

np.random.seed(seed=83832)
#D_in is the number of input variables. 
#H is the hidden dimension.
#D_out is the number of dimensions for the output. 
D_in, H, D_out = 3, 2, 1

# Randomly initialize weights og out 2 hidden layer network.
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
bias = np.random.randn(H, 1)

### Learn the Appropriate Weights via Backpropogation
- Learning rate adjust how quickly the model will adjust parameters. 

# -*- coding: utf-8 -*-

learning_rate = .01
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)

    #A relu is just the activation.
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

#CFully connected 


pred = np.maximum(x.dot(w1),0).dot(w2)

print (pred, "\n", y)


### Hidden Layers are Often Viewed as Unknown
- Just a weighting matrix

#However
w1

w2

# Relu just removes the negative numbers.  
h_relu

