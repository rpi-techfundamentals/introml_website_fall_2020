# Evaluation of Classifiers
Let's assume we have 2 different images, and the output for the second to last layer is the following.  The job of the final layer is to "squish" whatever comes out of the neural network. We are going to look at the differences between a sigmoid and a softmax.


```
          img1    img2
cat	      0.02    -1.42
dog	     -2.49    -3.93
plane	   -1.75    -3.19
fish	    2.07     0.63
building	1.25    -0.19
```

#Let's import some values
import torch
import torch.nn.functional as F
import torch.nn as nn

#Let's put the data into a tensor
predictions = torch.tensor([[0.02, -2.49, -1.75, 2.07, 1.25],
                           [-1.42, -3.93, -3.19, 0.63, -0.19]])
predictions

###  Softmax
A softmax assumes that here that classes are exclusive and probabilities add to 1. 

$softmax(x)_i = \frac{exp(x_i)}{\sum_{j}^{ }exp(x_j))}$

*Check out the excel notebook and you should see that you get the same values. Note that even though the inputs for the softmax are different, they yield the same probability estimates for each class.*


#Here we have to create the softmax layer and then pass the layers to it. 
my_softmax_layer = nn.Softmax(dim=1) #here we have to create the softmax layer and then 
softmax=my_softmax_layer(predictions)
softmax


###  Sigmoid 
This is used for binary classification as a final layer.  For each of the potential classes, the prediction is weighted to a 0/1 without considering the other classes.  This would be appropriate for the case where there could be multiple classes (for example a cat and a dog) in the image.

$S(x)={\frac {1}{1+e^{-x}}}={\frac {e^{x}}{e^{x}+1}}$

*Check out the excel spreadsheet.*


sigmoid=torch.sigmoid(predictions)
sigmoid

###  Evaluating the Results
Note that for the 2 examples, the resulting probabilities were the same.  

However, note that the negative values for the final layer predictions suggest that maybe there are multiple items in image one and maybe just a fish in image 2.

*MEAN SQUARED ERROR (MSE)*

${MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}$

predictions = torch.tensor([[0.02, -2.49, -1.75, 2.07, 1.25],
                           [-1.42, -3.93, -3.19, 0.63, -0.19]], requires_grad=True)
truth = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0]], requires_grad=False)

mse_loss=F.mse_loss(torch.sigmoid(predictions), truth )
print( "mse", mse_loss)


###  Exercise

  
1. Evaluate the loss function (MSE) for the softmax output.

2. Change the truth as well as the predictions above and notice the impact on the loss.

This exercise was adopted from the Fast.ai example. 