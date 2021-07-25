from numpy import random
import AI
import numpy as np

'''
this is sample usage for AI library
'''

# --- Build some dataset
'''
here we want to build a dataset for our model
'''
    # *** build random data set
'''
with this function you can generate random dataset with your own size
all of the labels in this dataset are float and between 100 and 1000
'''
def generate_dataset(lenght):
    X = np.random.rand(lenght) * 1000
    Y = np.random.rand(lenght) * 1000
    for i in range(len(X)):
        X[i] = round(X[i], 2)
        Y[i] = round(Y[i], 2)
    return X,Y

    # *** build ordered dataset
'''
here I build ordered dataset wich have rule inside it
I want to use this function to build my dataset

a = random.randint(1,100)
b = random.randint(1,100)
c = random.randint(1,100)

I choose random Coefficient for my equation
and my equation is:

a + bx + cx^2

also I will choose 20 numbers for X and they are between 1,20
'''
a,b = random.randint(1,100), random.randint(1,100)

X = list(range(1,21))
Y = [a + b * x for x in X]



# --- Building Model
'''
examples:

if degree = 3 we'll have this equation:
    a + bx + cx^2 + dx^3

if degree = 1 we'll have this equation:
    a + bx

'''
my_model = AI.model(degree = 1)


# --- Train the Model
'''
in trainig level our model will search for the best
weights (a,b,c, ...)

and will find this function:
h(x) = a + bx + cx^2 + dx^3 + ...

to fit on dataset:
X -> h(x) -> Y

input parameters:
trainig set : (X,Y)    --- mandatory

learning rate : int    --- mandatory   
it is size of our steps in our alorithm we have better accuracy if we choose it small
if it is too big our model will get to divergence and we'll get overflow error

epochs: int            --- mandatory
your model will train for {epochs} times

batch_size:            --- optional 
in each epoch your model will train on just bach size of dataset
it means if batch size = 4 our model will train on only 4 labels in each epoch(this 4 labels will be choosed randomly)
the default number of batch size is equal to count of labels in all of the dataset 

validation_set:  (X,Y) --- optional
if you have your own validation set you can use it to see validation loss
validation set wont be used in training and it will be used to calculate our model loss in new inputs
by default we have not any validation_set and validation_loss

validation_split: float --- optional
if you dont have your own validation set you can use validation_split
0 < validation_split < 1
this percent of data will be choosed randomly for validation_set and it will not in training


logging: bool          --- optional
if it's True you can see result of model in each epochs in terminal
by default logging = True


train function will return a list that says loss in each epochs
like this:
[
{'loss' : 100},
{'loss' : 90},
{'loss' : 80}
.
.
.
]
'''


my_model.train(training_set = (X, Y),
               learning_rate = 10 ** -3,
               epochs = 10 ** 4 * 5
)

my_model.save('model.txt')

# --- use model
"""
after trainig you can access to weights in a array
[a,b,c,d, ...]
in this variable:
my_model.weights

you use your model to predict x(simple input) in this way
result = my_model.predict(x)
"""


