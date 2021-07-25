import random

def random_split(X : list, Y : list, lenght : int) -> list:
    '''
    usage:
    with this function you can seprate dataset into 2 parts with sepecified lenght
    this function chooses members randomly and seprate them

    Example:

    Inputs:
    X = [1,2,3,4,5]
    Y = [2,3,4,5,6]
    lenght = 3

    Outputs:
    X1 = [3,1,5]
    Y1 = [4,2,6]

    X2 = [2,4]
    Y2 = [3,5]

    '''        
    members : list = [0] * lenght
    count : int = 0
    index : int
    m : int = len(X)

    while(count != lenght):
            
        index = random.randint(0, m - 1)
            
        if(index not in members):
            members[count] = index
            count += 1
        
    X1 : list = [X[i] for i in members]
    Y1 : list = [Y[i] for i in members]

    X2 : list = [X[i] for i in range(m) if i not in members]
    Y2 : list = [Y[i] for i in range(m) if i not in members]

    return [X1, Y1], [X2, Y2]


class model:
    def __init__(self, degree : int = 1) -> None:
        '''
        initializing our model
        degree, is degree of our model equation
        default value of degree is 1
    


        our function is h(x):
        h(x) = (θ0) + (θ1 * x) + (θ2 * x ^ 2) + ...
        

               degree
        h(x) = Σ        θi * x ^ i
               i = 0
        

    
        we want to fit our function to a dataset
        

        Trainig set = (X,Y)

        X = {x1, x2, x3, ...}
        Y = {y1, y2, y3, ...}


        xi ϵ X
        yi ϵ Y

        h(xi) = yi

        our mission is to find out best weights for our function
        weights = {θ0, θ1, θ2, ...}

        comment:
        this model is used for function wich has 1 input and 1 output  {that means we can plot it in (x,y) Coordinate plane}
        '''


        if(degree % 1 != 0):
            raise ValueError("degree must be integer")
        if(degree < 0):
            raise ValueError("degree must be posetive or zero")
        
        self.weights : list = [.0] * (degree + 1)
        #   ---  Default Weights are [.0 .0 .0 .0 ...]
        '''
        you can give your own weights too!
        
        example:

        my_model = Model(3)
        my_model.weights = [1, 2, 3, 4] ### here is some example weights and degree
    
        '''
    
    def load(self, path : str) -> None:
        '''
        you can load your model weights too!
        model weights can be saved in each file with any formats:

        a sample model.txt for model file:

        0.0
        1.5
        7.2
        8.3

        you can load your own weights to any files and load it as your model
        this feature helps you to load pre-trained models and weights
        '''
        data : str
        with open(path, 'r') as f:
            data = f.read()
        
        data = list(map(float, data.split()))
        self.weights = data

    def save(self, path : str) -> None:
        '''
        you can save your model after train
        and load it again later

        you save your trained model and use it anywhere you want
        
        model will save in {path} address and it can be in any formats

        a sample model.txt for model file:

        0.0
        1.5
        7.2
        8.3

        '''

        with open(path, 'w') as f:
            for i in range(len(self.weights)):

                f.write(f"{self.weights[i]} ")
    
    def predict(self, x) -> float:
        '''
        this function use our (loaded or trained or ...) weights and calculate h(x)

        our function is actually h(x):
        h(x) = (θ0) + (θ1 * x) + (θ2 * x ^ 2) + ...
        

               degree
        h(x) = Σ        θi * x ^ i
               i = 0
        

        and our weights are:
        weights = {θ0, θ1, θ2, ...}
        '''
        result : float = .0

        for i in range(len(self.weights)):
            result += self.weights[i] * x ** i
        
        return result

    def loss(self, X : list, Y : list) -> float:
        '''
        This function calculate our model errors
        if our model predict better and with high accuracy our loss function return smaller numbers

        here is how loss function mathematically works:

               degree
        h(x) = Σ        θi * x ^ i
               i = 0
        h(x) is our prediction function wich we want to have high accuracy
        
        our mission is finding best weights to reach minimum value of loss function
        or it better to say our mission is minimize loss function

        Trainig set = (X,Y)

        X = {x1, x2, x3, ...}
        Y = {y1, y2, y3, ...}

        m = lenght of X and Y = |X| = |Y|

        Our loss function is Mean squared error (MSE) and here is J function:

                                         m
        J(θ0, θ1, θ2, ...) = 1/(2*m) *   Σ  (h(xi) - yi)^2
                                         i=1
        
        this function uses X,Y as inputs to calculate MSE

        This function round loss number to avoid errors
        '''
        m : int = len(X)
        result : float = .0

        for i in range(m):
            result += (self.predict(X[i]) - Y[i]) ** 2
        
        result /= 2 * m

        return round(result, 6)
    
    def loss_derivate(self, X : list, Y : list, index : int):
        '''
        we have loss function wich calculate our loss

        here is how loss function mathematically works:

               degree
        h(x) = Σ        θi * x ^ i
               i = 0
        h(x) is our prediction function wich we want to have high accuracy
        
    
        Trainig set = (X,Y)

        X = {x1, x2, x3, ...}
        Y = {y1, y2, y3, ...}

        m = lenght of X and Y = |X| = |Y|

        Our loss function is Mean squared error (MSE) and here is J function:

                                         m
        J(θ0, θ1, θ2, ...) = 1/(2*m) *   Σ  (h(xi) - yi)^2
                                         i=1
        
   
        
        
        ∂J(θ0, θ1, θ2, ...)/ ∂θi is partial derivate of J by θi
        we calculate partial derivate of J for all J inputs:
        
                                            m
        ∂J(θ0, θ1, θ2, ...)/ ∂θj =  1/m *   Σ  (h(xi) - yi) * (xi) ^ j
                                            i=1

        this function is partial derivate of J function
        
        we will use this to minimize loss function in train function
        This function round loss number to avoid errors


        '''
        result : float = .0
        m : int = len(X)

        for i in range(m):
            result += (self.predict(X[i]) - Y[i]) * (X[i] ** index)
        
        result /= m

        return round(result, 6)

    def train(self, training_set : tuple, learning_rate : float, epochs : int, batch_size : int = None, validation_set : tuple = None, validation_split : float = None, logging : bool = True) -> list:
        '''
        we have loss function wich calculate our loss

        here is how loss function mathematically works:

               degree
        h(x) = Σ        θi * x ^ i
               i = 0
        h(x) is our prediction function wich we want to have high accuracy
        
    
        Trainig set = (X,Y)

        X = {x1, x2, x3, ...}
        Y = {y1, y2, y3, ...}

        m = lenght of X and Y = |X| = |Y|

        Our loss function is Mean squared error (MSE) and here is J function:

                                         m
        J(θ0, θ1, θ2, ...) = 1/(2*m) *   Σ  (h(xi) - yi)^2
                                         i=1
        
        we want to minimize our loss function(J)
        so we need {θ0, θ1, θ2, ...} wich the J function output is minimum

        gradient descent algorithm is our soloution
        wich we update θi in this way:

        θj = θj - a * ∂J(θ0, θ1, θ2, ...)/ ∂θi
        
        ∂J(θ0, θ1, θ2, ...)/ ∂θi is partial derivate of J by θi

        a is number that we called learning rate
        if learning rate is big number we will have divergence and loss number will increase
        if it is too small trainig action will be slow


        we calculate partial derivate of J for all J inputs:
        
                                            m
        ∂J(θ0, θ1, θ2, ...)/ ∂θj =  1/m *   Σ  (h(xi) - yi) * (xi) ^ j
                                            i=1

        this function is partial derivate of J function
        
        this algorithm works perfectly for models with 1 degree and will find global minimum of our loss function
        but for higher degree it may find the local minimum of loss function and maybe its not the best function we can have.



        Inputs:

        training_set = (X,Y)                                                                               --- Mandatory 
        X,Y are our dataset wich we want to fit function on them
        X and Y are 1d arrays(in python list) they are numbers(int or float) and have same member counts
        in the best performance our function(h) has X as subset of domain and Y as subset of range
        

        learning_rate:   float                                                                            --- Mandatory
        it shows that how much your steps big in learning
        if learning rate is big number we will have divergence and loss number will increase
        if it is too small trainig action will be slow
        choosing this number is very important in training action

        epochs : int                                                                                      --- Mandatory
        our model will be trained in {epochs} times
        for example, if epochs is 10 our model will be trained 10 times

        batch_size: int                                                                                   --- Optional
        we train {batch_size} count of labels from dataset in each epoch
        m = lenght of X and Y = |X| = |Y|
        0 < batch_size < m
        default bachsize is equal to m

        validation_set = (X,Y)                                                                            --- Optional
        X, Y wont be used in trainig term
        X, Y are used to calculate loss on new data
        this loss is called as validation loss
        we have not validation_set by default

        validation_split : float                                                                          --- Optional
        if you dont have your own validation set you can use validation_split
        0 < validation_split < 1
        this percent of data will be choosed randomly for validation_set and it wont be used in training
        we have not validation_split by default

        logging : bool
        the default value is True
        if it is True it will print each epoch on screen


        Output:
        if you have not validation_set or validation_split as an input your output will be like this:
        
        [
            {'loss' : 4241},
            {'loss' : 3533},
            {'loss' : 2323},
            .
            .
            .
        ]
        it shows loss in each epoch
        example:
        epoch 0 -> loss[0]
        
        
        else -> your output will be like this :
        [
            {'loss' : 4241, 'val_loss' : 7722},
            {'loss' : 3533, 'val_loss' : 5322},
            {'loss' : 2323, 'val_loss' : 4322},
            .
            .
            .
        ]
        it shows loss in each epoch
        example:
        epoch 0 -> loss[0]
        
        
        '''
        X, Y  = training_set
        validation : bool = False


        if(validation_set != None):
            validation_X, validation_Y = validation_set
            validation = True

        elif(validation_split != None):
            
            if(not(validation_split < 1 and validation_split > 0)):
                raise ValueError("validation_split must be between 0 and 1")
            
            validation_set_lenght : int = int(len(X) * validation_split)
            
            validation_set, training_set = self.random_split(X, Y, validation_set_lenght)
            
            X, Y  = training_set
            validation_X, validation_Y = validation_set
            
            validation = True
        else:
            pass
        
        
        if(batch_size != None):
            if(batch_size > len(X) or batch_size < 0):
                batch_size = None
        
        loss : float
        validation_loss : float

        new_weights = self.weights[:]
        losses : int = []

        for epoch in range(epochs):
            
            
            if(batch_size != None):
                X_epoch, Y_epoch = self.random_split(X, Y, batch_size)[0]
            else:
                X_epoch, Y_epoch = X, Y

            for i in range(len(self.weights)):
            
                new_weights[i] -= learning_rate * self.loss_derivate(X_epoch, Y_epoch, i)
            
            self.weights = new_weights[:]

            loss = self.loss(X, Y)
            
            if(logging):
                if(validation):
                    validation_loss = self.loss(validation_X, validation_Y)
                    losses.append({"loss" : loss, "val_loss": validation_loss})
                    print(f"epoch : {epoch}\t\tloss : {loss}\t\tval_loss : {validation_loss}")
                    
                else:
                    losses.append({"loss" : loss})
                    print(f"epoch : {epoch}\t\tloss : {loss}")