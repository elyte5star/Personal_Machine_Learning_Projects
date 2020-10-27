import numpy as np
import random
import sys
from os import system, name 
from time import sleep
class ANN():

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, no_hidden_layers=1, hidden_layer_size=28,output_size=1,max_iterations=20, learning_rate=0.1):
        self.no_inputs = no_inputs
        self.no_hidden_layers = no_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        prev_nodes = self.no_inputs
        self.hidden_weights = list()
        for k in range(self.no_hidden_layers):
            self.hidden_weights.append((2*np.random.random((self.hidden_layer_size,prev_nodes))-1)/prev_nodes)
            prev_nodes = hidden_layer_size + 1
        self.output_weights = (2*np.random.random((self.output_size,prev_nodes))-1)/prev_nodes
        self.layer_output=list()
        self.hidden_gradients = list()
        self.output_gradient=0
        self.output_error=0
       
    def feed_forward(self,data,label):
        last_output = data
        self.layer_output=list()
        for k in range(self.no_hidden_layers):
            self.layer_output.append(np.hstack((np.ones((1)),self.activate(np.dot(last_output,self.hidden_weights[k].T)))))
            last_output = self.layer_output[k]
        self.layer_output.append(self.activate(np.dot(last_output,self.output_weights.T)))
        self.output_error = np.array([self.layer_output[self.no_hidden_layers]-label])
        self.output_gradient = self.layer_output[self.no_hidden_layers-1][:,np.newaxis] * self.output_error

    def backpropagation(self,data,label):
        self.hidden_gradients = list()
        next_layer_error = self.output_error
        next_weight = self.output_weights[:,1:]
        for l in range(self.no_hidden_layers,0,-1):
            temp_layer = l - 1
            hidden_error = self.layer_output[temp_layer][1:] * (1 - self.layer_output[temp_layer][1:]) * np.dot(next_layer_error, next_weight)   
            m = self.layer_output[temp_layer-1]
            if temp_layer == 0:
                m = data
            self.hidden_gradients.append(m[:,np.newaxis]*hidden_error[np.newaxis,:])
            next_layer_error = hidden_error
            next_weight = self.hidden_weights[temp_layer][:,1:]

    def predict(self,data):
        initial_output = data
        layer_output=list()
        for k in range(self.no_hidden_layers):
            layer_output.append(np.hstack((np.ones((1)),self.relu(np.dot(initial_output,self.hidden_weights[k].T)))))
            initial_output = layer_output[k]
        return self.activate(np.dot(initial_output,self.output_weights.T))    
    #===================================#
    # Performs the activation function. #
    # Expects an array of values of     #
    # shape (1,N) where N is the number #
    # of nodes in the layer.            #
    #===================================#
    def activate(self, a):
        return 1/(1 + np.exp(-a))
    #===============================#
    # Trains the net using labelled #
    # training data.                #
    #===============================#
    def train(self, training_data, labels):
        assert len(training_data) == len(labels)
        both = list(zip(training_data, labels))
        for i in range(self.max_iterations):
            random.shuffle(both)
            for data,label in both:
                self.feed_forward(data,label)
                self.backpropagation(data,label)
                self.update_weights()   
        return
        
    def update_weights(self):
        self.output_weights = self.output_weights - self.learning_rate*self.output_gradient.T
        for k in range(self.no_hidden_layers):
            self.hidden_weights[k] = self.hidden_weights[k] - self.learning_rate * self.hidden_gradients[self.no_hidden_layers -k -1].T[:,:,0]
    #=========================================#
    # Tests the prediction on each element of #
    # the testing data. Prints the precision, #
    # recall, and accuracy.                   #
    #=========================================#
    def test(self, testing_data, labels):
        assert len(testing_data) == len(labels)
        True_positive,False_negative,False_positive,True_negative=0.0,0.0,0.0,0.0
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        for data,label in zip(testing_data,labels):
            prediction = self.predict(data)
            if prediction > 0.5 and label == True:
                True_positive+=1
            elif prediction < 0.5 and label == False:
                True_negative+=1
            elif prediction > 0.5 and label == False:
                False_positive+=1
            elif prediction < 0.5 and label == True:
                False_negative+=1
            #accuracy+=1-abs(label-prediction)
        precision = True_positive/(True_positive + False_positive)
        recall = True_positive/(True_positive + False_negative)
        accuracy=(True_positive +True_negative)/(True_negative+True_positive + False_positive + False_negative)
        #accuracy/=len(labels)
        print("Accuracy: ",np.round(accuracy,2))
        print("Precision: ",np.round(precision,2))
        print("Recall: ", np.round(recall,2))

   #PART 5
    def relu(self,X):
        return np.maximum(0,X)
    
    def feed_forward_REL(self,data,label):
        last_output = data
        self.layer_output=list()
        for k in range(self.no_hidden_layers):
            self.layer_output.append(np.hstack((np.ones((1)),self.relu(np.dot(last_output,self.hidden_weights[k].T)))))
            last_output = self.layer_output[k]
        self.layer_output.append(self.relu(np.dot(last_output,self.output_weights.T)))
        self.output_error = np.array([self.layer_output[self.no_hidden_layers]-label])
        self.output_gradient = self.layer_output[self.no_hidden_layers-1][:,np.newaxis] * self.output_error
  
    """
    def dRelu(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    """

    def backpropagation_rel(self,data,label):
        self.hidden_gradients = list()
        next_layer_error = self.output_error
        next_weight = self.output_weights[:,1:]
        for l in range(self.no_hidden_layers,0,-1):
            temp_layer = l - 1
            m = self.layer_output[temp_layer][1:] > 0
            b = 1*m #convert bool
            hidden_error = b*np.dot(next_layer_error, next_weight)
            y = self.layer_output[temp_layer-1]
            if temp_layer == 0:
                y = data
            self.hidden_gradients.append(y[:,np.newaxis]*hidden_error[np.newaxis,:])
            next_layer_error = hidden_error
            next_weight = self.hidden_weights[temp_layer][:,1:]


    def predict_rel(self,data):
        initial_output = data
        layer_output=list()
        for k in range(self.no_hidden_layers):
            layer_output.append(np.hstack((np.ones((1)),self.relu(np.dot(initial_output,self.hidden_weights[k].T)))))
            initial_output = layer_output[k]
        return self.relu(np.dot(initial_output,self.output_weights.T))    
