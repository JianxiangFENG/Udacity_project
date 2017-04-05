import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab

def MSE(y, Y):
    return np.mean((y-Y)**2)

def load_data(data_path):
	rides = pd.read_csv(data_path)
	rides[:24*10].plot(x='dteday', y='registered')
	return rides

def preprocess_data(rides):
	#get dummy variables
	dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
	for each in dummy_fields:
		dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
		rides = pd.concat([rides, dummies], axis=1)

	fields_to_drop = ['instant', 'dteday', 'season', 'weathersit','weekday', 'atemp', 'mnth', 'workingday', 'hr']
	data = rides.drop(fields_to_drop, axis=1)
	#standlize the continuous variables
	quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
	# Store scalings in a dictionary so we can convert back later
	scaled_features = {}
	for each in quant_features:
		mean, std = data[each].mean(), data[each].std()
		scaled_features[each] = [mean, std]
		data.loc[:, each] = (data[each] - mean)/std

	#Splitting the data into training, testing, and validation sets
	# Save the last 21 days 
	test_data = data[-21*24:]
	data = data[:-21*24]
	# Separate the data into features and targets
	target_fields = ['cnt', 'casual', 'registered']
	features, targets = data.drop(target_fields, axis=1), data[target_fields]
	test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
	n_records = features.shape[0]
	split = np.random.choice(features.index,size=int(n_records*0.8),replace=False)
	train_features, train_targets = features.ix[split], targets.ix[split]
	val_features, val_targets = features.drop(split), targets.drop(split)

	return test_data, scaled_features, train_features, train_targets, val_features, val_targets, test_features, test_targets

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                                (self.hidden_nodes, self.input_nodes))
    
        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.output_nodes, self.hidden_nodes))
        
        self.learning_rate = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # TODO: Activation function is the sigmoid function
        self.activation_function = lambda x:1/(1+np.exp(-x))
        
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden,inputs)# signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)#10*1 signals from hidden layer
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output,hidden_outputs)# signals into final output layer
        final_outputs = final_inputs# signals from final output layer
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error
        output_errors = targets - final_outputs#1 Output layer error is the difference between desired target and actual output.
        
        # TODO: Backpropagated error
        hidden_errors = output_errors*self.weights_hidden_to_output#1*10 errors propagated to the hidden layer
        hidden_grad = hidden_errors.T*hidden_outputs*(1-hidden_outputs)#10*1 hidden layer gradients
        
        # TODO: Update the weights
        self.weights_hidden_to_output += self.learning_rate*output_errors*hidden_outputs.T#1*10 update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.learning_rate*hidden_grad*inputs.T#10*56 update input-to-hidden weights with gradient descent step
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden,inputs)# signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)# signals from hidden layer
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output,hidden_outputs) # signals into final output layer
        final_outputs = final_inputs# signals from final output layer
        
        return final_outputs

if __name__ == '__main__':
	epochs = 700
	learning_rate = 0.03
	hidden_nodes = 10
	output_nodes = 1
	
	data_path = 'bike-sharing-dataset/hour.csv'
	rides = load_data(data_path)

	test_data, scaled_features, train_features, train_targets, val_features, val_targets, test_features, test_targets = preprocess_data(rides)

	N_i = train_features.shape[1]
	network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
	losses = {'train':[], 'validation':[]}
	for e in range(epochs):
    	# Go through a random batch of 128 records from the training data set
		batch = np.random.choice(train_features.index, size=128)
		for record, target in zip(train_features.ix[batch].values,train_targets.ix[batch]['cnt']):
			network.train(record, target)
    
		if e%(epochs/10) == 0:
			# Calculate losses for the training and test sets
			train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
			val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
			losses['train'].append(train_loss)
			losses['validation'].append(val_loss)

			# Print out the losses as the network is training
			print('Training loss: {:.4f}'.format(train_loss))
			print('Validation loss: {:.4f}'.format(val_loss))

	plt.plot(losses['train'], label='Training loss')
	plt.plot(losses['validation'], label='Validation loss')
	plt.legend()

	
	fig, ax = plt.subplots(figsize=(8,4))
	mean, std = scaled_features['cnt']
	predictions = network.run(test_features)*std + mean
	ax.plot(predictions[0], label='Prediction')
	ax.plot((test_targets['cnt']*std + mean).values, label='Data')
	ax.set_xlim(right=len(predictions))
	ax.legend()

	dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
	dates = dates.apply(lambda d: d.strftime('%b %d'))
	ax.set_xticks(np.arange(len(dates))[12::24])
	_ = ax.set_xticklabels(dates[12::24], rotation=45)
	pylab.show()
	pylab.ion()