import numpy as np
import pandas as pd
import datetime
import read_data
from prettytable import PrettyTable

# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
def train(L,X,y):
    (n,d)=np.shape(X)
    theta = np.zeros((d, 1))
    for i in range(0, L):
	X, y = shuffle_in_unison(X,y)
        for t in range(0, n):
            if (y[t] * (np.dot(X[t],  theta))[0]) <= 0:
                theta = theta + np.array([y[t]* X[t]]).T

    return theta

def shuffle_in_unison(a,b):
	assert len(a)==len(b)
	shuffled_a = np.empty(a.shape, dtype=a.dtype)
	shuffled_b = np.empty(b.shape, dtype=b.dtype)
	permutation = np.random.permutation(len(a))
	for old_index, new_index in enumerate(permutation):
		shuffled_a[new_index] = a[old_index]
		shuffled_b[new_index] = b[old_index]
	return shuffled_a, shuffled_b

# Input: numpy vector theta of d rows, 1 column
#        numpy vector x of d rows, 1 column
# Output: label (+1 or -1)
def linpred(theta,x):
    return 1 if np.dot(theta,x) > 0 else -1

def run():
	t = PrettyTable(['# of iterations', '# of mistakes', 'Date included?'])
	iteration_number = 10000
	for a in range(0,2):		
		#Reading the data from the training set with all features included
		features_matrix_all, labels_matrix_all = read_data.read_training_data_all()
	
		#Running our linear perceptron on 10,000/100,000 iterations
		theta = train(iteration_number, features_matrix_all, labels_matrix_all)
	
		#Testing convergence by seeing if the perceptron will make any mistakes predicting on the same data it trained on
		mistakes_count = 0
		for i in range(0, len(features_matrix_all)):
			xii = np.matrix(features_matrix_all[i])
			ai = linpred(xii, theta)
			bi = labels_matrix_all[i]
			if(ai != bi):
				mistakes_count+=1
			
		t.add_row([iteration_number, mistakes_count, "Yes"])
	
	for b in range(0,2):
		#Reading the data from the training set with no date/time feature
		features_matrix_no_time, labels_matrix_no_time = read_data.read_training_data_no_time()
		
		#Running our linear perceptron with no date on 10,000/100,000 iterations
		theta = train(iteration_number, features_matrix_no_time, labels_matrix_no_time)
		
		#Testing convergence by seeing if the perceptron will make any mistakes predicting on the same data it trained on
		mistakes_count = 0
		for i in range(0, len(features_matrix_no_time)):
			xii = np.matrix(features_matrix_no_time[i])
			ai = linpred(xii, theta)
			bi = labels_matrix_no_time[i]
			if(ai != bi):
				mistakes_count+=1
		
		t.add_row([iteration_number, mistakes_count, "No"])	
	print t

if __name__=='__main__':
	run()
