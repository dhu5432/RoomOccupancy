import numpy as np
from random import randint
import time
import read_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
def logistic_regression(X, y):
	clf = LogisticRegression()
	return clf.fit(X,np.ravel(y))

def neural_network(L, X, y):	
	activation = ['identity', 'logistic', 'tanh', 'relu']
	solver = ['lbfgs', 'sgd', 'adam']
	alpha = [0.0001, 0.001, 0.01, 0.1, 0.00001]
	learning_rate = ['constant', 'invscaling', 'adaptive']
	

	args = [activation[randint(0, len(activation)-1)], solver[randint(0, len(solver)-1)], alpha[randint(0, len(alpha)-1)], learning_rate[randint(0, len(learning_rate)-1)]]
	
	mlp = MLPClassifier(hidden_layer_sizes=(1),activation=args[0], solver=args[1], alpha = args[2], learning_rate=args[3],  max_iter = L)
	return mlp.fit(X, y), args
	



if __name__=='__main__':
	features_matrix, labels_matrix = read_data.read_training_data()
	testFeatures_matrix, testLabels_matrix = read_data.read_testing_data1()
	for i in range(0, 100):
		model, args = neural_network(1000, features_matrix, np.ravel(labels_matrix))
		yHat = model.predict(testFeatures_matrix)
		count = 0
		for a in range(0, len(yHat)):
			if yHat[a] == testLabels_matrix[a]:
				count+=1
	
		F = open("resultsNN.txt","a")
		F.write("{0}, {1}".format(1.0*count/len(yHat), args))
		F.write("\n")
		F.close()
