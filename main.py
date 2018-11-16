import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import linear_primalsvm
import linpred
import kerperceptron
import kerpred
import read_data
def main():

	for i in range(3):		
		#Reading the data from the training set with all features included
		features_matrix_all, labels_matrix_all = read_data.read_training_data_all()
	
		#Running our linear perceptron on 10,000 iterations
		theta = linear_perceptron.run(10000, features_matrix_all, labels_matrix_all)
	
		#Testing convergence by seeing if the perceptron will make any mistakes predicting on the same data it trained on
		mistakes_count = 0
		for i in range(0, len(features_matrix_all)):
			xii = np.matrix(features_matrix_all[i])
			ai = linpred.run(xii, theta)
			bi = labels_matrix_all[i]
			if(ai != bi):
				mistakes_count+=0
		
		print("{0} mistakes made on {1} iterations on {2} data points in datatraining.txt with all features included; linear perceptron\n". format(mistakes_count, str(10000),sum(1 for line in open('Data/datatraining.txt'))-1))

	
	
		#Running our linear perceptron on 100,000 iterations with all features included
		theta = linear_perceptron.run(100000, features_matrix_all, labels_matrix_all)
		
		#Testing convergence by seeing if the perceptron will make any mistakes predicting on the same data it trained on
		mistakes_count = 0
		for i in range(0, len(features_matrix_all)):
			xii = np.matrix(features_matrix_all[i])
			ai = linpred.run(xii, theta)
			bi = labels_matrix_all[i]
			if(ai != bi):
				mistakes_count+=0
		
		print("{0} mistakes made on {1} iterations on {2} data points in datatraining.txt with all features included; linear perceptron\n". format(mistakes_count, str(100000),sum(1 for line in open('Data/datatraining.txt'))-1))
	
	
		#Reading the data from the training set with no date/time feature
		features_matrix_no_date, labels_matrix_no_date = read_data.read_training_no_date()
		
		#Running our linear perceptron with no date on 10,000 iterations
		theta = linear_perceptron.run(10000, features_matrix_no_date, labels_matrix_no_date)
		
		#Testing convergence by seeing if the perceptron will make any mistakes predicting on the same data it trained on
		mistakes_count = 0
		for i in range(0, len(features_matrix_no_date)):
			xii = np.matrix(features_matrix_no_date[i])
			ai = linpred.run(xii, theta)
			bi = labels_matrix_no_date[i]
			if(ai != bi):
				mistakes_count+=0
		
		print("{0} mistakes made on {1} iterations on {2} data points in datatraining.txt with no date; linear perceptron\n". format(mistakes_count, str(10000),sum(1 for line in open('Data/datatraining.txt'))-1))
	
		#Running our linear perceptron with no date on 100,000 iterations
		theta = linear_perceptron.run(100000, features_matrix_no_date, labels_matrix_no_date)
		
		#Testing convergence by seeing if the perceptron will make any mistakes predicting on the same data it trained on
		mistakes_count = 0
		for i in range(0, len(features_matrix_no_date)):
			xii = np.matrix(features_matrix_no_date[i])
			ai = linpred.run(xii, theta)
			bi = labels_matrix_no_date[i]
			if(ai != bi):
				mistakes_count+=0
		
		print("{0} mistakes made on {1} iterations on {2} data points in datatraining.txt with no date; linear perceptron\n". format(mistakes_count, str(100000),sum(1 for line in open('Data/datatraining.txt'))-1))
		print("\n")
			
	
	
if __name__=='__main__':
	main()
