import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import read_data
import k_fold_cross_validation
import rbf_svm
import main_graph

def main():
	# Testing for linear separability by determining whether or not
	# our linear perceptron ever converges
	while(1):
		print("What would you like run?")
		print("Press 1 for linear perceptron (takes awhile to run)")
		print("Press 2 for RBF SVM")
		print("Press 3 for K Fold Cross Validation of the RBF SVM")
		print("Press 4 to view all graphs")
		input_var = raw_input()
		if input_var == "1":
			linear_perceptron.run()			
		elif input_var == "2":
			rbf_svm.run()
		elif input_var == "3":
			num_folds = raw_input("How many folds would you like? ")
			shuffle = raw_input("Would you like to shuffle the data before folding? (y, n) ")
			shuffle_bool = True
			if shuffle == "n" or shuffle== "N":
				shuffle_bool = False
			features_matrix, labels_matrix = read_data.read_training_data_all()
			k_fold_cross_validation.run(int(num_folds), features_matrix, labels_matrix, shuffle_bool)
		elif input_var == "4":
			main_graph.run()
	
		print("\n")	
	
	
if __name__=='__main__':
	main()
