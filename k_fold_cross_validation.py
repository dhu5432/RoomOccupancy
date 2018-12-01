import numpy as np
import math
import linear_perceptron
import read_data
import rbf_svm
from prettytable import PrettyTable
def run(k,X,y,c, shuffle=False):
	(n, d) = np.shape(X)
	accuracy_percentage = []
	if shuffle==False:
		print "{0} fold cross validation with no shuffling".format(k)
	else:
		print "{0} fold cross validation with shuffling".format(k)
		X, y = linear_perceptron.shuffle_in_unison(X,y)	
	table = PrettyTable(['Fold #', '# of elements in fold', 'Accuracy Percentage'])
	start_index = 0
	end_index = (n/k)*d
	accuracy_average=0
	for i in range(0,k):
		start_row =int(np.floor((n*i)/k))
		end_row = int(np.floor(n*(i+1))/k)-1 
	

		fold = np.zeros((end_row-start_row+1,d))
		rest = np.zeros((n-(end_row-start_row+1), d))
		
		fold_labels = np.zeros((end_row-start_row+1,1))
		rest_labels = np.zeros((n-(end_row-start_row+1),1))

		fold_row = 0
		rest_row = 0

		for a in range(0, n):
			if a >= start_row and a <=end_row:
				for col in range(d):
					fold[fold_row,col] = X[a,col]
					fold_labels[fold_row, 0]=y[a]
				fold_row+=1
					
			else:
				for column in range(d):
					rest[rest_row, column] = X[a, column]
					rest_labels[rest_row, 0] = y[a]
				rest_row+=1
		clf = rbf_svm.train(c, rest, rest_labels)
		correctly_classified=0
		mistakes = 0
		for z in range(len(fold)):
			Xi = fold[z].reshape(1,-1)
			if(clf.predict(Xi)==fold_labels[z]):
				correctly_classified+=1
			else:
				mistakes+=1
		
		#print("# of mistakes: {0}".format(mistakes))
		#mistakes_array.append(round(1.0*correctly_classified/len(fold),3))		
		table.add_row([i+1, len(fold), round(100.0*correctly_classified/len(fold),4)])
		accuracy_average += round(100.0*correctly_classified/len(fold),4)
		accuracy_percentage.append(round(100.0*correctly_classified/len(fold),4))
	#print mistakes_array
	print table
	accuracy_average = accuracy_average/k*1.0
	variance = 0
	for i in accuracy_percentage:
		variance += (i-accuracy_average)**2
	variance = variance/k
	print "C value: {0}".format(c)
	print "Average accuracy (mean): {0}".format(accuracy_average)
	print "Variance of accuracy: {0}".format(variance)
	print "Standard deviation of accuracy: {0}".format(math.sqrt(variance))
	

if __name__=='__main__':
	features_matrix, labels_matrix = read_data.read_testing_data2_all()
	features1_matrix, labels_matrix1 = read_data.read_testing_data1_all()
	print features_matrix.shape
	print features1_matrix.shape
	features2_matrix = np.vstack((features_matrix, features1_matrix))
	labels2_matrix = np.vstack((labels_matrix, labels_matrix1))
	user_input = raw_input("Would you like to shuffle the data before performing validation? (y, n) ")
	shuffle = True
	if user_input == 'n' or user_input=='N':
		shuffle = False
	user_input1 = raw_input("How many folds? ")
	c = raw_input("What do you want your C value to be? ")
	run(int(user_input1), features2_matrix, labels2_matrix, int(c),  shuffle)
