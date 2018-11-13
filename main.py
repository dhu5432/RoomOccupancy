
import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import linear_primalsvm
import linpred
import read_data
def main():
	labels_matrix, features_matrix = read_data.read_data()
	print(labels_matrix[::500])
	#theta = linear_perceptron.run(10000, features_matrix[0:500], labels_matrix[0:500])
	theta = linear_perceptron.run(10000, features_matrix[::2],labels_matrix[::2])
	print theta
	count = 0
	for i in range(1,len(features_matrix),2):
		xii = np.matrix(features_matrix[i])
		ai = linpred.run(xii,theta)
		bi = labels_matrix[i]
		if(ai != bi):
		    print('mistake')
		    count+=1
	print(count,'  = mistakes made')
	#print linear_perceptron.run(1000, features_matrix, labels_matrix)
	#print linear_perceptron.run(10000, features_matrix[::5],labels_matrix[::5])
	#print linear_primalsvm.run(features_matrix, labels_matrix)








if __name__=='__main__':
	main()
