
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
	labels_matrix, features_matrix = read_data.read_data()
	#print(labels_matrix[::500])
	#theta = linear_perceptron.run(10000, features_matrix[0:500], labels_matrix[0:500])
	#theta = linear_perceptron.run(10000, features_matrix,labels_matrix)
	theta = kerperceptron.run(10, features_matrix[::800],labels_matrix[::800])
	print theta
	print "----------"
	count = 0
	for i in range(0,len(features_matrix[::800])):
		xii = np.matrix(features_matrix[i])
		ai = kerpred.run(theta,features_matrix[::800], labels_matrix[::800],xii)
		#ai = linpred.run(xii,theta)
		#bi = labels_matrix[i]
		#if(ai != bi):
		 #   print('mistake')
		  #  count+=1
	print(count,'  = mistakes made')
	#print linear_perceptron.run(1000, features_matrix, labels_matrix)
	#print linear_perceptron.run(10000, features_matrix[::5],labels_matrix[::5])
	#print linear_primalsvm.run(features_matrix, labels_matrix)








if __name__=='__main__':
	main()
