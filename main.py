
import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import linear_primalsvm
import linpred
import kerperceptron
import kerpred
import read_data
<<<<<<< HEAD
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
=======
def main():	
	for a in range(0,4):
		iterations1=10000
		labels_matrix, features_matrix = read_data.read_data()
		#print(labels_matrix[::500])
		#theta = linear_perceptron.run(10000, features_matrix[0:500], labels_matrix[0:500])
		theta = linear_perceptron.run(iterations1, features_matrix,labels_matrix)
		count = 0
		for i in range(0,len(features_matrix)):
			xii = np.matrix(features_matrix[i])
			ai = linpred.run(xii,theta)
			bi = labels_matrix[i]
			if(ai != bi):
			#print('mistake')
				count+=1
	#print(count,"  = mistakes made on "+str(iterations1)+" iterations")
>>>>>>> 996012694c7bcd1c5f39b2b5d848f2071cabb2c3
	#print linear_perceptron.run(1000, features_matrix, labels_matrix)
	#print linear_perceptron.run(10000, features_matrix[::5],labels_matrix[::5])
	#print linear_primalsvm.run(features_matrix, labels_matrix)

		
	#file = open("results.txt", "a")
		print("{0} mistakes made on {1} iterations on {2} data points\n". format(count, str(iterations1),sum(1 for line in open('Data/datatraining.txt'))-1))
	#file.close()


	#file = open("results10000.txt", "a")
	#file.write("\n")
	#file.close()
	




if __name__=='__main__':
	main()
