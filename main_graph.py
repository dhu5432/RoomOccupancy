import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import read_data
import k_fold_cross_validation
import rbf_svm
import matplotlib.pyplot as pp

def run():
	

	X, y = read_data.read_training_data_all()
	X1, y1 = read_data.read_training_data_no_time()		
	m = np.mean(X, axis=0)
	s = np.std(X1, axis=0)
	pp.figure()
	pp.xticks(np.arange(6), ('Time', 'Temperatures', 'Humidity', 'Light', 'CO2', 'HumidityRatio'))
	pp.plot(m, 'b+') # b for blue, + for cross
	pp.xlabel('Feature')
	pp.ylabel('Mean')
	pp.figure()
	pp.xticks(np.arange(5), ('Temperatures', 'Humidity', 'Light', 'CO2', 'HumidityRatio'))
	pp.plot(s, 'ro') # r for red, o for circle
	pp.xlabel('Feature')
	pp.ylabel('Standard deviation')
	#pp.show() # This command will show the two figures, and wait
	

	#theta = rbf_svm.train(10,X,y)
	#print("----")
	#print(theta)
	#print("----")
	#pp.figure()
	#pp.plot(theta, 'b.') # b for blue, . for dot
	#pp.xlabel('Feature')
	#pp.ylabel('Theta')
 	#pp.show() # This command will show the figure, and wait 
	
	

	names = ['Time','Temperatures', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
	X, y = read_data.read_training_data_all()	
	positive_samples = list(np.where(y==1)[0])
	negative_samples = list(np.where(y==-1)[0])
	

	for i in range(0,6):
		for j in range(i,6):
			if i != j:
				pp.figure()
				pp.plot(X[positive_samples,i], X[positive_samples,j], 'bo') # b for blue, o for circle
				pp.plot(X[negative_samples,i], X[negative_samples,j], 'ro') # r for red, o for circle
				pp.xlabel('Feature: '+str(names[i]))
				pp.ylabel('Feature: '+str(names[j]))
				#pp.show() # This command will open the figure, and wait
 	pp.show()







	
if __name__=='__main__':
	main()
