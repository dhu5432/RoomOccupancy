import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import read_data
import k_fold_cross_validation
import rbf_svm
import matplotlib.pyplot as pp

def main():
	

	X, y = read_data.read_training_data_no_time()

	m = np.mean(X, axis=0)
	#s = np.std(X, axis=0)
	pp.figure()
	pp.plot(m, 'b+') # b for blue, + for cross
	pp.xlabel('Feature')
	pp.ylabel('Mean')
	pp.figure()
	#pp.plot(s, 'ro') # r for red, o for circle
	#pp.xlabel('Feature')
	#pp.ylabel('Standard deviation')
	pp.show() # This command will show the two figures, and wait
  









	
if __name__=='__main__':
	main()
