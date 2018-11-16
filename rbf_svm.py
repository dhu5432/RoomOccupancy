import pandas as pd
import numpy as np
from sklearn import svm
import read_data
from prettytable import PrettyTable
def train(c, X, y):
	y = y.ravel()	
	rbf_svm = svm.SVC(C=c, gamma='scale', kernel='rbf')
	rbf_svm.fit(X,y)
	return rbf_svm




def run():
	training_features_matrix_all, training_labels_matrix_all = read_data.read_training_data_all()
	training_features_matrix_no_time, training_labels_matrix_no_time = read_data.read_training_data_no_time()

	testing1_features_matrix_all, testing1_labels_matrix_all = read_data.read_testing_data1_all()
	testing1_features_matrix_no_time, testing1_labels_matrix_no_time = read_data.read_testing_data1_no_time()

	testing2_features_matrix_all, testing2_labels_matrix_all = read_data.read_testing_data2_all()
	testing2_features_matrix_no_time, testing2_labels_matrix_no_time = read_data.read_testing_data2_no_time()

	C = [0.0001, 0.001, 0.01, 1, 10, 100, 1000,10000,100000]

	t = PrettyTable(['C value', 'Training mistakes', 'Test mistakes on test set 1', 'Test mistakes on test set 2'])
	#Testing slack variables with date included
	print("Testing RBF SVM with the time included")
	for c in C:
		training_mistakes_count = 0
		testing_mistakes_count1 = 0
		testing_mistakes_count2 = 0
		clf = train(c, training_features_matrix_all, training_labels_matrix_all)
		
		for i in range(len(training_labels_matrix_all)):
			Xi = training_features_matrix_all[i].reshape(1,-1)
			if(clf.predict(Xi) != training_labels_matrix_all[i]):
				training_mistakes_count+=1
		
		for i in range(len(testing1_labels_matrix_all)):
			Xi = testing1_features_matrix_all[i].reshape(1,-1)
			if(clf.predict(Xi) != testing1_labels_matrix_all[i]):
				testing_mistakes_count1+=1

		for i in range(len(testing2_labels_matrix_all)):
			Xi = testing2_features_matrix_all[i].reshape(1,-1)
			if(clf.predict(Xi) != testing2_labels_matrix_all[i]):
				testing_mistakes_count2+=1
		t.add_row([c, training_mistakes_count, testing_mistakes_count1, testing_mistakes_count2])
	print t
	print("")

	t1 = PrettyTable(['C value', 'Training mistakes', 'Test mistakes on test set 1', 'Test mistakes on test set 2'])
	#Testing slack variables without date
	print("Testing RBF SVM without the time")
	for c in C:
		training_mistakes_count = 0
		testing_mistakes_count1 = 0
		testing_mistakes_count2 = 0
		clf = train(c, training_features_matrix_no_time, training_labels_matrix_no_time)
		
		for i in range(len(training_labels_matrix_no_time)):
			Xi = training_features_matrix_no_time[i].reshape(1,-1)
			if(clf.predict(Xi) != training_labels_matrix_no_time[i]):
				training_mistakes_count+=1
		
		for i in range(len(testing1_labels_matrix_no_time)):
			Xi = testing1_features_matrix_no_time[i].reshape(1,-1)
			if(clf.predict(Xi) != testing1_labels_matrix_no_time[i]):
				testing_mistakes_count1+=1

		for i in range(len(testing2_labels_matrix_no_time)):
			Xi = testing2_features_matrix_no_time[i].reshape(1,-1)
			if(clf.predict(Xi) != testing2_labels_matrix_no_time[i]):
				testing_mistakes_count2+=1
		t1.add_row([c, training_mistakes_count, testing_mistakes_count1, testing_mistakes_count2])
	print t1




if __name__=='__main__':
	run()
