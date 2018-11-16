import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import linear_primalsvm
import linpred

<<<<<<< HEAD
def read_data():
	features_dataframe = pd.read_table('Data/datatraining.txt', sep=",",header=None, usecols=[1,2,3,4,5,6], skiprows=[0])
	features_dataframe.columns = ["Date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
=======
def read_training_data():

	features_dataframe = pd.read_table('Data/datatraining.txt', sep=",",header=None, usecols=[2,3,4,5,6], skiprows=[0])
	features_dataframe.columns = ["Temperatures", "Humidity", "Light", "CO2", "HumidityRatio"]


>>>>>>> 996012694c7bcd1c5f39b2b5d848f2071cabb2c3

	features_matrix = features_dataframe.values


	labels_dataframe = pd.read_table('Data/datatraining.txt', sep=",", header=None, usecols=[7], skiprows=[0])

	labels_dataframe.columns = ["occupancy"]
	labels_matrix = labels_dataframe.values
	'''
	index = 0
	for date_time in features_matrix:
		splitted = date_time[0].split()
		[hours, minutes, seconds] = [int(x) for x in splitted[1].split(':')]
		x = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
		date_time = x.seconds
		features_matrix[index][0] = date_time
		index+=1
	'''	
	fx,fy = features_matrix.shape
	labels_matrix[labels_matrix == 0] = -1
	
	return features_matrix, labels_matrix

def read_testing_data2():

	features_dataframe = pd.read_table('Data/datatest2.txt', sep=",",header=None, usecols=[2,3,4,5,6], skiprows=[0])
	features_dataframe.columns = ["Temperatures", "Humidity", "Light", "CO2", "HumidityRatio"]



	features_matrix = features_dataframe.values


	labels_dataframe = pd.read_table('Data/datatest2.txt', sep=",", header=None, usecols=[7], skiprows=[0])

	labels_dataframe.columns = ["occupancy"]
	labels_matrix = labels_dataframe.values
	'''
	index = 0
	for date_time in features_matrix:
		splitted = date_time[0].split()
		[hours, minutes, seconds] = [int(x) for x in splitted[1].split(':')]
		x = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
		date_time = x.seconds
		features_matrix[index][0] = date_time
		index+=1
	'''	
	fx,fy = features_matrix.shape
	labels_matrix[labels_matrix == 0] = -1
	
	return features_matrix, labels_matrix

def read_testing_data1():

	features_dataframe = pd.read_table('Data/datatest.txt', sep=",",header=None, usecols=[2,3,4,5,6], skiprows=[0])
	features_dataframe.columns = ["Temperatures", "Humidity", "Light", "CO2", "HumidityRatio"]



	features_matrix = features_dataframe.values


	labels_dataframe = pd.read_table('Data/datatest.txt', sep=",", header=None, usecols=[7], skiprows=[0])

	labels_dataframe.columns = ["occupancy"]
	labels_matrix = labels_dataframe.values
	'''
	index = 0
	for date_time in features_matrix:
		splitted = date_time[0].split()
		[hours, minutes, seconds] = [int(x) for x in splitted[1].split(':')]
		x = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
		date_time = x.seconds
		features_matrix[index][0] = date_time
		index+=1
	'''	
	fx,fy = features_matrix.shape
	labels_matrix[labels_matrix == 0] = -1
	
	return features_matrix, labels_matrix


	
