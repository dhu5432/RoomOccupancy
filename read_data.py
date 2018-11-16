import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import linear_primalsvm
import linpred
def read_training_data_all():

	features_dataframe = pd.read_table('Data/datatraining.txt', sep=",",header=None, usecols=[1,2,3,4,5,6], skiprows=[0])
	features_dataframe.columns = ["date/time", "Temperatures", "Humidity", "Light", "CO2", "HumidityRatio"]


	features_matrix = features_dataframe.values


	labels_dataframe = pd.read_table('Data/datatraining.txt', sep=",", header=None, usecols=[7], skiprows=[0])

	labels_dataframe.columns = ["occupancy"]
	labels_matrix = labels_dataframe.values

	#Converting the date time feature from a string into an integer	
	index = 0
	for date_time in features_matrix:
		splitted = date_time[0].split()
		[hours, minutes, seconds] = [int(x) for x in splitted[1].split(':')]
		x = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
		date_time = x.seconds
		features_matrix[index][0] = date_time
		index+=1
	
	fx,fy = features_matrix.shape
	labels_matrix[labels_matrix == 0] = -1
	
	return features_matrix, labels_matrix

def read_training_data_no_date():

	features_dataframe = pd.read_table('Data/datatraining.txt', sep=",",header=None, usecols=[2,3,4,5,6], skiprows=[0])
	features_dataframe.columns = ["Temperatures", "Humidity", "Light", "CO2", "HumidityRatio"]


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


	
