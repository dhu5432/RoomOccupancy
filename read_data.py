import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import linear_primalsvm

#in data test.txt we are using first 500 samples as training, and sampels 1000-1100 for testing
#features_dataframe = pd.read_table('occupancy_data/datatest.txt', sep=",",header=None, usecols=[1,2,3,4,5,6], skiprows=[0])
#features_dataframe.columns = ["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
#print features.HumidityRatio

features_dataframe = pd.read_table('Data/datatest.txt', sep=",",header=None, usecols=[1,2,3,4,5,6], skiprows=[0])
features_dataframe.columns = ["Date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


features_matrix = features_dataframe.values

#print features.HumidityRatio



labels_dataframe = pd.read_table('Data/datatest.txt', sep=",", header=None, usecols=[7], skiprows=[0])

labels_dataframe.columns = ["occupancy"]
labels_matrix = labels_dataframe.values

index = 0
for date_time in features_matrix:
	splitted = date_time[0].split()
	[hours, minutes, seconds] = [int(x) for x in splitted[1].split(':')]
	x = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
	date_time = x.seconds
	features_matrix[index][0] = date_time
	index+=1
	
#print features_matrix
fx,fy = features_matrix.shape
#print labels.occupancy
labels_matrix[labels_matrix == 0] = -1
print(labels_matrix[::500])
theta = linear_perceptron.run(10000, features_matrix[0:500], labels_matrix[0:500])
print theta
#print linear_perceptron.run(1000, features_matrix, labels_matrix)
#print linear_perceptron.run(10000, features_matrix[::5],labels_matrix[::5])
#print linear_primalsvm.run(features_matrix, labels_matrix)
