import pandas as pd
import linear_perceptron
import numpy as np
import datetime

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

#print labels.occupancy

print sol_linperceptron.run(100, features_matrix, labels_matrix)

