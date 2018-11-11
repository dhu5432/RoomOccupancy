import pandas as pd
import sol_linperceptron
import numpy as np

#features_dataframe = pd.read_table('occupancy_data/datatest.txt', sep=",",header=None, usecols=[1,2,3,4,5,6], skiprows=[0])
#features_dataframe.columns = ["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
#print features.HumidityRatio

features_dataframe = pd.read_table('Data/datatest.txt', sep=",",header=None, usecols=[2,3,4,5,6], skiprows=[0])
features_dataframe.columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


features_matrix = features_dataframe.values

#print features.HumidityRatio



labels_dataframe = pd.read_table('Data/datatest.txt', sep=",", header=None, usecols=[7], skiprows=[0])

labels_dataframe.columns = ["occupancy"]
labels_matrix = labels_dataframe.values

#print labels.occupancy

print sol_linperceptron.run(100, features_matrix, labels_matrix)

