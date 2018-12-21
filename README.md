# Problem:
Given various parameters such as temperature or light level, we want to be able to predict whether an office room is occupied. If we can 
accurately predict occupancy in a room, we can therefore cut down on energy costs (why turn the head/A.C. up when there is no one in the
room)

# Dataset (Data/):
The data set contains samples with seven different different attributes (current date in year-month-day hour:minute:second format), 
temperature (Celsius), relative humidity (percentage), light (lux), CO2 (ppm), humidity ratio (kg water vapor/kg air), and occupancy 
(0 for unoccupied and 1 for occupied). Each sample is consecutively taken every minute over several days.

https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

# Preprocessing data (read_data.py):
Currently, the data is read into numpy matrices. In read_data.py, we have separate functions to read in the data from the three data text
files (two test sets and one training set). We use pandas, a python data analysis library, to read the information into DataFrames (an 
object for efficient data manipulation) and then convert it into numpy matrices. The first six columns in the text files are the 
independent variables: date/time, temperature, humidity, light, CO2, and humidity ratio (which comprises our features matrix) while
the last column is the dependent variable (which will make up our labels matrix). 

The only non-numerical feature in our data is the date/time, and we remedy this by disregarding the date and converting the time string
into a numerical number of seconds past midnight. We believe that the date does not matter too much because the training data set only 
encompasses seven different days for 8,000+ data points. We changed all the 0's in the labels matrix to -1 (for unoccupied) so that it 
would be compatible with our perceptron algorithm.
