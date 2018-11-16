import pandas as pd
import linear_perceptron
import numpy as np
import datetime
import read_data
def main():
	# Testing for linear separability by determining whether or not
	# our linear perceptron ever converges
	linear_perceptron.run()			
	
	
if __name__=='__main__':
	main()
