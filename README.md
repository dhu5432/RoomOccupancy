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

# Algorithms
## Linear perceptron (linear_perceptron.py):
We implemented our own linear perceptron algorithm to run on the data. This perceptron algorithms contains similarities with the
perceptron algorithms covered in class but with the addition of shuffling. Depending on the order of training data, perceptron can
converge to different Θ (theta) values because the training algorithm stops learning when it stops making mistakes and there can be
multiple combinations/permutations of the data in which no mistakes occur
(which could mean that it would only learn from a handful of examples). Also, re-permuting the data after each iteration can lead to a
faster convergence.

Initially, we ran the perceptron mentioned above on the training data set with all the features to determine if it was linearly
separable and determined that the data was not linearly separable. We came to this conclusion by running our perceptron on the training
data set for a large number of iterations (10,000 and 100,000). Then used the θ returned to predict the labels from the same data set
used to train the perceptron. Theoretically, if this data was linearly separable and the perceptron converged, then it would have made
no mistakes when predicting labels from θ in the same data set that it was trained on. However, we found that the algorithm made a non-
zero number of mistakes each time it was run, showing that that the data is not linearly separable.

At this point, we decided to test linear perceptron without the time included as a feature because we know that the room the data is 
representing is the occupancy of an office room, which would mean that occupancy would cluster during work hours (generally 8 am
to 5 pm). This would mean that occupancy of the room would be clustered around specific times for each day. Also, it is possible that 
the company could decide to have a party after hours or on the weekend or that a holiday falls on a business day, skewing predictions.
We again ran perceptron on the training dataset with all the features but the time. We again determined that it was not linearly 
separable, in the same manner as before. Therefore showing that even without the date/time included as a feature, it was not linearly 
separable.

![Perceptron Results](https://github.com/dhu5432/RoomOccupancy/blob/master/Pictures/PerceptronRuns.PNG)

Now that we know that data is not linearly separable, we have decided to
replace the linear perceptron with a radial basis kernel support vector
machine in hopes to separate the data.

## Radial Basis Kernel SVM (rbf_svm.py):
We used sci-kit learn's RBF support vector machine on the training data in 
hopes to achieve better results than the linear perceptron algorithm. When
it came to tuning the parameters for this SVM, we had to decide whether or
not to include time because of the reasons mentioned above and determine an appropriate C/slack value. For large C values, the SVM will 
choose a smaller-margin hyperplane if the hyperplane does a better job of getting all the training points classified correctly while a
small C value will cause the optimizer to look for a larger-margin separating hyperplane, even at the expense of misclassifying more
points. To determine an optimal C, we initialized our SVMs with C values from 0.0001 to 100,000 and counted the number of mistakes it 
would make on the training set set. In addition, we tested the inclusion of the date/time feature and whether or not that would make a 
significant difference in accuracy (in conjunction with the C values). Our results are as follows: 

![RBF SVM Results](https://github.com/dhu5432/RoomOccupancy/blob/master/Pictures/RBF%20SVM%20Runs.PNG)

It seems that the time makes a difference in prediction accuracy because the number of mistakes the SVM makes on the training set is 
higher when time is not included for corresponding C values. With the time included, the SVM improves its accuracy as C values get 
larger with no significant differences/improvements past C=100. With this in mind, we decided to include time as a feature and a C value
of 100 in order to avoid overfitting in our RBF SVM. '

# Cross Validation (k_fold_cross_validation.py):
For cross validation, because we determined that the data was not linearly separable, we did not perform k-fold validation on the linear 
perceptron algorithm. Instead, we did a k-fold cross validation with radial basis kernel on the two testing sets combined. K-fold cross
validation works by splitting the data set into k disjoint sets, and training a classifier on k-1 of those and test it on the remaining
set. We do this k times, each time holding out a different set as the “testing” part. We then average the performance over all k parts 
to get an estimate on how well the model will perform in the future on unseen datasets. In our particular case, we decided to use 
accuracy percentage as our performance measure. Usually, more folds (a higher k) will lead to better estimates, but each new fold 
requires training a new classifier, which is computationally expensive. We decided on k=10 folds because it seemed high enough to
accurately measure model performance but not too high where it would take an unreasonable amount of time to run. Our results are as follows when we first ran the 10 fold cross validation on the training data: 

![Cross Validation (No Shuffling) Results](https://github.com/dhu5432/RoomOccupancy/blob/master/Pictures/CrossValidationNoShuffling.PNG)

Overall, the results seemed quite positive as most folds were hitting accuracy percentages in the 90-100 range but there were 1 fold
whose accuracy were significantly lower than the others (88%), skewing the overall average accuracy and the standard deviation. We
believed that this could have been caused by the fact that in the training set, the data is organized in blocks of unoccupied/occupied.
For example, lines 1-16 in the training set are occupied data points (1), then lines 17-830 are all unoccupied (0), then lines 831-1122
are mostly occupied (1). We believed that it would be better to shuffle the data set before cross validating to ensure a more even 
distribution of unoccupied vs occupied data points in each fold. Our results of the 10 fold cross validation with shuffling are as
follows:
![Cross Validation (With Shuffling) Results](https://github.com/dhu5432/RoomOccupancy/blob/master/Pictures/CrossValidationShuffling.PNG)

As we can see, the reuslts/accuracy of our RBF SVM is far better when shuffling the data before cross validating. From our cross validation, we have determined that our RBF SVM with a C (slack) value of 100 is a good classifier for this data set. 

To replicate the results we got, see [manual.txt](https://github.com/dhu5432/RoomOccupancy/blob/master/manual.txt) for instructions on how to run the code. 

