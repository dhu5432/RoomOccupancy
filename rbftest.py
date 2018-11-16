import pandas as pd
import numpy as np
from sklearn import svm
import read_data
def main():
    X,y = read_data.read_training_data_all()
    print(y[100:110])
    y = y.ravel()
    clf = svm.SVC(gamma='scale',kernel='rbf')
    clf2 = svm.SVC(gamma='scale',kernel='linear')
    clf.fit(X[:,1:],y)
    clf2.fit(X[:,1:],y)
    count = 0
    count2 = 0
    for i in range(len(y)):
        Xi = X[i,1:].reshape(1,-1)
        if clf.predict(Xi) != y[i]:
            count+=1
        if clf2.predict(Xi) != y[i]:
            count2+=1
    print('rbf mistakes made = ',count)
    print('linaer mistakes made = ',count2)
main()
