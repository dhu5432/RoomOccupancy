import pandas as pd
import numpy as np
from sklearn import svm
import read_data
def main():
    X,y = read_data.read_training_data_all()
    Xtest,Ytest = read_data.read_testing_data1()
    y = y.ravel()
    C = [100,1000,10000]
    print('training data has ', len(X), ' samples')
    print('testing data has ' , len(Xtest), ' samples')
    for c in C:
        clf = svm.SVC(C=c,gamma='scale',kernel='rbf')
#        clf2 = svm.SVC(C=c,gamma='scale',kernel='linear')
        clf.fit(X,y)
#        clf2.fit(X,y)
        count = 0
        count2 = 0
        for i in range(len(y)):
            Xi = X[i].reshape(1,-1)
            if clf.predict(Xi) != y[i]:
                count+=1
        for i in range(len(Ytest)):
            Xi = Xtest[i].reshape(1,-1)
            if clf.predict(Xi) != Ytest[i]:
                count2+=1
#            if clf2.predict(Xi) != y[i]:
#                count2+=1
        print('rbf training mistakes made = ',count, ' with c = ',c)
        print('rbf test mistakes made = ', count2, ' with c = ', c)
 #       print('linaer mistakes made = ',count2, ' with c = ', c)
main()
