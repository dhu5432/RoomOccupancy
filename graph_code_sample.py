import matplotlib.pyplot as pp


#graphs for 2 features
#example from case study one comparing features 20 and 10
pp.figure()
pp.plot(X[:,10], X[:,20], ’go’) # g for green, o for circle
pp.xlabel(’Feature 10’)
pp.ylabel(’Feature 20’)
pp.show() # This command will open the figure, and wait

#with both positive and negative samples
pp.figure()
 pp.plot(X[positive_samples,10], X[positive_samples,20], ’bo’) # b for blue, o for circle
 pp.plot(X[negative_samples,10], X[negative_samples,20], ’ro’) # r for red, o for circle
pp.xlabel(’Feature 10’)
 pp.ylabel(’Feature 20’)
 pp.show() # This command will open the figure, and wait



#graph for graphing the means and standard deveations for each feature
m = np.mean(X, axis=0)
s = np.std(X, axis=0)
pp.figure()
pp.plot(m, ’b+’) # b for blue, + for cross
pp.xlabel(’Feature’)
pp.ylabel(’Mean’)
pp.figure()
pp.plot(s, ’ro’) # r for red, o for circle
pp.xlabel(’Feature’)
pp.ylabel(’Standard deviation’)
pp.show() # This command will show the two figures, and wait



#graphing the theta vs features graph
theta = linprimalsvm.run(X, y)
pp.figure()
pp.plot(theta, ’b.’) # b for blue, . for dot
pp.xlabel(’Feature’)
pp.ylabel(’Theta’)
pp.show() # This command will show the figure, and wait


#graphing the theta as a bar graph
theta = linprimalsvm.run(X_small, y)
pp.figure()
pp.bar(range(theta.size), theta)
pp.xlabel(’Feature’)
pp.ylabel(’Theta’)
pp.show() # This command will show the figure, and wait




