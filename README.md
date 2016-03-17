This is the Java implementation of two Machine Learning email classifiers using algorithms:
a. Naive Bayes
b. Logistic Regression

The data set used for training and testing of the classifiers are a set of emails which can be classified into 2 classes namely: 
'HAM' and 'SPAM'.

The program will output the classification of each of the test input emails as either of the classes and also give the accuracy of the algorithms implemented on the test data set.

The Naive Bayes classification implemented in the program uses add-one Laplace smoothing. The calculations have been done in the log-scale so as to avoid underflow.

The Logistic Regression classification is a MCAP Logistic Regression algorithm with L2 regularization. You can have different values of eta and lambda. The gradient ascent has been limited to a certain number of iterations.
