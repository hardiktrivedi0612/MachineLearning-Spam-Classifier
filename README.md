This is the Java implementation of two Machine Learning email classifiers using algorithms:
a. Naive Bayes
b. Logistic Regression

The data set used for training and testing of the classifiers are a set of emails which can be classified into 2 classes namely: 
'HAM' and 'SPAM'.

The program will output the classification of each of the test input emails as either of the classes and also give the accuracy of the algorithms implemented on the test data set.

The Naive Bayes classification implemented in the program uses add-one Laplace smoothing. The calculations have been done in the log-scale so as to avoid underflow.

The Logistic Regression classification is a MCAP Logistic Regression algorithm with L2 regularization. You can have different values of eta and lambda. The gradient ascent has been limited to a certain number of iterations.

HOW TO RUN:
Execute command:

	javac Assignment2.java
	
The format for the next command is:

	java Assignment2 <train_ham_folder> <train_spam_folder> <test_ham_folder> <test_spam_folder>

	Where:
	<train_ham_folder> : Folder where the ham training files are present
	<train_spam_folder> : Folder where the spam training files are present
	<test_ham_folder> : Folder where the ham test files are present
	<test_spam_folder> : Folder where the spam test files are present

SAMPLE INPUT:

	java Assignment2 train/ham train/spam test/ham test/spam
	
	Where:
	train/ham : Folder where the ham training files are present
	train/spam : Folder where the spam training files are present
	test/ham : Folder where the ham test files are present
	test/spam : Folder where the spam test files are present
	
	
NOTE: FOR CUSTOM INPUT FILES, PLEASE MAKE SURE THAT THE INPUT DIRECTORIES ARE IN THE SAME FOLDER AS THE JAVA FILES.
