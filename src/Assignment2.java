
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Hardik
 */
public class Assignment2 {

    public static final String currentDirectory = System.getProperty("user.dir");
    public static String hamTrainingDataDir = currentDirectory + "/train/ham/";
    public static String hamTestDataDir = currentDirectory + "/test/ham/";
    public static String spamTrainingDataDir = currentDirectory + "/train/spam/";
    public static String spamTestDataDir = currentDirectory + "/test/spam/";

    public static final String hamClassString = "ham";
    public static final String spamClassString = "spam";

    public static final String stopwordsFileName = "StopWords.txt";

    /*
     Input format:
     java Assignment2 <train_ham_folder> <train_spam_folder> <test_ham_folder> <test_spam_folder>
    
     Sample Input:
     java Assignment2 train/ham train/spam test/ham test/spam
     */
    public static void main(String[] args) {

        if (args.length < 4) {
            System.out.println("Some of the parameters seem to be missing. Please check the README file and try again.");
            return;
        }

        try {
            hamTrainingDataDir = currentDirectory + "/" + args[0] + "/";
            spamTrainingDataDir = currentDirectory + "/" + args[1] + "/";
            hamTestDataDir = currentDirectory + "/" + args[2] + "/";
            spamTestDataDir = currentDirectory + "/" + args[3] + "/";
        } catch (Exception e) {
            System.out.println("There seems to be some error in the directory names entered. Please check the input and try again");
            return;
        }

        try {
            File file = new File(hamTrainingDataDir);
            if (!file.exists()) {
                System.out.println("Ham Training folder does not exist. Please make sure that it is in the same folder as this java file. "
                        + "Check the README file for more details");
                return;
            }
            file = new File(spamTrainingDataDir);
            if (!file.exists()) {
                System.out.println("Spam Training folder does not exist. Please make sure that it is in the same folder as this java file. "
                        + "Check the README file for more details");
                return;
            }
            file = new File(hamTestDataDir);
            if (!file.exists()) {
                System.out.println("Ham Test folder does not exist. Please make sure that it is in the same folder as this java file. "
                        + "Check the README file for more details");
                return;
            }
            file = new File(spamTestDataDir);
            if (!file.exists()) {
                System.out.println("Spam Test folder does not exist. Please make sure that it is in the same folder as this java file. "
                        + "Check the README file for more details");
                return;
            }
        } catch (Exception e) {
            System.out.println("Some of the directories entered do not exist. Please check the inputs and try again.");
            return;
        }

        //Reading the stopwords
        HashSet<String> stopWords = new HashSet<>();
        try {
            File stopWordsFile = new File(currentDirectory + "/" + stopwordsFileName);
            BufferedReader stopWordsReader = new BufferedReader(new FileReader(stopWordsFile));
            String stopWord = stopWordsReader.readLine();
            while (stopWord != null) {
                String words[] = stopWord.split(" ");
                stopWords.addAll(Arrays.asList(words));
                stopWord = stopWordsReader.readLine();
            }
        } catch (Exception ex) {
            System.out.println("Error reading the stopwords file. Please make sure that the \"StopWords.txt\" file is in the same folder as the java files."
                    + "Check the README file for more details");
            return;
        }

        //Defining the classes
        ArrayList<String> classes = new ArrayList<>();
        classes.add(hamClassString);
        classes.add(spamClassString);

        //Creating the vocabularies of training data
        HashSet<String> vocabularyWithStopWords = new HashSet<String>();
        HashSet<String> vocabularyWithoutStopWords = new HashSet<String>();
        BufferedReader reader = null;
        int nHam = 0, nSpam = 0, n = 0;
        try {
            File hamTrainingFolder = new File(hamTrainingDataDir);
            for (File hamTextFile : hamTrainingFolder.listFiles()) {
                reader = new BufferedReader(new FileReader(hamTextFile));
                String line = reader.readLine();
                while (line != null) {
                    String terms[] = line.split(" ");
                    for (String term : terms) {
                        if (!vocabularyWithStopWords.contains(term)) {
                            vocabularyWithStopWords.add(term);
                        }
                        if (!vocabularyWithoutStopWords.contains(term) && !stopWords.contains(term)) {
                            vocabularyWithoutStopWords.add(term);
                        }
                    }
                    line = reader.readLine();
                }
                nHam++;
                n++;
            }
            File spamTrainingFolder = new File(spamTrainingDataDir);
            for (File spamTextFile : spamTrainingFolder.listFiles()) {
                reader = new BufferedReader(new FileReader(spamTextFile));
                String line = reader.readLine();
                while (line != null) {
                    String terms[] = line.split(" ");
                    for (String term : terms) {
                        if (!vocabularyWithStopWords.contains(term)) {
                            vocabularyWithStopWords.add(term);
                        }
                        if (!vocabularyWithoutStopWords.contains(term) && !stopWords.contains(term)) {
                            vocabularyWithoutStopWords.add(term);
                        }
                    }
                    line = reader.readLine();
                }
                nSpam++;
                n++;
            }
        } catch (Exception e) {
            Logger.getLogger(Assignment2.class.getName()).log(Level.SEVERE, null, e);
            System.out.println("Error in creating vocabulary. Try again");
            return;
        }
        //vocabulary successfully created
        System.out.println("Vocabularies successfully created");

        int input = 0;
        int correctlyClassified = 0;
        int totalCount = 0;
        double eta = 0;
        double lambda = 0;
        int iterations = 0;
        int toEnterValues = -1;
        boolean valuesEntered = false;
        BufferedReader consoleReader = new BufferedReader(new InputStreamReader(System.in));
        MultinomialNaiveBayes naiveBayesClassifier = null;
        LogisticRegression logisticRegressionClassifier = null;
        while (input != 5) {
            System.out.println("");
            System.out.println("");
            System.out.println("Select what you want to do from the following: (Enter integer according to the menu items)\n"
                    + "1. Naive Bayes (including the stop words)\n"
                    + "2. Naive Bayes (not including the stop words)\n"
                    + "3. Logistic Regression (including the stop words)\n"
                    + "4. Logistic Regression (not including the stop words)\n"
                    + "5. Exit");
            try {
                input = Integer.parseInt(consoleReader.readLine());

            } catch (IOException ex) {
                System.out.println("Please enter numbers only. Please try again");
                input = 0;
                continue;
            }
            System.out.println("");
            System.out.println("");
            switch (input) {
                case 1:
                    System.out.println("Naive Bayes (including the stop words) selected. Starting training.");
                    naiveBayesClassifier = new MultinomialNaiveBayes(vocabularyWithStopWords, nHam, nSpam, n, classes);
                    try {
                        naiveBayesClassifier.trainMultinomialNB();
                    } catch (IOException ex) {
                        System.out.println("Error in training the data for the Naive Bayes classifier. Check the data and try again");
                        continue;
                    }
                    System.out.println("Training the Naive Bayes classifier was successful");
                    System.out.println("Starting testing Naive Bayes Classifier");
                    //Testing the accuracy of naive bayes classifier
                    correctlyClassified = 0;
                    totalCount = 0;
                    try {
                        File hamTestFolder = new File(hamTestDataDir);
                        for (File hamTextFile : hamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(hamTextFile));
                            if (naiveBayesClassifier.applyMultinomailNB(reader).equalsIgnoreCase(hamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                        File spamTestFolder = new File(spamTestDataDir);
                        for (File spamTextFile : spamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(spamTextFile));
                            if (naiveBayesClassifier.applyMultinomailNB(reader).equalsIgnoreCase(spamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                    } catch (Exception e) {
                        System.out.println("Error while testing data for Naive Bayes Classifier. Please check the data and try again");
                        continue;
                    }
                    System.out.println("Testing completed!");
                    System.out.println("Accuracy of the Naive Bayes classifier ===> " + ((double) correctlyClassified / (double) totalCount) * 100 + "%");
                    break;
                case 2:
                    System.out.println("Naive Bayes (not including the stop words) selected. Starting training.");
                    naiveBayesClassifier = new MultinomialNaiveBayes(vocabularyWithoutStopWords, nHam, nSpam, n, classes);
                    try {
                        naiveBayesClassifier.trainMultinomialNB();
                    } catch (IOException ex) {
                        System.out.println("Error in training the data for the Naive Bayes classifier. Check the data and try again");
                        continue;
                    }
                    System.out.println("Training the Naive Bayes classifier was successful");
                    System.out.println("Starting testing Naive Bayes Classifier");
                    //Testing the accuracy of naive bayes classifier
                    correctlyClassified = 0;
                    totalCount = 0;
                    try {
                        File hamTestFolder = new File(hamTestDataDir);
                        for (File hamTextFile : hamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(hamTextFile));
                            if (naiveBayesClassifier.applyMultinomailNB(reader).equalsIgnoreCase(hamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                        File spamTestFolder = new File(spamTestDataDir);
                        for (File spamTextFile : spamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(spamTextFile));
                            if (naiveBayesClassifier.applyMultinomailNB(reader).equalsIgnoreCase(spamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                    } catch (Exception e) {
                        System.out.println("Error while testing data for Naive Bayes Classifier. Please check the data and try again");
                        continue;
                    }
                    System.out.println("Testing completed!");
                    System.out.println("Accuracy of the Naive Bayes classifier ===> " + ((double) correctlyClassified / (double) totalCount) * 100 + "%");
                    break;
                case 3:
                    System.out.println("Logistic Regression classifier (including stop words selected)");
                    System.out.println("Would you like to enter the values for eta, lambda and no_of_iterations? (Enter 1 for yes / Enter 0 for no)");
                    toEnterValues = -1;
                    eta = 0.001;
                    lambda = 0.075;
                    iterations = 13;
                    try {
                        toEnterValues = Integer.parseInt(consoleReader.readLine());
                    } catch (Exception e) {
                        System.out.println("Please enter yes or no only. Try again");
                        continue;
                    }
                    valuesEntered = false;

                    if (toEnterValues != -1) {
                        if (toEnterValues == 1) {
                            while (!valuesEntered) {
                                System.out.println("Enter the value for eta:");
                                try {
                                    eta = Double.parseDouble(consoleReader.readLine());
                                } catch (Exception e) {
                                    System.out.println("Please enter only double values for eta");
                                    continue;
                                }
                                System.out.println("Enter the value of lambda:");
                                try {
                                    lambda = Double.parseDouble(consoleReader.readLine());
                                } catch (Exception e) {
                                    System.out.println("Please enter only double values for lambda");
                                    continue;
                                }
                                System.out.println("Enter the number of iterations:");
                                try {
                                    iterations = Integer.parseInt(consoleReader.readLine());
                                } catch (Exception e) {
                                    System.out.println("Please enter only integer values for number of iterations");
                                    continue;
                                }
                                valuesEntered = true;
                            }
                        }
                    }

                    try {
                        System.out.println("Creating data for logistic classifier");
                        logisticRegressionClassifier = new LogisticRegression(vocabularyWithStopWords, nHam, nSpam, n, classes);
                        logisticRegressionClassifier.eta = eta;
                        logisticRegressionClassifier.lambda = lambda;
                        logisticRegressionClassifier.gradientAscentIterations = iterations;
                        System.out.println("Data Creation complete. Starting training of Logistic Regression classifier");
                        logisticRegressionClassifier.trainLogisticClassifier();
                        System.out.println("Training the Logistic Regression Classifier was successful");
                    } catch (Exception ex) {
                        System.out.println("Error while training Logistic Regression Classifier. Error might be due to overflow. Please check the values of eta, lambda and no_of_iterations and try again");
                        continue;
                    }
                    System.out.println("Starting testing for Logistic Regression classifier");
                    correctlyClassified = 0;
                    totalCount = 0;
                    try {
                        File hamTestFolder = new File(hamTestDataDir);
                        for (File hamTextFile : hamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(hamTextFile));
                            if (logisticRegressionClassifier.applyLogisticRegressionClassifier(reader).equalsIgnoreCase(hamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                        File spamTestFolder = new File(spamTestDataDir);
                        for (File spamTextFile : spamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(spamTextFile));
                            if (logisticRegressionClassifier.applyLogisticRegressionClassifier(reader).equalsIgnoreCase(spamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                    } catch (Exception e) {
                        System.out.println("Error while testing data for Logistic Regression Classifier. Please check the data and try again");
                        continue;
                    }
                    System.out.println("Testing completed!");
                    System.out.println("Accuracy of the Logistic Regression classifier for eta = " + logisticRegressionClassifier.eta + ", lambda = " + logisticRegressionClassifier.lambda + " and no_of_iterations = " + logisticRegressionClassifier.gradientAscentIterations + "  is ===> " + ((double) correctlyClassified / (double) totalCount) * 100 + "%");
                    break;
                case 4:
                    System.out.println("Logistic Regression classifier (not including stop words selected)");
                    System.out.println("Would you like to enter the values for eta, lambda and no_of_iterations? (Enter 1 for yes / Enter 0 for no)");
                    toEnterValues = -1;
                    eta = 0.001;
                    lambda = 0.01;
                    iterations = 20;
                    try {
                        toEnterValues = Integer.parseInt(consoleReader.readLine());
                    } catch (Exception e) {
                        System.out.println("Please enter yes or no only. Try again");
                        continue;
                    }
                    valuesEntered = false;

                    if (toEnterValues != -1) {
                        if (toEnterValues == 1) {
                            while (!valuesEntered) {
                                System.out.println("Enter the value for eta:");
                                try {
                                    eta = Double.parseDouble(consoleReader.readLine());
                                } catch (Exception e) {
                                    System.out.println("Please enter only double values for eta");
                                    continue;
                                }
                                System.out.println("Enter the value of lambda:");
                                try {
                                    lambda = Double.parseDouble(consoleReader.readLine());
                                } catch (Exception e) {
                                    System.out.println("Please enter only double values for lambda");
                                    continue;
                                }
                                System.out.println("Enter the number of iterations:");
                                try {
                                    iterations = Integer.parseInt(consoleReader.readLine());
                                } catch (Exception e) {
                                    System.out.println("Please enter only integer values for number of iterations");
                                    continue;
                                }
                                valuesEntered = true;
                            }
                        }
                    }

                    try {
                        System.out.println("Creating data for logistic classifier");
                        logisticRegressionClassifier = new LogisticRegression(vocabularyWithoutStopWords, nHam, nSpam, n, classes);
                        logisticRegressionClassifier.eta = eta;
                        logisticRegressionClassifier.lambda = lambda;
                        logisticRegressionClassifier.gradientAscentIterations = iterations;
                        System.out.println("Data Creation complete. Starting training of Logistic Regression classifier");
                        logisticRegressionClassifier.trainLogisticClassifier();
                        System.out.println("Training the Logistic Regression Classifier was successful");
                    } catch (Exception ex) {
                        System.out.println("Error while training Logistic Regression Classifier. Error might be due to overflow. Please check the values of eta, lambda and no_of_iterations and try again");
                        continue;
                    }
                    System.out.println("Starting testing for Logistic Regression classifier");
                    correctlyClassified = 0;
                    totalCount = 0;
                    try {
                        File hamTestFolder = new File(hamTestDataDir);
                        for (File hamTextFile : hamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(hamTextFile));
                            if (logisticRegressionClassifier.applyLogisticRegressionClassifier(reader).equalsIgnoreCase(hamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                        File spamTestFolder = new File(spamTestDataDir);
                        for (File spamTextFile : spamTestFolder.listFiles()) {
                            reader = new BufferedReader(new FileReader(spamTextFile));
                            if (logisticRegressionClassifier.applyLogisticRegressionClassifier(reader).equalsIgnoreCase(spamClassString)) {
                                correctlyClassified++;
                            }
                            totalCount++;
                        }
                    } catch (Exception e) {
                        System.out.println("Error while testing data for Logistic Regression Classifier. Please check the data and try again");
                        continue;
                    }
                    System.out.println("Testing completed!");
                    System.out.println("Accuracy of the Logistic Regression classifier for eta = " + logisticRegressionClassifier.eta + ", lambda = " + logisticRegressionClassifier.lambda + " and no_of_iterations = " + logisticRegressionClassifier.gradientAscentIterations + "  is ===> " + ((double) correctlyClassified / (double) totalCount) * 100 + "%");
                    break;
                case 5:
                    break;
                default:
                    System.out.println("Please enter a value between [0-5]");
                    break;

            }
        }
    }
}
