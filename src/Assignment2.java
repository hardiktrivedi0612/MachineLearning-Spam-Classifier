
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Hardik
 */
public class Assignment2 {

    public static final String currentDirectory = System.getProperty("user.dir");
//    public static final String hamTrainingDataDir = currentDirectory + "/train/ham/";
    public static final String hamTrainingDataDir = currentDirectory + "/TestTraining/ham/";
//    public static final String hamTestDataDir = currentDirectory + "/test/ham/";
    public static final String hamTestDataDir = currentDirectory + "/TestTest/ham/";
//    public static final String spamTrainingDataDir = currentDirectory + "/train/spam/";
    public static final String spamTrainingDataDir = currentDirectory + "/TestTraining/spam/";
//    public static final String spamTestDataDir = currentDirectory + "/test/spam/";
    public static final String spamTestDataDir = currentDirectory + "/TestTest/spam/";
    public static final String hamClassString = "ham";
    public static final String spamClassString = "spam";

    public static void main(String[] args) {

        //Defining the classes
        ArrayList<String> classes = new ArrayList<String>();
        classes.add(hamClassString);
        classes.add(spamClassString);

        //Multinomial Naive Bayes
        //Creating the vocabulary of training data
        HashSet<String> vocabulary = new HashSet<String>();
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
                        if (!vocabulary.contains(term)) {
                            vocabulary.add(term);
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
                        if (!vocabulary.contains(term)) {
                            vocabulary.add(term);
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
        System.out.println("Vocabulary successfully created");

        MultinomialNaiveBayes naiveBayesClassifier = new MultinomialNaiveBayes(vocabulary, nHam, nSpam, n, classes);
        try {
            naiveBayesClassifier.trainMultinomialNB();
        } catch (IOException ex) {
            System.out.println("Error in training the data for the Naive Bayes classifier");
            Logger.getLogger(Assignment2.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        System.out.println("Training the Naive Bayes classifier was successful");

        //Testing the accuracy of naive bayes classifier
        int correctlyClassified = 0, totalCount = 0;
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
            System.out.println("Error while testing data for Naive Bayes Classifier");
            Logger.getLogger(Assignment2.class.getName()).log(Level.SEVERE, null, e);
            return;
        }

        System.out.println("Accuracy of the Naive Bayes classifier ===> " + ((double) correctlyClassified / (double) totalCount) * 100);

        //Logistic Regression Classifier
        LogisticRegression logisticRegressionClassifier = null;
        try {
            logisticRegressionClassifier = new LogisticRegression(vocabulary, nHam, nSpam, n, classes);
            logisticRegressionClassifier.trainLogisticClassifier();
            System.out.println("Training the lLogistic Regression Classifier was successful");
        } catch (IOException ex) {

            System.out.println("Error while training Logistic Regression Classifier");
            Logger.getLogger(Assignment2.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
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
            Logger.getLogger(Assignment2.class.getName()).log(Level.SEVERE, null, e);
            System.out.println("Error while testing data for Logistic Regression Classifier");
            return;
        }
        System.out.println("Accuracy of the Logistic Regression classifier ===> " + ((double) correctlyClassified / (double) totalCount) * 100);
    }

}
