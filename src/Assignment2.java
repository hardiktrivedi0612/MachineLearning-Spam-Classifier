
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
    public static final String hamTrainingDataDir = currentDirectory + "/train/ham/";
//    public static final String hamTrainingDataDir = currentDirectory + "/TestTraining/ham/";
    public static final String hamTestDataDir = currentDirectory + "/test/ham/";
//    public static final String hamTestDataDir = currentDirectory + "/TestTest/ham/";
    public static final String spamTrainingDataDir = currentDirectory + "/train/spam/";
//    public static final String spamTrainingDataDir = currentDirectory + "/TestTraining/spam/";
    public static final String spamTestDataDir = currentDirectory + "/test/spam/";
//    public static final String spamTestDataDir = currentDirectory + "/TestTest/spam/";
    public static final String hamClassString = "ham";
    public static final String spamClassString = "spam";

    public static final String stopwordsFileName = "StopWords.txt";

    public static void main(String[] args) {

        //Defining the classes
        ArrayList<String> classes = new ArrayList<>();
        classes.add(hamClassString);
        classes.add(spamClassString);

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
            System.out.println("Error reading the stopwords file. Please make sure that the file is in the same folder as the java files.");
            Logger.getLogger(Assignment2.class.getName()).log(Level.SEVERE, null, ex);
        }
        //Multinomial Naive Bayes
        //Creating the vocabulary of training data
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
        System.out.println("Vocabulary successfully created");

        MultinomialNaiveBayes naiveBayesClassifier = new MultinomialNaiveBayes(vocabularyWithStopWords, nHam, nSpam, n, classes);
//        MultinomialNaiveBayes naiveBayesClassifier = new MultinomialNaiveBayes(vocabularyWithoutStopWords, nHam, nSpam, n, classes);

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
            System.out.println("Creating data for logistic classifier");
            
//            logisticRegressionClassifier = new LogisticRegression(vocabularyWithStopWords, nHam, nSpam, n, classes);
            logisticRegressionClassifier = new LogisticRegression(vocabularyWithoutStopWords, nHam, nSpam, n, classes);

            System.out.println("Data Creation complete. Starting training of Logistic Regression classifier");
            logisticRegressionClassifier.trainLogisticClassifier();
            System.out.println("Training the Logistic Regression Classifier was successful");
        } catch (Exception ex) {
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
//                    System.out.println("Classified as ham");
                    correctlyClassified++;
                }
                totalCount++;
            }
            File spamTestFolder = new File(spamTestDataDir);
            for (File spamTextFile : spamTestFolder.listFiles()) {
                reader = new BufferedReader(new FileReader(spamTextFile));
                if (logisticRegressionClassifier.applyLogisticRegressionClassifier(reader).equalsIgnoreCase(spamClassString)) {
//                    System.out.println("Classified as spam");
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
