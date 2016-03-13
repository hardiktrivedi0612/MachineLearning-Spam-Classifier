
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Hardik
 */
public class LogisticRegression {

    HashSet<String> vocabulary;
    HashMap<String, Integer> Tct;
    int nHam, nSpam, n;
    ArrayList<String> classes;
//    ArrayList<Double[]> weights;
    double[] weights; //only one because we only have 2 classes
    double lambda = 0.075;
    double eta = 0.001;
    int gradientAscentIterations = 13;
    List<HashMap<String, Integer>> X;
    List<String> Y;

    public LogisticRegression(HashSet<String> vocabulary, int nHam, int nSpam, int n, ArrayList<String> classes) throws FileNotFoundException, IOException {
        this.vocabulary = vocabulary;
        this.nHam = nHam;
        this.nSpam = nSpam;
        this.n = n;
        this.classes = classes;
        weights = new double[vocabulary.size() + 1];
        X = new ArrayList<>();
        Y = new ArrayList<>();
        BufferedReader reader = null;
        File hamTrainingFolder = new File(Assignment2.hamTrainingDataDir);
        for (File hamTextFile : hamTrainingFolder.listFiles()) {
            Tct = new HashMap<>();
            for (String term : vocabulary) {
                Tct.put(term, 0);
            }
            reader = new BufferedReader(new FileReader(hamTextFile));
            String line = reader.readLine();
            while (line != null) {
                String terms[] = line.split(" ");
                for (String term : terms) {
                    if (vocabulary.contains(term)) {
                        Tct.put(term, Tct.get(term) + 1);
                    }
                }
                line = reader.readLine();
            }

            X.add(Tct);
            Y.add(Assignment2.hamClassString);
        }
        File spamTrainingFolder = new File(Assignment2.spamTrainingDataDir);
        for (File spamTextFile : spamTrainingFolder.listFiles()) {
            Tct = new HashMap<>();
            for (String term : vocabulary) {
                Tct.put(term, 0);
            }
            reader = new BufferedReader(new FileReader(spamTextFile));
            String line = reader.readLine();
            while (line != null) {
                String terms[] = line.split(" ");
                for (String term : terms) {
                    if (vocabulary.contains(term)) {
                        Tct.put(term, Tct.get(term) + 1);
                    }
                }
                line = reader.readLine();
            }
            X.add(Tct);
            Y.add(Assignment2.spamClassString);
        }
    }

    public void trainLogisticClassifier() throws Exception {

        int iterations = gradientAscentIterations;
        do {
            for (int i = 0; i < n; i++) {
                HashMap<String, Integer> dataInstance = X.get(i);
                String className = Y.get(i);

                //Estimate weights
                double prob = estimate(dataInstance, className);

                int j = 1;
                for (String key : vocabulary) {
                    int delta = (className.equals(Assignment2.hamClassString)) ? 1 : 0;
                    weights[j] += (eta * (double) dataInstance.get(key) * ((double) delta - prob)) - (eta * lambda * weights[j] * weights[j]);
                    j++;
                }
            }
        } while (--iterations > 0);
    }

    private double estimate(HashMap<String, Integer> X, String className) throws Exception {
        double parameterValue = weights[0];

        int i = 1;
        for (String key : vocabulary) {
            parameterValue += (weights[i++] * (double) X.get(key));
        }

        double expVal = Math.exp(parameterValue);

        if (Double.isNaN(expVal)) {
            //Exception for when the value of sum(wi*xi) increased greater than that Math.exp can handle
            throw new Exception("Please change the values of eta and lambda for proper results.");
        }

        double returnValue = (expVal / ((double) 1 + expVal));
        if (className.equals(Assignment2.hamClassString)) {
            return returnValue;
        } else {
            return 1 - returnValue;
        }
    }

    public String applyLogisticRegressionClassifier(BufferedReader documentReader) throws Exception {
        if (documentReader == null) {
            return null;
        }

        Tct = new HashMap<>();
        for (String term : vocabulary) {
            Tct.put(term, 0);
        }
        String line = documentReader.readLine();
        while (line != null) {
            String terms[] = line.split(" ");
            for (String term : terms) {
                if (vocabulary.contains(term)) {
                    Tct.put(term, Tct.get(term) + 1);
                }
            }
            line = documentReader.readLine();
        }

        double prob[] = new double[classes.size()];
        int i = 0;
        for (String className : classes) {
            prob[i] = estimate(Tct, className);
            i++;
        }

        int index = 0;
        double max = prob[0];
        for (int j = 0; j < prob.length; j++) {
            if (prob[j] > max) {
                max = prob[j];
                index = j;
            }
        }
        return classes.get(index);

    }
}
