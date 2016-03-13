
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

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
//    ArrayList<Double[]> wList;
    double[] wList; //only one because we only have 2 classes
    double lambda = 0.025;
    double eta = 0.001;
    int gradientAscentIterations = 14;
    List<HashMap<String, Integer>> X;
    List<String> Y;

    public LogisticRegression(HashSet<String> vocabulary, int nHam, int nSpam, int n, ArrayList<String> classes) throws FileNotFoundException, IOException {
        this.vocabulary = vocabulary;
        this.nHam = nHam;
        this.nSpam = nSpam;
        this.n = n;
        this.classes = classes;
//        wList = new ArrayList<>();
        wList = new double[vocabulary.size() + 1];
        //Initializing the parameters for each of the classes
//        for (String className : classes) {
//            Double[] w = new Double[vocabulary.size() + 1];
//            for (int i = 0; i < vocabulary.size() + 1; i++) {
//                double zero = 0;
//                w[i] = zero;
//            }
//            wList.add(w);
//        }
        X = new ArrayList<>();
        Y = new ArrayList<>();
        int count = 0;
        BufferedReader reader = null;
        File hamTrainingFolder = new File(Assignment2.hamTrainingDataDir);
        for (File hamTextFile : hamTrainingFolder.listFiles()) {
            Tct = new HashMap<>();
            for (String term : vocabulary) {
                Tct.put(term, 0);
            }
//            System.out.println("reading file ===>" + hamTextFile.getName());
            reader = new BufferedReader(new FileReader(hamTextFile));
            String line = reader.readLine();
            while (line != null) {
                String terms[] = line.split(" ");
                for (String term : terms) {
                    Tct.put(term, Tct.get(term) + 1);
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
//            System.out.println("reading file ===>" + spamTextFile.getName());
            reader = new BufferedReader(new FileReader(spamTextFile));
            String line = reader.readLine();
            while (line != null) {
                String terms[] = line.split(" ");
                for (String term : terms) {
                    Tct.put(term, Tct.get(term) + 1);
                }
                line = reader.readLine();
            }
            X.add(Tct);
            Y.add(Assignment2.spamClassString);
        }
    }

    public void trainLogisticClassifier() throws Exception {

//        for (String className : classes) {
//            //Estimating parameters for each
//            int iterations = gradientAscentIterations;
//            do {
//                double[] prob = new double[n];
//                for (int i = 0; i < n; i++) {
//                    prob[i] = estimate(X.get(i), className);
//                }
////                System.out.println("Estimates===>");
////                for (Double d : prob) {
////                    System.out.print(d + "\t");
////                }
////                System.out.println("");
////                System.out.println("");
//
//                double gradient[] = new double[vocabulary.size()];
//                int i = 0;
//                for (String key : vocabulary) {
//                    gradient[i] = 0;
//                    for (int j = 0; j < n; j++) {
//                        int delta = (className.equals(Y.get(j))) ? 1 : 0;
//                        gradient[i] += ((double) X.get(j).get(key) * ((double) delta - prob[j]));
//                    }
//                    gradient[i] -= ((double) lambda * wList.get(classes.indexOf(className))[i] * wList.get(classes.indexOf(className))[i]);
//                    wList.get(classes.indexOf(className))[i] += ((double) eta * gradient[i]);
//                    i++;
//                    if (i == vocabulary.size()) {
//                        break;
//                    }
//                }
//                
////                for(double d : gradient) {
////                    System.out.print(d + " ");
////                }
////                System.out.println("");
////                System.out.println("W values == ");
////                for (Double[] w : wList) {
////                    for (int j = 0; j < w.length; j++) {
////                        System.out.print(w[j] + "\t");
////                    }
////                    System.out.println("");
////                    System.out.println("");
////                }
////                System.out.println("");
////                System.out.println("");
//            } while (--iterations > 0);
//        }
        int iterations = gradientAscentIterations;
        do {
            for (int i = 0; i < n; i++) {
                HashMap<String, Integer> dataInstance = X.get(i);
                String className = Y.get(i);

                //Estimate weights
                double prob = estimate(dataInstance, className);

//                System.out.println("Probability ==>" + prob);
//                System.out.println("i==" + i);

                int j = 1;
                for (String key : vocabulary) {
                    int delta = (className.equals(Assignment2.hamClassString)) ? 1 : 0;
                    wList[j] += (eta * (double) dataInstance.get(key) * ((double) delta - prob)) - (eta * lambda * wList[j] * wList[j]);
                    j++;
                }
            }

//            System.out.println("Weights == >");
//            for (double d : wList) {
//                System.out.print(d + ",");
//            }
//            System.out.println("");
            System.out.println("Done for iteration " + iterations);
        } while (--iterations > 0);
    }

    private double estimate(HashMap<String, Integer> X, String className) throws Exception{
//        double parameterValue = wList.get(classes.indexOf(className))[0];
        double parameterValue = wList[0];

        int i = 1;
        for (String key : vocabulary) {
//            parameterValue += (wList.get(classes.indexOf(className))[i++] * (double) X.get(key));
            parameterValue += (wList[i++] * (double) X.get(key));
        }

//        System.out.println("Param Value = " + parameterValue);
        double expVal = Math.exp(parameterValue);

        if (Double.isNaN(expVal)) {
            //Exception for when the value of sum(wi*xi) increased greater than that Math.exp can handle
            throw new Exception("Please change the values of eta and lambda for proper results.");
        }

//        System.out.println("Exp Val = " + expVal);
//        if (Double.POSITIVE_INFINITY == expVal || expVal == Double.NaN) {
//            System.out.println(parameterValue);
//            for(String key : X.keySet()) {
//                if(X.get(key)!=0)
//                    System.out.print(key + "==>"+X.get(key) + " ");
//            }
//            System.out.println("");
//        }
//        System.out.println("Estimate = " + ((double) expVal / ((double) 1 + expVal)));
//        if (className.equals(Assignment2.spamClassString)) {
//            return 1- ((double) 1 / (double) 1 + expVal);
//        }
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

//        System.out.println(prob[0] + ", " + prob[1]);
        return classes.get(index);

    }
}
