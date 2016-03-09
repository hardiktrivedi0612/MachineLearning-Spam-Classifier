
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
    ArrayList<Double[]> wList;
    double lambda = 0.1;
    double eta = 0.01;
    int gradientAscentIterations = 2;
    List<HashMap<String, Integer>> X;
    List<String> Y;

    public LogisticRegression(HashSet<String> vocabulary, int nHam, int nSpam, int n, ArrayList<String> classes) throws FileNotFoundException, IOException {
        this.vocabulary = vocabulary;
        this.nHam = nHam;
        this.nSpam = nSpam;
        this.n = n;
        this.classes = classes;
        wList = new ArrayList<>();
        //Initializing the parameters for each of the classes
        for (String className : classes) {
            Double[] w = new Double[vocabulary.size() + 1];
            for (int i = 0; i < vocabulary.size() + 1; i++) {
                double zero = 0;
                w[i] = zero;
            }
            wList.add(w);
        }
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

    public void trainLogisticClassifier() throws FileNotFoundException, IOException {

        for (String className : classes) {
            //Estimating parameters for each
            int iterations = gradientAscentIterations;
            do {
                double[] prob = new double[n];
                for (int i = 0; i < n; i++) {
                    prob[i] = estimate(X.get(i), className);
                }
                System.out.println("Estimates===>");
                for (Double d : prob) {
                    System.out.print(d + "\t");
                }
                System.out.println("");
                System.out.println("");

                double gradient[] = new double[vocabulary.size()];
                int i = 0;
                for (String key : vocabulary) {
                    gradient[i] = 0;
                    for (int j = 0; j < n; j++) {
                        int delta = (className.equals(Y.get(j))) ? 1 : 0;
                        gradient[i] += ((double) X.get(j).get(key) * ((double) delta - prob[j]));
                    }
                    gradient[i] -= ((double) lambda * wList.get(classes.indexOf(className))[i]);
                    wList.get(classes.indexOf(className))[i] += ((double) eta * gradient[i]);
                    i++;
                    if (i == vocabulary.size()) {
                        break;
                    }
                }
//                for (Double[] w : wList) {
//                    for (int j = 0; j < w.length; j++) {
//                        System.out.print(w[j] + "\t");
//                    }
//                    System.out.println("");
//                    System.out.println("");
//                }
//                System.out.println("");
//                System.out.println("");
            } while (--iterations > 0);
        }
    }

    private double estimate(HashMap<String, Integer> X, String className) {
        double parameterValue = wList.get(classes.indexOf(className))[0];

        int i = 1;
        for (String key : vocabulary) {
            parameterValue += (wList.get(classes.indexOf(className))[i++] * (double) X.get(key));
        }

//        System.out.println("Param Value = " + parameterValue);
        double expVal = Math.exp(parameterValue);
//        System.out.println("Estimate = " + ((double) expVal / ((double) 1 + expVal)));
        return ((double) expVal / ((double) 1 + expVal));
    }

    public String applyLogisticRegressionClassifier(BufferedReader documentReader) throws IOException {
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
