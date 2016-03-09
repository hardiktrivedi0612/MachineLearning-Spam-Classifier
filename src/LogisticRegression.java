
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

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
    int gradientAscentIterations = 1;

    public LogisticRegression(HashSet<String> vocabulary, int nHam, int nSpam, int n, ArrayList<String> classes) {
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
        Tct = new HashMap<>();
        for (String term : vocabulary) {
            Tct.put(term, 0);
        }
    }

    public void trainLogisticClassifier() throws FileNotFoundException, IOException {
        for (String className : classes) {
            //Estimating wi 

            for (int i = 0; i < gradientAscentIterations; i++) {
                for (int j = 1; j < vocabulary.size() + 1; j++) {
                    double total = 0;

                    BufferedReader reader = null;
                    File hamTrainingFolder = new File(Assignment2.hamTrainingDataDir);
                    for (File hamTextFile : hamTrainingFolder.listFiles()) {
                        System.out.println("reading file ===>" + hamTextFile.getName());
                        reader = new BufferedReader(new FileReader(hamTextFile));
                        String line = reader.readLine();
                        while (line != null) {
                            String terms[] = line.split(" ");
                            for (String term : terms) {
                                Tct.put(term, Tct.get(term) + 1);
                            }
                            line = reader.readLine();
                        }

                        double estimate = estimate(Tct, className);
                        int delta = (className.equals(Assignment2.hamClassString)) ? 1 : 0;

                        for (String key : Tct.keySet()) {
                            total += (Tct.get(key) * (delta - estimate));
                        }

                        //Make Tct zero again
                        for (String term : vocabulary) {
                            Tct.put(term, 0);
                        }
                    }
                    File spamTrainingFolder = new File(Assignment2.spamTrainingDataDir);
                    for (File spamTextFile : spamTrainingFolder.listFiles()) {
                        reader = new BufferedReader(new FileReader(spamTextFile));
                        String line = reader.readLine();
                        while (line != null) {
                            String terms[] = line.split(" ");
                            for (String term : terms) {
                                Tct.put(term, Tct.get(term) + 1);
                            }
                            line = reader.readLine();
                        }
                        double estimate = estimate(Tct, className);
                        int delta = (className.equals(Assignment2.spamClassString)) ? 1 : 0;

                        for (String key : Tct.keySet()) {
                            total += (Tct.get(key) * (delta - estimate));
                        }
                        //Make Tct zero again
                        for (String term : vocabulary) {
                            Tct.put(term, 0);
                        }
                    }
                    wList.get(classes.indexOf(className))[j] = wList.get(classes.indexOf(className))[j] + (eta * total) - (lambda * eta * wList.get(classes.indexOf(className))[j]);
                }
            }
        }
        for(Double[] w : wList) {
            for (int i = 0; i < w.length; i++) {
                System.out.print(w[i]+"\t");
            }
            System.out.println("");
        }
    }

    private double estimate(HashMap<String, Integer> Tct, String className) {
        if (className.equals(Assignment2.hamClassString)) {
            double parameterValue = wList.get(0)[0];
            for (int i = 1; i < vocabulary.size() + 1; i++) {
                parameterValue += (wList.get(0)[i] * Tct.get((String) vocabulary.toArray()[i - 1]));
            }
            double expVal = Math.pow(Math.E, (parameterValue));
            return ((double) 1 / ((double) 1 + expVal));
        } else {
            double parameterValue = wList.get(1)[0];
            for (int i = 1; i < vocabulary.size() + 1; i++) {
                parameterValue += (wList.get(1)[i] * Tct.get((String) vocabulary.toArray()[i - 1]));
            }
            double expVal = Math.pow(Math.E, (parameterValue));
            return (expVal / ((double) 1 + expVal));
        }
    }
    
}
