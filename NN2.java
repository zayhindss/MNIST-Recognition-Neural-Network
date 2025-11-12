/*
 * CSC 475 - Neural Network (Part 2) Assignment #2
 * Name : Isaiah Hinds
 * CWID : 10391359
 * Date : 10/20/25
 * 
 * DESCRIPTION: 3-layer feedforward neural network for MNIST 0-9 digit recognition.
 * Implements SGD training, weight saving/loading, and ASCII art visualization.
 * Uses sigmoid activation function.
 * 
 * imported full java.io and java.util to save readability
 * libraries actually used:
 * 
 * from java.io:
 * filereader and bufferedreader for reading MNIST CSV files
 * filewriter and bufferedwriter for saving/loading weights
 * printwriter for formatted output to weight files
 * ioexception for handling file read/write errors
 * 
 * from java.util:
 * collections for random shuffling of training data
 * lists 
 * arrays for array manipulations
 * random for random number generation
 * scanner for reading user input
 */

import java.io.*;
import java.util.*;

public class NN2 {

    //HELPER FUNCTIONS//

    //sigmoid activation function
    static double sigmoid(double x){ 
        return 1.0 / (1.0 + Math.exp(-x)); 
    }

    //random in range [a,b)
    static double randRange(Random r, double a, double b) {
        return a + (b - a) * r.nextDouble(); 
    }

    //argmax
    //index of max value in array (converts ouput into predicted label)
    static int argMax(double[] v) {
        int idx = 0; double best = v[0];
        for (int i = 1; i < v.length; i++) if (v[i] > best) { best = v[i]; idx = i; }
        return idx;
    }

    //NETWORK PARAMETERS//
    static final int INPUT = 28 * 28; //784 pixel input
    static final int HIDDEN = 128; // hidden neurons
    static final int OUTPUT = 10; // digits 0-9
    static final double ETA = 3.0; //LR for SGD steps
    static final int BATCH = 10; //mini-batch size for SGD
    static final int EPOCHS = 30; //training epochs

    //FILE PATHS//
    static String TRAIN_CSV = "mnist_train.csv";
    static String TEST_CSV  = "mnist_test.csv";

    //NETWORK WEIGHTS AND BIASES//
    double[][] Wxh = new double[HIDDEN][INPUT]; //hidden x input weights
    double[][] Why = new double[OUTPUT][HIDDEN]; //output x hidden weights
    double[] bh = new double[HIDDEN]; //hidden biases
    double[] by = new double[OUTPUT]; //output biases

    //hidden pre-activations//
    double[] h = new double[HIDDEN]; //hidden activations
    double[] z = new double[HIDDEN]; //hidden sums
    //output activations//
    double[] y = new double[OUTPUT]; //output activations

    //SAMPLE CLASS//
    //simple value object to hold labeled samples
    //x is normalized pixel vector length=784, label is int 0-9
    static class Sample {
        final int label;
        final double[] x;
        Sample(int label, double[] x) { this.label = label; this.x = x; }
    }

    NN2(Random rng) {
        //randomly initialize Wxh and bh in [-1.0, 1.0)
        for (int j = 0; j < HIDDEN; j++) {
            for (int i = 0; i < INPUT; i++) Wxh[j][i] = randRange(rng, -1.0, 1.0);
            bh[j] = randRange(rng, -1.0, 1.0);
        }
        //randomly initialize Why and by in [-1.0, 1.0)
        for (int k = 0; k < OUTPUT; k++) {
            for (int j = 0; j < HIDDEN; j++) Why[k][j] = randRange(rng, -1.0, 1.0);
            by[k] = randRange(rng, -1.0, 1.0);
        }
    }

    //FORWARD PASS//
    //x -> hidden pre-acts -> hidden acts -> output pre-acts -> output acts
    //returns reference to y buffer (size OUTPUT)
    double[] forward(double[] x) {
        //hidden layer
        for (int j = 0; j < HIDDEN; j++) {
            double s = bh[j];
            for (int i = 0; i < INPUT; i++) s += Wxh[j][i] * x[i];
            z[j] = s; h[j] = sigmoid(s);
        }
        //output layer
        for (int k = 0; k < OUTPUT; k++) {
            double s = by[k];
            for (int j = 0; j < HIDDEN; j++) s += Why[k][j] * h[j];
            y[k] = sigmoid(s);
        }
        return y;
    }

    //SGD MINI-BATCH//
    //accumulates gradients over batch and updates weights/biases
    void sgdMiniBatch(List<Sample> batch) {
        //gradient accumulators initialized to zero
        double[][] gWxh = new double[HIDDEN][INPUT];
        double[][] gWhy = new double[OUTPUT][HIDDEN];
        double[] gbh = new double[HIDDEN];
        double[] gby = new double[OUTPUT];
        //error signals for output and hidden layers
        double[] deltaOut = new double[OUTPUT];
        double[] deltaHid = new double[HIDDEN];

        //accumulate gradients over mini-batch
        for (Sample s : batch) {
            //forward pass gets predicted output for this sample
            double[] predOut = forward(s.x);
            //BACKPROPAGATION//
            //output layer errors and gradients
            Arrays.fill(deltaOut, 0.0);
            for (int k = 0; k < OUTPUT; k++) {
                double t = (k == s.label) ? 1.0 : 0.0; //one-hot target
                deltaOut[k] = (predOut[k] - t) * (predOut[k] * (1.0 - predOut[k]));
                gby[k] += deltaOut[k];
                for (int j = 0; j < HIDDEN; j++) gWhy[k][j] += deltaOut[k] * h[j];
            }
            //hidden layer errors and gradients
            Arrays.fill(deltaHid, 0.0);
            for (int j = 0; j < HIDDEN; j++) {
                double sum = 0.0;
                for (int k = 0; k < OUTPUT; k++) sum += Why[k][j] * deltaOut[k];
                deltaHid[j] = sum * (h[j] * (1.0 - h[j]));
                gbh[j] += deltaHid[j];
                for (int i = 0; i < INPUT; i++) gWxh[j][i] += deltaHid[j] * s.x[i];
            }
        }

        //apply averaged gradients to weights and biases
        double scale = ETA / batch.size();
        
        //update Wxh and bh
        for (int j = 0; j < HIDDEN; j++) {
            for (int i = 0; i < INPUT; i++) Wxh[j][i] -= scale * gWxh[j][i];
            bh[j] -= scale * gbh[j];
        }
        //update Why and by
        for (int k = 0; k < OUTPUT; k++) {
            for (int j = 0; j < HIDDEN; j++) Why[k][j] -= scale * gWhy[k][j];
            by[k] -= scale * gby[k];
        }
    }

    //TRAINING FUNCTION//
    //shuffles training data and performs SGD over multiple epochs
    //uses arraylist buffer to shuffle data
    //prints accuracy after each epoch
    void train(List<Sample> train, int epochs, int batchSize, Random rng) {
        ArrayList<Sample> buf = new ArrayList<>(train);
        for (int e = 1; e <= epochs; e++) {
            Collections.shuffle(buf, rng);
            for (int i = 0; i < buf.size(); i += batchSize)
                sgdMiniBatch(buf.subList(i, Math.min(i + batchSize, buf.size())));
            int ok = 0;
            for (Sample s : train) if (argMax(forward(s.x)) == s.label) ok++;
            System.out.printf("Epoch %2d Accuracy = %d/%d (%.2f%%)%n", e, ok, train.size(), 100.0 * ok / train.size());
        }
    }

    //SAVE WEIGHTS//
    //uses print writer to write weights to file
    void save(String file) throws IOException {
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file)))) {
            pw.println("#SAVED WEIGHTS ");
            pw.printf("DIMENSIONS %d %d %d%n", INPUT, HIDDEN, OUTPUT);
            // Wxh: HIDDEN x INPUT
            pw.printf("Wxh %d %d%n", HIDDEN, INPUT);
            for (int j = 0; j < HIDDEN; j++) {
                for (int i = 0; i < INPUT; i++) {
                    if (i > 0) pw.print(' ');
                    pw.printf(java.util.Locale.US, "%.8f", Wxh[j][i]);
                }
                pw.println();
            }
            // bh: HIDDEN
            pw.printf("bh %d%n", HIDDEN);
            for (int j = 0; j < HIDDEN; j++) {
                if (j > 0) pw.print(' ');
                pw.printf(java.util.Locale.US, "%.8f", bh[j]);
            }
            pw.println();
            
            // Why: OUTPUT x HIDDEN
            pw.printf("Why %d %d%n", OUTPUT, HIDDEN);
            for (int k = 0; k < OUTPUT; k++) {
                for (int j = 0; j < HIDDEN; j++) {
                    if (j > 0) pw.print(' ');
                    pw.printf(java.util.Locale.US, "%.8f", Why[k][j]);
                }
                pw.println();
            }
            // by: OUTPUT
            pw.printf("by %d%n", OUTPUT);
            for (int k = 0; k < OUTPUT; k++) {
                if (k > 0) pw.print(' ');
                pw.printf(java.util.Locale.US, "%.8f", by[k]);
            }
            pw.println();
        }
    }

    //LOAD WEIGHTS//
    //loads weights from file saved by save()
    //uses buffered reader to read file line by line
    //lines starting with # are comments and ignored
    //goes through sections for Wxh, bh, Why, by
    void load(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        String section = "";
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            
            if (line.startsWith("Wxh")) {
                section = "Wxh";
                continue;
            } 
            else if (line.startsWith("bh")) {
                section = "bh";
                continue;
            } 
            else if (line.startsWith("Why")) {
                section = "Why";
                continue;
            } 
            else if (line.startsWith("by")) {
                section = "by";
                continue;
            } 
            else if (line.startsWith("dims")) {
                continue; 
            }

            String[] parts = line.split("\\s+");
            if (section.equals("Wxh")) {
                for (int j = 0; j < HIDDEN; j++) {
                    for (int i = 0; i < INPUT; i++) {
                        if (i < parts.length) Wxh[j][i] = Double.parseDouble(parts[i]);
                    }
                    line = br.readLine();
                    if (line == null) break;
                    parts = line.trim().split("\\s+");
                }
                section = "";
            } 
            else if (section.equals("bh")) {
                for (int j = 0; j < HIDDEN && j < parts.length; j++)
                    bh[j] = Double.parseDouble(parts[j]);
            }
            else if (section.equals("Why")) {
                for (int k = 0; k < OUTPUT; k++) {
                    for (int j = 0; j < HIDDEN; j++) {
                        if (j < parts.length) Why[k][j] = Double.parseDouble(parts[j]);
                    }
                    line = br.readLine();
                    if (line == null) break;
                    parts = line.trim().split("\\s+");
                }
                section = "";
            } 
            else if (section.equals("by")) {
                for (int k = 0; k < OUTPUT && k < parts.length; k++)
                    by[k] = Double.parseDouble(parts[k]);
            }
        }
        br.close();
    }

    //LOAD CSV DATA / READ FROM 784 PIXELS INPUT//
    //pixels are scaled to [0,1] by dividing by 255.0
    //uses buffered reader to read file line by line
    //each line split by commas, first value is label, rest are pixel values
    //stops after limit rows (limit>0)
    static List<Sample> loadCSV(String path, int limit) throws IOException {
        ArrayList<Sample> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] t = line.split(",");
                int label = Integer.parseInt(t[0]);
                double[] x = new double[INPUT];
                for (int i = 0; i < INPUT; i++) x[i] = Integer.parseInt(t[i + 1]) / 255.0;
                list.add(new Sample(label, x));
                if (limit > 0 && list.size() >= limit) break;
            }
        }
        return list;
    }

    //ASCII ART//
    static String toAscii(double[] x) {
        final char[] ramp = " .:-=+*#%@".toCharArray();
        StringBuilder sb = new StringBuilder(28 * 29);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                double v = x[r * 28 + c];
                sb.append(ramp[(int) Math.round(v * (ramp.length - 1))]);
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    //MAIN MENU LOOP//
    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        Random rng = new Random(1); //kept seed at 1 for testing consistency
        NN2 NN = new NN2(rng);
        List<Sample> train = null, test = null;
        boolean hasWeights = false;

        while (true) {
            System.out.println("\n // MENU //");
            System.out.println("[1] Train");
            System.out.println("[2] Load");
            System.out.println("[3] TRAINING accuracy");
            System.out.println("[4] TESTING accuracy");
            System.out.println("[5] Step through TESTING samples");
            System.out.println("[6] Show misclassified TESTING samples");
            System.out.println("[7] Save network state");
            System.out.println("[0] Exit");
            System.out.print("Choice: ");
            String choice = sc.nextLine().trim();

            switch (choice) {
                //TRAIN
                case "1":
                    if (train == null) train = loadCSV(TRAIN_CSV, 0);
                    NN.train(train, EPOCHS, BATCH, rng);
                    hasWeights = true; break;
                //LOAD
                case "2":
                    System.out.print("CHOOSE LOADED WEIGHTS FILE : ");
                    String f = sc.nextLine().trim();
                    if (f.isEmpty()){
                        System.out.println("ERROR: NO FILE ENTERED");
                        break;
                    }
                    else if (!new File(f).exists()){
                        System.out.println("ERROR: FILE DOESN'T EXIST");
                        break;
                    }
                    NN.load(f); hasWeights = true;
                    System.out.println("LOADED" + f); 
                    break;
                //TRAINING ACCURACY
                case "3":
                    if (!hasWeights) { 
                        System.out.println("ERROR : TRAIN OR LOAD FIRST."); 
                        break; 
                    }
                    if (train == null) train = loadCSV(TRAIN_CSV, 0);
                    eval(NN, train, "TRAIN"); 
                    break;
                //TESTING ACCURACY
                case "4":
                    if (!hasWeights) { 
                        System.out.println("ERROR : TRAIN OR LOAD FIRST."); 
                        break; 
                    }
                    if (test == null) test = loadCSV(TEST_CSV, 0);
                    eval(NN, test, "TEST"); 
                    break;
                //STEP THROUGH TESTING SAMPLES
                case "5":
                    if (!hasWeights) { 
                        System.out.println("ERROR : TRAIN OR LOAD FIRST."); 
                        break; 
                    }
                    if (test == null) test = loadCSV(TEST_CSV, 0);
                    step(NN, test, sc, false); 
                    break;
                //SHOW MISCLASSIFIED TESTING SAMPLES
                case "6":
                    if (!hasWeights) { 
                        System.out.println("ERROR : TRAIN OR LOAD FIRST."); 
                        break; 
                    }
                    if (test == null) test = loadCSV(TEST_CSV, 0);
                    step(NN, test, sc, true); 
                    break;
                //SAVE NETWORK STATE
                case "7":
                    if (!hasWeights) { 
                        System.out.println("ERROR : NO WEIGHTS TO SAVE. TRAIN OR LOAD FIRST."); 
                        break; 
                    }
                    System.out.print("SAVE CURRENT WEIGHT SET TO : ");
                    f = sc.nextLine().trim(); 
                    NN.save(f);
                    System.out.println("SAVED " + f);
                    break;
                //EXIT
                case "0": return;
                default: System.out.println("INVALID CHOICE"); break;
            }
        }
    }

    //EVALUATION FUNCTION//
    static void eval(NN2 NN, List<Sample> data, String tag) {
        int ok = 0;
        for (Sample s : data) if (argMax(NN.forward(s.x)) == s.label) ok++;
        System.out.printf("%s accuracy = %d/%d (%.2f%%)%n",
                tag, ok, data.size(), 100.0 * ok / data.size());
    }

    //STEP THROUGH SAMPLES//
    //if onlyWrong is true, only shows misclassified samples
    static void step(NN2 NN, List<Sample> data, Scanner sc, boolean onlyWrong) {
        for (int i = 0; i < data.size(); i++) {
            Sample s = data.get(i);
            double[] out = NN.forward(s.x);
            int pred = argMax(out);
            if (onlyWrong && pred == s.label) continue;
            System.out.println("\n#" + i + (pred != s.label ? "  WRONG" : ""));
            System.out.println(toAscii(s.x));
            System.out.printf("LABELED = %d // PREDICTED = %d%n", s.label, pred);
            System.out.print("[ENTER=next,Q=quit] ");
            if (sc.nextLine().trim().equalsIgnoreCase("q")) break;
        }
    }
}