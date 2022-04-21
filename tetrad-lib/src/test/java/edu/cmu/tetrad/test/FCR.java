package edu.cmu.tetrad.test;

import java.text.DecimalFormat;
import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;


public class FCR {

    public FCR() {
    }

  
    public static double measure(int[] truth, double[] probability, double T_plus, double T_minus) {
    	double fcr = 0.0;
        if (truth.length != probability.length) {
            throw new IllegalArgumentException(String.format("The vector sizes don't match: %d != %d.", truth.length, probability.length));
        }

        // for large sample size, overflow may happen for pos * neg.
        // switch to double to prevent it.
        double pos = 0;
        double neg = 0;

        for (int i = 0; i < truth.length; i++) {
            if (truth[i] == 0) {
                neg++;
            } else if (truth[i] == 1) {
                pos++;
            } else {
                throw new IllegalArgumentException("AUC is only for binary classification. Invalid label: " + truth[i]);
            }
        }

        int[] label = truth.clone();
        double[] prediction = probability.clone();

        QuickSort.sort(prediction, label);
        ArrayUtils.reverse(prediction);
        ArrayUtils.reverse(label);

        double C_plus = 0.0;
        double C_minus = 0.0;

        double[] c_minus_ratio = new double[label.length];
        double[] c_plus_ratio = new double[label.length];
        for (int i = 0; i < label.length; i++) {
        	int j = label.length-i-1;
        	if (i==0 ){
        		c_plus_ratio[i] = label[i] / (i+1);
        		c_minus_ratio[j] = label[j] / (i+1);
        		
        		if (c_plus_ratio[i] >= T_plus){
        			C_plus = i+1;
        		}
        		if (c_minus_ratio[j] <= T_minus){
        			C_minus = i+1;
        		}
        	}
        	else{
        		c_plus_ratio[i] = (c_plus_ratio[i-1]*(i) + label[i]) / (i+1);
        		c_minus_ratio[j] = (c_minus_ratio[j+1]*(i) + label[j]) / (i+1);
        		if (c_plus_ratio[i] >= T_plus){
        			C_plus = i+1;
        		}
        		if (c_minus_ratio[j] <= T_minus){
        			C_minus = i+1;
        		}
        	}
        	
        }
        
//        DecimalFormat df = new DecimalFormat("0.000");
//        System.out.println("label: ");
//        Arrays.stream(label).forEach(e -> System.out.print(df.format(e) + ", " ));
//        System.out.println("\n probs: ");
//        Arrays.stream(prediction).forEach(e -> System.out.print(df.format(e) + ", " ));
//        System.out.println("\n cminu: ");
//        Arrays.stream(c_minus_ratio).forEach(e -> System.out.print(df.format(e) + ", " ));
//        System.out.println("\n cplus: ");
//        Arrays.stream(c_plus_ratio).forEach(e -> System.out.print(df.format(e) + ", " ));
//        System.out.println("\n Cplus: " +C_plus);
//        System.out.println("\n Cminus: " +C_minus);
//        
        fcr = (C_plus + C_minus)/prediction.length;
        System.out.println("\n C_plus: " + C_plus+ ", C_minus: " + C_minus);
        return fcr;
    }
}
