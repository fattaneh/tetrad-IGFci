
package edu.cmu.tetrad.test;


import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentMap;

import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
//import edu.cmu.tetrad.util.RandomUtil;
//import edu.cmu.tetrad.util.TextTable;

public class TestBSC {		
	public static void main(String[] args) {

		TestBSC t = new TestBSC();
		t.test1();
	}


	public void test1(){
		int numVars = 4;
		int numCases = 2000;
		int minCat = 2;
		int maxCat = 2;

		List<Node> vars = new ArrayList<>();
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}

		Graph dag = new EdgeListGraph(vars);
		dag.addDirectedEdge(dag.getNode("X0"), dag.getNode("X2"));
		dag.addDirectedEdge(dag.getNode("X1"), dag.getNode("X2"));
		dag.addDirectedEdge(dag.getNode("X2"), dag.getNode("X3"));
		dag.addDirectedEdge(dag.getNode("X1"), dag.getNode("X3"));
		BayesPm pm = new BayesPm(dag, minCat, maxCat);
		MlBayesIm im = new MlBayesIm(pm, MlBayesIm.MANUAL);
		im.setProbability(0, 0, 0, 0.75);
		im.setProbability(0, 0, 1, 0.25);
		im.setProbability(1, 0, 0, 0.62);
		im.setProbability(1, 0, 1, 0.38);
		im.setProbability(2, 0, 0, 0.92);
		im.setProbability(2, 1, 0, 0.92);
		im.setProbability(2, 2, 0, 0.31);
		im.setProbability(2, 3, 0, 0.65);
		im.setProbability(2, 0, 1, 0.08);
		im.setProbability(2, 1, 1, 0.08);
		im.setProbability(2, 2, 1, 0.69);
		im.setProbability(2, 3, 1, 0.35);
		im.setProbability(3, 0, 0, 0.6);
		im.setProbability(3, 1, 0, 0.25);
		im.setProbability(3, 2, 0, 0.9);
		im.setProbability(3, 3, 0, 0.9);
		im.setProbability(3, 0, 1, 0.4);
		im.setProbability(3, 1, 1, 0.75);
		im.setProbability(3, 2, 1, 0.1);
		im.setProbability(3, 3, 1, 0.1);

		System.out.println("IM:" + im);
		DataSet data = im.simulateData(numCases, false);
		DataSet test = im.simulateData(1, false);
		test.setDouble(0, 0, 0);
		test.setDouble(0, 1, 1);
		test.setDouble(0, 2, 1);
		test.setDouble(0, 3, 1);
		IndependenceTest ind = new IndTestProbabilistic(data);
		ind.isIndependent(data.getVariable("X2"), data.getVariable("X3"), data.getVariable("X1"));
		
		List<Integer> rows = new ArrayList<Integer>();
		System.out.println("Var in index 0: "+data.getVariable(0));
		for (int i = 0; i < data.getNumRows(); i++){
			if (data.getInt(i, 1) != test.getInt(0, 1)){
				rows.add(i);
			}	
		}
		System.out.println(data.getNumRows());
		data.removeRows(rows.stream().mapToInt(i->i).toArray());
		System.out.println(data.getNumRows());
		ind = new IndTestProbabilistic(data);
		ind.isIndependent(data.getVariable("X2"), data.getVariable("X3"), data.getVariable("X1"));
//		Rfci rfci = new Rfci(ind);
////		rfci.setVerbose(true);
//		Graph out = rfci.search();
//
//		System.out.println("Dag: "+dag);
//		System.out.println("out: " + out);
//		System.out.println("test: " + test);
//		for (Entry<IndependenceFact, Double> k: ((IndTestProbabilistic) ind).getH().entrySet()){
//			System.out.println(k.getKey() + ": " + k.getValue());
//		}

	}

	public void test3(){
		int numCases = 300;
		int minCat = 2;
		int maxCat = 2;

		List<Node> vars = new ArrayList<>();
		//			for (int i = 0; i < numVars; i++) {
		vars.add(new DiscreteVariable("Y"));
		vars.add(new DiscreteVariable("Z"));
		vars.add(new DiscreteVariable("X"));

		//			}

		Graph dag = new EdgeListGraph(vars);
		dag.addDirectedEdge(dag.getNode("Y"), dag.getNode("X"));
		dag.addDirectedEdge(dag.getNode("Z"), dag.getNode("X"));

		BayesPm pm = new BayesPm(dag, minCat, maxCat);
		MlBayesIm im = new MlBayesIm(pm, MlBayesIm.MANUAL);
		im.setProbability(0, 0, 0, 0.75);
		im.setProbability(0, 0, 1, 0.25);
		im.setProbability(1, 0, 0, 0.51);
		im.setProbability(1, 0, 1, 0.49);
		im.setProbability(2, 0, 0, 0.9);
		im.setProbability(2, 0, 1, 0.1);
		im.setProbability(2, 1, 0, 0.9);
		im.setProbability(2, 1, 1, 0.1);
		im.setProbability(2, 2, 0, 0.23);
		im.setProbability(2, 2, 1, 0.77);
		im.setProbability(2, 3, 0, 0.52);
		im.setProbability(2, 3, 1, 0.48);


		System.out.println("IM:" + im);
		DataSet data = im.simulateData(numCases, false);
		DataSet test = im.simulateData(1, false);
		test.setDouble(0, 0, 0);
		test.setDouble(0, 1, 1);
		test.setDouble(0, 2, 1);

		BDeuScore popScore = new BDeuScore(data);
		Fges popFges = new Fges (popScore);
		Graph outP = popFges.search();

		ISBDeuScore csi = new ISBDeuScore(data, test);
		ISFges fgs = new ISFges(csi);
		fgs.setPopulationGraph(SearchGraphUtils.chooseDagInPattern(outP));
		//			fgs.setInitialGraph(SearchGraphUtils.chooseDagInPattern(outP));
		Graph out = fgs.search();

		System.out.println("test: " +test);
		System.out.println("dag: "+dag);
		System.out.println("Pop: " + outP);
		System.out.println("IS: " + out + "\n");
		System.out.println("IS_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(out))+"\n");
		System.out.println("Pop_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(outP)));

	}

}


