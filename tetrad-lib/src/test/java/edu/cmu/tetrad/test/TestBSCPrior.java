
package edu.cmu.tetrad.test;


import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentMap;

import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.DirichletBayesIm;
import edu.cmu.tetrad.bayes.DirichletEstimator;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
//import edu.cmu.tetrad.util.RandomUtil;
//import edu.cmu.tetrad.util.TextTable;
import edu.cmu.tetrad.util.RandomUtil;

public class TestBSCPrior{		
	public static void main(String[] args) {

		TestBSCPrior t = new TestBSCPrior();
		t.test2();
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
		im.setProbability(2, 0, 0, 0.22);
		im.setProbability(2, 1, 0, 0.92);
		im.setProbability(2, 2, 0, 0.31);
		im.setProbability(2, 3, 0, 0.65);
		im.setProbability(2, 0, 1, 0.78);
		im.setProbability(2, 1, 1, 0.08);
		im.setProbability(2, 2, 1, 0.69);
		im.setProbability(2, 3, 1, 0.35);
		im.setProbability(3, 0, 0, 0.6);
		im.setProbability(3, 1, 0, 0.25);
		im.setProbability(3, 2, 0, 0.82);
		im.setProbability(3, 3, 0, 0.9);
		im.setProbability(3, 0, 1, 0.4);
		im.setProbability(3, 1, 1, 0.75);
		im.setProbability(3, 2, 1, 0.18);
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
	public void test2(){
		RandomUtil.getInstance().setSeed(1454140788L);
		int numVars = 20;
		double edgesPerNode = 3.0;
		int minCat = 2, maxCat = 3;
		int numCases = 1000;
		boolean threshold = false;
		double cutoff = 0.5;

		final int numEdges = (int) (numVars * edgesPerNode);
		IKnowledge knowledge = new Knowledge2();
		List<Node> vars = new ArrayList<>();
		int[] tiers = new int[numVars];
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
			tiers[i] = i;
			knowledge.addToTier(i, "X" + i);
		}
		int numSim = 100;
		int numBNs = 100;
		int[] truth = new int[numSim];
		boolean[] pred = new boolean[numSim];
		double[] pred_prob = new double[numSim];
		double[] pred_prob_new = new double[numSim];
		
		for (int i = 0; i < numBNs; i++){
			// generate true BN and its parameters
			Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, 0, numEdges, 30, 15, 15, false, true);
//			System.out.println("trueBN: " + trueBN);
			BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
			MlBayesIm im = new MlBayesIm(pm, MlBayesIm.RANDOM);

			// simulate train and test data from BN
			DataSet trainData = im.simulateData(numCases, false, tiers);
			trueBN = GraphUtils.replaceNodes(trueBN, trainData.getVariables());

			IndependenceTest dsep = new IndTestDSep(trueBN);
			boolean isDsep = dsep.isIndependent(trueBN.getNode("X1"), trueBN.getNode("X4"), trueBN.getNode("X0"));
			int inDsepInt = isDsep ? 1 : 0;
			truth[i] = inDsepInt;
//			System.out.println("isDsep: " + isDsep );

			IndTestProbabilistic indTestBsc = new IndTestProbabilistic(trainData);

			Node x = trainData.getVariable("X1");
			Node y = trainData.getVariable("X4");
			List<Node> z = new ArrayList<Node>();
			z.add(trainData.getVariable("X0"));
			
			boolean isInd = indTestBsc.isIndependent(x, y, z);
			pred_prob[i] = indTestBsc.getPosterior();
			DataSet[] dataSets = new DataSet[numSim];
			for (int bs = 0 ; bs < numSim ; bs++){
				dataSets[bs] = DataUtils.getBootstrapSample(trainData, numCases);
			}
			IndTestProbabilisticBootstrap indTestBsc_BS = new IndTestProbabilisticBootstrap(dataSets, numSim);
			boolean isInd_BS = indTestBsc_BS.isIndependent(x, y, z);
			pred[i] = isInd_BS;
			pred_prob_new[i] = indTestBsc_BS.getPosterior();
//			System.out.println("isInd: " + isInd );
			
//			Graph indBN = createBN(x, y, z, false);
//			Graph depBN = createBN(x, y, z, true);
//
//			BayesIm indIM = estimateParameters(trainData, indBN);
//			BayesIm depIM = estimateParameters(trainData, depBN);
//
//
//			double pInd_Ans = computePTruth_Ans(numSim, numCases, threshold, cutoff, x, y, z, indIM, depIM, isInd);
//			if (Double.isNaN(pInd_Ans)){
//				pInd_Ans = 0.5;
//			}
//			pred[i] = RandomUtil.getInstance().nextDouble() < pInd_Ans;
//			pred_prob_new[i] = pInd_Ans;
//			System.out.println("------------------");
		}
		double auroc = AUC.measure(truth, pred_prob);
		double auroc_new = AUC.measure(truth, pred_prob_new);
		System.out.println("auroc old method: " + auroc);
		System.out.println("auroc new method: " + auroc_new);
		System.out.println("truth: " + Arrays.toString(truth));
		System.out.println("pred_prob_old: " + Arrays.toString(pred_prob));
		System.out.println("pred_prob_new: " + Arrays.toString(pred_prob_new));


	}


	private double computePTruth_Ans(int numSim, int numCases, boolean threshold, double cutoff, Node x, Node y, List<Node> z,
			BayesIm indIM, BayesIm depIM, boolean isInd) {

		double ind_ind = 0.0, ind_dep = 0.0, dep_dep = 0.0, dep_ind = 0.0;		
		double[] indCounts = new double[numSim];
		double[] depCounts = new double[numSim];

		for (int i = 0; i < numSim; i++){
			DataSet indData = indIM.simulateData(numCases, false);
			IndTestProbabilistic indTest = new IndTestProbabilistic(indData);
			indTest.setThreshold(threshold);
			indTest.setCutoff(cutoff);
			List<Node> _z = new ArrayList<Node>();
			for (Node n : z){
				_z.add(indData.getVariable(n.getName()));
			}
			boolean isInd_ind = indTest.isIndependent(indData.getVariable(x.getName()), indData.getVariable(y.getName()), _z);
			if(isInd_ind){
				ind_ind += 1;
			}
			else{
				ind_dep += 1;			
			}
			indCounts[i] = indTest.getPosterior();

			// use the dep data 
			DataSet depData = depIM.simulateData(numCases, false);
			IndTestProbabilistic depTest = new IndTestProbabilistic(depData);
			depTest.setThreshold(threshold);
			depTest.setCutoff(cutoff);
			_z = new ArrayList<Node>();
			for (Node n : z){
				_z.add(depData.getVariable(n.getName()));
			}
			boolean isInd_dep = depTest.isIndependent(depData.getVariable(x.getName()), depData.getVariable(y.getName()), _z);
			if(isInd_dep){
				dep_ind += 1;
			}
			else{
				dep_dep +=1;
			}
			depCounts[i] = depTest.getPosterior();
		}
		double pAI_TI = ind_ind/numSim;
		double pAD_TI = ind_dep/numSim;

		double pAI_TD = dep_ind/numSim;
		double pAD_TD = dep_dep/numSim;

		double priorInd = 0.5, priorDep = 0.5;
		
		double pTI_AI = (pAI_TI * priorInd )/(pAI_TI * priorInd + pAI_TD * priorDep);
		double pTI_AD = (pAD_TI * priorInd )/(pAD_TI * priorInd + pAD_TD * priorDep);

		double pTD_AD = (pAD_TD * priorDep )/(pAD_TD * priorDep + pAD_TI * priorDep);
		double pTD_AI = (pAI_TD * priorDep )/(pAI_TD * priorDep + pAI_TI * priorDep);

//		if (Double.isNaN(pTI_AI)){
//			System.out.println("pAI_TI * priorInd " + (pAI_TI * priorInd));
//			System.out.println("pAI_TD * priorDep " + (pAI_TD * priorDep));
//			System.out.println("indProbs: " + Arrays.toString(indCounts));
//			System.out.println("depProbs: " + Arrays.toString(depCounts));
//
//		}
//		if (Double.isNaN(pTI_AD)){
//			System.out.println("pAD_TI * priorInd " + (pAD_TI * priorInd));
//			System.out.println("pAD_TD * priorDep " + (pAD_TD * priorDep));
//			System.out.println("indProbs: " + Arrays.toString(indCounts));
//			System.out.println("depProbs: " + Arrays.toString(depCounts));
//
//		}
//		System.out.println("indProbs: " + Arrays.toString(indCounts));
//		System.out.println("depProbs: " + Arrays.toString(depCounts));

//		System.out.println("pTI_AI: " + pTI_AI);
//		System.out.println("pTD_AI: " + pTD_AI);
//
//		System.out.println("pTD_AD: " + pTD_AD);
//		System.out.println("pTI_AD: " + pTI_AD);

		if (isInd){
			return pTI_AI;
		}
		else{
			return pTI_AD;
		}
	}


	private BayesIm estimateParameters(DataSet trainData, Graph indBN) {
		BayesPm indPM = new BayesPm(indBN);
		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(indPM, 1.0);
		BayesIm indIM = DirichletEstimator.estimate(prior, trainData);
		return indIM;
	}


	private Graph createBN(Node x, Node y, List<Node> z, boolean isDep) {
		List<Node> testVars = new ArrayList<Node>();
		testVars.add(x);
		testVars.add(y);
		testVars.addAll(z);
		Graph BN = new EdgeListGraph(testVars); 

		for (Node n : z){
			BN.addDirectedEdge(n, x);
			BN.addDirectedEdge(n, y);
			//			depBN.addDirectedEdge(n, x);
			//			depBN.addDirectedEdge(n, y);
		}
		if (isDep){
			BN.addDirectedEdge(x, y);
		}
		return BN;
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


