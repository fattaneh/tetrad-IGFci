package edu.cmu.tetrad.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;
import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.search.returnObject;
public class TestISFges_CN {
	
	public static void main(String[] args) {
		TestISFges_CN tcn = new TestISFges_CN();
		String workingDirectory = System.getProperty("user.dir");
		System.out.println(workingDirectory);

		Path trainDataFile = Paths.get("/Users/fattanehjabbari/CCD-Project/CS-BN/causal_news/train.csv");
		Path testDataFile = Paths.get("/Users/fattanehjabbari/CCD-Project/CS-BN/causal_news/test.csv");

		char delimiter = ',';

		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
		VerticalDiscreteTabularDatasetReader testDataReader = new VerticalDiscreteTabularDatasetFileReader(testDataFile, DelimiterUtils.toDelimiter(delimiter));
		DataSet trainData = null, testData = null;
		try {
			trainData = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
			testData = (DataSet) DataConvertUtils.toDataModel(testDataReader.readInData());
//			System.out.println(trainData.getNumRows() +", " + trainData.getNumColumns());
//			System.out.println(testData.getNumRows() +", " + testData.getNumColumns());
		} catch (Exception IOException) {
			IOException.printStackTrace();
		}
		
		int[][] trainDataArray = dataSetToArray(trainData);
		int[][] testDataArray = dataSetToArray(testData);

		returnObject ro = tcn.runIFGES(trainDataArray, testDataArray, 0.1);
//		System.out.println(ro.instanceGraphs.size());
//		System.out.println(Arrays.deepToString(ro.probabilities));
	}
	private static int[][] dataSetToArray(DataSet dataSet) {
		int[][] data = new int[dataSet.getNumColumns()][];

		for (int j = 0; j < dataSet.getNumColumns(); j++) {
			data[j] = new int[dataSet.getNumRows()];

			for (int i = 0; i < dataSet.getNumRows(); i++) {
				data[j][i] = dataSet.getInt(i, j);
			}
		}
		return data;
	}	
	public returnObject runIFGES(int[][] trainMatrix, int[][] testMatrix, double kappa) {
		int numVars = trainMatrix.length;
//		System.out.println("numVars: " + numVars);
		List<Node> variables = new ArrayList<>();
		for (int i = 0; i < numVars - 1; i++) {
			variables.add(new DiscreteVariable("X" + i, 2));
		}
		variables.add(new DiscreteVariable("Y", 3));

		DataSet trainData = new BoxDataSet(new VerticalIntDataBox(trainMatrix), variables);
		DataSet testData = new BoxDataSet(new VerticalIntDataBox(testMatrix), variables);
		System.out.println(trainData.getNumRows() +", " + trainData.getNumColumns());
		System.out.println(testData.getNumRows() +", " + testData.getNumColumns());
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainData.getVariable(i).getName().equals("Y")){
				knowledge.addToTier(0, trainData.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(1, trainData.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);

		// learn the population model
		BDeuScore scoreP = new BDeuScore(trainData);
		Fges fgesP = new Fges (scoreP);
		fgesP.setKnowledge(knowledge);
		Graph dagP = fgesP.search();
		dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());
		
		// estimate MAP parameters from the population model
		BayesPm pmP = new BayesPm(dagP);

		DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
		BayesIm imP = DirichletEstimator.estimate(priorP, trainData);

		double k_add = kappa;
		double k_delete = kappa;
		double k_reverse = kappa;
		double[] llr = new double[testData.getNumRows()];
		double average = 0.0;

		Map <Key, Double> stats= new HashMap<Key, Double>();
		double[][] probabilities = new double[testData.getNumRows()][7];
		List<Graph> instanceGraphs = new ArrayList<Graph>();
//		System.out.println("disease, p0, p1, p2, p3, p4, p5");

		for (int i = 0; i < testData.getNumRows(); i++){
			DataSet test = testData.subsetRows(new int[]{i});
			
			// learn the instance-specific model
			ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
			scoreI.setKAddition(k_add);
			scoreI.setKDeletion(k_delete);
			scoreI.setKReorientation(k_reverse);
			ISFges fgesI = new ISFges(scoreI);
			fgesI.setKnowledge(knowledge);
			fgesI.setPopulationGraph(dagP);
			fgesI.setInitialGraph(dagP);
			Graph dagI = fgesI.search();
			dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
			GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(dagI, dagP);
			int n_a = cmp.getEdgesAdded().size();
			int n_d = cmp.getEdgesRemoved().size();
			Key cur_key = new Key(n_a, n_d);
			if(stats.get(cur_key)!=null)
				stats.put(cur_key, stats.get(cur_key)+1.0);
			else
				stats.put(cur_key, 1.0);
			
			// estimate MAP parameters from the instance-specific model
			BayesPm pmI = new BayesPm(dagI);
			DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
			BayesIm imI = DirichletEstimator.estimate(priorI, trainData);
			fgesI.setPopulationGraph(dagP);
			llr[i] = fgesI.scoreDag(dagI) - fgesI.scoreDag(dagP);
			instanceGraphs.add(dagP);
			int yIndex_p = imP.getNodeIndex(imP.getNode("Y"));
			int[] parents_p = imP.getParents(yIndex_p);
			Arrays.sort(parents_p);
			int[] values_p = new int[parents_p.length];
			
			for (int no = 0; no < parents_p.length; no++){
				values_p [no] = test.getInt(0, parents_p[no]);
			}
			double prob0_p = imP.getProbability(yIndex_p, imP.getRowIndex(yIndex_p, values_p), 0);
			double prob1_p = imP.getProbability(yIndex_p, imP.getRowIndex(yIndex_p, values_p), 1);			
			double prob2_p = imP.getProbability(yIndex_p, imP.getRowIndex(yIndex_p, values_p), 2);
			
			int yIndex_i = imI.getNodeIndex(imI.getNode("Y"));
			int[] parents_i = imI.getParents(yIndex_i);
			Arrays.sort(parents_i);
			int[] values_i = new int[parents_i.length];
		
			for (int no = 0; no < parents_i.length; no++){
				values_i [no] = test.getInt(0, parents_i[no]);
			}
			double prob0_i = imI.getProbability(yIndex_i, imI.getRowIndex(yIndex_i, values_i), 0);
			double prob1_i = imI.getProbability(yIndex_i, imI.getRowIndex(yIndex_i, values_i), 1);
			double prob2_i = imI.getProbability(yIndex_i, imI.getRowIndex(yIndex_i, values_i), 2);

			List<Node> parents_i_list = new ArrayList<Node>();
			for (int no = 0; no < parents_i.length; no++){
				parents_i_list.add(imI.getNode(parents_i[no]));
			}
			
//			System.out.println(test.getInt(0,yIndex_i) +", " + prob0_p + ", " + prob1_p + ", " + prob2_p + ", "+ prob0_i+ ", "+ prob1_i+ ", "+ prob2_i);
			probabilities[i][0] = test.getInt(0,yIndex_i);
			probabilities[i][1] = prob0_p;
			probabilities[i][2] = prob1_p;
			probabilities[i][3] = prob2_p;
			probabilities[i][4] = prob0_i;
			probabilities[i][5] = prob1_i;
			probabilities[i][6] = prob2_i;
//			System.out.println("-----------------");
			average += llr[i];
		}
//		for (Key key : stats.keySet()){
//			System.out.println(key.print(key) + ":" + (stats.get(key)/testData.getNumRows())*100);
//		}
//		System.out.println(average/testData.getNumRows());
		return new returnObject(instanceGraphs, dagP, probabilities);
	}
}
