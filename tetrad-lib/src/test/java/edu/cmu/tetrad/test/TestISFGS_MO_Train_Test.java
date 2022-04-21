package edu.cmu.tetrad.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.test.Key;
import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFGS_MO_Train_Test {
	public static void main(String[] args) {

		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/PPT_data/";

		String dataName = "ppt"; 
		String pathToTrainData = pathToFolder + dataName + "_train_june28.csv";
		String pathToTestData = pathToFolder + dataName + "_test_june28.csv";

		String target1 = "Pain_Response";
		String target2 = "PhysFunction_Response";
		String target3 = "ImpChange_Response";

		// Read in the data
		DataSet trainData = readData(pathToTrainData);
		DataSet testData = readData(pathToTestData);	

		// Create the knowledge
		IKnowledge knowledge = createKnowledge_multipleOutcome (trainData, target1, target2, target3);

		// learn the population model using all training data
		double samplePrior = 1.0;
		double structurePrior = 1.0;
		Graph graphP = BNlearn_pop(trainData, knowledge, samplePrior, structurePrior);
		System.out.println("Pop graph:" + graphP.getEdges());

		double T_plus = 0.9;
		double T_minus = 0.1;

		for (int p = 1; p <= 10; p++){

			double k_add =  p/10.0; //Math.pow(10, -1.0*p);

			System.out.println("kappa = " + k_add);

			double[] probs_is_1 = new double[testData.getNumRows()];
			double[] probs_p_1 = new double[testData.getNumRows()];
			int[] truth_1 = new int[testData.getNumRows()];
			
			double[] probs_is_2 = new double[testData.getNumRows()];
			double[] probs_p_2 = new double[testData.getNumRows()];
			int[] truth_2 = new int[testData.getNumRows()];
			
			double[] probs_is_3 = new double[testData.getNumRows()];
			double[] probs_p_3 = new double[testData.getNumRows()];
			int[] truth_3 = new int[testData.getNumRows()];
			
			Map <Key, Double> stats= new HashMap<Key, Double>();
			// PrintStream out;
			PrintStream outForAUC;
			try {
				File dir = new File( pathToFolder + "/outputs/" + dataName);
				dir.mkdirs();
				String outputFileName = dataName + "-AUROC-Kappa"+ k_add +".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			
			outForAUC.println("Pain_Response, PhysFunction_Response, ImpChange_Response, P1_pop, P2_pop, P3_pop, P1_IS, P2_IS, P3_IS");

			// loop over test cases
			for (int i = 0; i < testData.getNumRows(); i++){
				 if(i%100 == 0){
					 System.out.println(i);
				 }

				DataSet test = testData.subsetRows(new int[]{i});

				// learn the IS graph
				Graph graphI = learnBNIS(trainData, test, k_add, graphP, knowledge, samplePrior);

				// compute probability distribution of the target variable
				int[] targetIndices = new int[3];
				targetIndices[0] = trainData.getColumn(trainData.getVariable(target1));
				targetIndices[1] = trainData.getColumn(trainData.getVariable(target2)); 
				targetIndices[2] = trainData.getColumn(trainData.getVariable(target3)); 

				truth_1[i] = test.getInt(0, targetIndices[0]);
				truth_2[i] = test.getInt(0, targetIndices[1]);
				truth_3[i] = test.getInt(0, targetIndices[2]);


				//get the prob from IS model
				probs_is_1[i] = estimation(trainData, test, graphI, target1);
				probs_is_2[i] = estimation(trainData, test, graphI, target2);
				probs_is_3[i] = estimation(trainData, test, graphI, target3);

				//get the prob from population model
				probs_p_1[i] = estimation(trainData, test, graphP, target1);
				probs_p_2[i] = estimation(trainData, test, graphP, target2);
				probs_p_3[i] = estimation(trainData, test, graphP, target3);
						 
				//graph comparison
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();
				Key cur_key = new Key(n_a, n_d);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);

				outForAUC.println(test.getInt(0, targetIndices[0]) +", " + test.getInt(0, targetIndices[1]) + ", " + test.getInt(0, targetIndices[2]) 
				+", " +  probs_p_1[i] + ", " +  probs_p_2[i] + ", " +  probs_p_3[i]+ ", " + 
						probs_is_1[i] + ", " + probs_is_2[i] + ", " + probs_is_3[i]);
			}
			
			//// target 1
			System.out.println("------------ Pain_Response -------------");
			double auroc_p_1 = AUC.measure(truth_1, probs_p_1);
			double auroc_1 = AUC.measure(truth_1, probs_is_1);
			double fcr_p_1 = FCR.measure(truth_1, probs_p_1, T_plus, T_minus);
			double fcr_1 = FCR.measure(truth_1, probs_is_1, T_plus, T_minus);
			System.out.println( "AUROC_Pop: "+ auroc_p_1);
			System.out.println( "AUROC_IS : "+ auroc_1);
			System.out.println( "FCR_Pop: "+ fcr_p_1);
			System.out.println( "FCR_IS : "+ fcr_1);

			//// target 2
			System.out.println("------------ PhysFunction_Response -------------");
			double auroc_p_2 = AUC.measure(truth_2, probs_p_2);
			double auroc_2 = AUC.measure(truth_2, probs_is_2);
			double fcr_p_2 = FCR.measure(truth_2, probs_p_2, T_plus, T_minus);
			double fcr_2 = FCR.measure(truth_2, probs_is_2, T_plus, T_minus);
			System.out.println( "AUROC_Pop: "+ auroc_p_2);
			System.out.println( "AUROC_IS : "+ auroc_2);
			System.out.println( "FCR_Pop: "+ fcr_p_2);
			System.out.println( "FCR_IS : "+ fcr_2);

			////  target 3
			System.out.println("------------ ImpChange_Response -------------");
			double auroc_p_3 = AUC.measure(truth_3, probs_p_3);
			double auroc_3 = AUC.measure(truth_3, probs_is_3);
			double fcr_p_3 = FCR.measure(truth_3, probs_p_3, T_plus, T_minus);
			double fcr_3 = FCR.measure(truth_3, probs_is_3, T_plus, T_minus);
			System.out.println( "AUROC_Pop: "+ auroc_p_3);
			System.out.println( "AUROC_IS : "+ auroc_3);
			System.out.println( "FCR_Pop: "+ fcr_p_3);
			System.out.println( "FCR_IS : "+ fcr_3);



			for (Key k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);
			}
			System.out.println("-----------------");
			outForAUC.close();
		}
	}

	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge, double samplePrior){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
		scoreI.setSamplePrior(samplePrior);
		scoreI.setKAddition(kappa);
		scoreI.setKDeletion(kappa);
		scoreI.setKReorientation(kappa);
		ISFges fgesI = new ISFges(scoreI);
		fgesI.setKnowledge(knowledge);
		fgesI.setPopulationGraph(graphP);
		fgesI.setInitialGraph(graphP);
		Graph graphI = fgesI.search();
		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
		return graphI;
	}

	private static double estimation(DataSet trainData, DataSet test, Graph graph, String target){

		DagInPatternIterator iterator = new DagInPatternIterator(graph);
		Graph dag = iterator.next();
		dag = GraphUtils.replaceNodes(dag, trainData.getVariables());
		BayesPm pm = new BayesPm(dag);

		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pm, 1.0);
		BayesIm im = DirichletEstimator.estimate(prior, trainData);

		int targetIndex = im.getNodeIndex(im.getNode(target));

		int[] parents = im.getParents(targetIndex);
		Arrays.sort(parents);
		int[] values = new int[parents.length];
		for (int no = 0; no < parents.length; no++){
			values [no] = test.getInt(0, parents[no]);
		}
		double prob = im.getProbability(targetIndex, im.getRowIndex(targetIndex, values), 1);
		return prob;
	}
//	private static double[] estimation(DataSet trainData, DataSet test, Graph graph, String target){
//
//		DagInPatternIterator iterator = new DagInPatternIterator(graph);
//		Graph dag = iterator.next();
//		dag = GraphUtils.replaceNodes(dag, trainData.getVariables());
//		BayesPm pm = new BayesPm(dag);
//
//		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pm, 1.0);
//		BayesIm im = DirichletEstimator.estimate(prior, trainData);
//
//		int targetIndex = im.getNodeIndex(im.getNode(target));
//		
//		double prob[] = new double[3];
//		int[] parents = im.getParents(targetIndex);
//		Arrays.sort(parents);
//
//		
//		int[] values = new int[parents.length];
//		for (int no = 0; no < parents.length; no++){
//			values [no] = test.getInt(0, parents[no]);
//		}
//		for (int i = 0; i < 3; i++){
//			prob[i] = im.getProbability(targetIndex, im.getRowIndex(targetIndex, values), i);
//		}
//		return prob;
//	}

	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double samplePrior, double structurePrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setSamplePrior(samplePrior);
		scoreP.setStructurePrior(structurePrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setKnowledge(knowledge);
		fgesP.setSymmetricFirstStep(true);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		return graphP;
	}

	private static IKnowledge createKnowledge_multipleOutcome(DataSet trainDataOrig, String target1, String target2, String target3) {
		int numVars = trainDataOrig.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (trainDataOrig.getVariable(i).getName().equals(target1) || trainDataOrig.getVariable(i).getName().equals(target2) || trainDataOrig.getVariable(i).getName().equals(target3)){
				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);
		knowledge.setTierForbiddenWithin(1, true);

		return knowledge;
	}
	private static DataSet readData(String pathToData) {
		Path trainDataFile = Paths.get(pathToData);
		char delimiter = ',';
		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
		DataSet trainDataOrig = null;
		try {
			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
			System.out.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
		} catch (Exception IOException) {
			IOException.printStackTrace();
		}
		return trainDataOrig;
	}
	private static Map<String, Double> sortByValue(Map<String, Double> dEGdist, final boolean order)
	{
		List<Entry<String, Double>> list = new LinkedList<>(dEGdist.entrySet());

		// Sorting the list based on values
		list.sort((o1, o2) -> order ? o1.getValue().compareTo(o2.getValue()) == 0
				? o1.getKey().compareTo(o2.getKey())
						: o1.getValue().compareTo(o2.getValue()) : o2.getValue().compareTo(o1.getValue()) == 0
						? o2.getKey().compareTo(o1.getKey())
								: o2.getValue().compareTo(o1.getValue()));
		return list.stream().collect(Collectors.toMap(Entry::getKey, Entry::getValue, (a, b) -> b, LinkedHashMap::new));

	}
}
//package edu.cmu.tetrad.test;
//
//import java.io.File;
//import java.io.FileOutputStream;
//import java.io.PrintStream;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.Arrays;
//import java.util.HashMap;
//import java.util.LinkedHashMap;
//import java.util.LinkedList;
//import java.util.List;
//import java.util.Map;
//import java.util.Map.Entry;
//import java.util.stream.Collectors;
//
//import edu.cmu.tetrad.bayes.*;
//import edu.cmu.tetrad.data.DataSet;
//import edu.cmu.tetrad.data.IKnowledge;
//import edu.cmu.tetrad.data.Knowledge2;
//import edu.cmu.tetrad.graph.*;
//import edu.cmu.tetrad.search.*;
//import edu.cmu.tetrad.test.Key;
// import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
//import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
//import edu.cmu.tetrad.util.DataConvertUtils;
//import edu.cmu.tetrad.util.DelimiterUtils;
//
//public class TestISFges_RealData {
//	public static void main(String[] args) {
//
//		String pathToTrainData = "/Users/fattanehjabbari/CCD-Project/CS-BN/chronic_pancreatitis_fattane/train.csv";
//		String pathToTestData = "/Users/fattanehjabbari/CCD-Project/CS-BN/chronic_pancreatitis_fattane/test.csv";
//		DataSet trainData = readData(pathToTrainData);
//		DataSet testData = readData(pathToTestData);
//		String target = "disease";
//
//		
//		// Create the knowledge
//		IKnowledge knowledge = createKnowledge(trainData, target);
//		
//		// learn the population model
//		double samplePrior = 1.0;
//		Graph graphP = BNlearn_pop(trainData, knowledge, samplePrior);
//		System.out.println("Pop graph:" + graphP.getEdges());
//
//		// estimate MAP parameters from the population model
//		DagInPatternIterator iterator = new DagInPatternIterator(graphP);
//		Graph dagP = iterator.next();
//		dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());
//		BayesPm pmP = new BayesPm(dagP);
//		DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
//		BayesIm imP = DirichletEstimator.estimate(priorP, trainData);
//		
//		double T_plus = 0.9;
//		double T_minus = 0.1;
//		//outer LOOCV
//		for (int p = 1; p <= 10; p++){
//			
//			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
////			double k_delete = k_add;
////			double k_reverse = k_add;
//			System.out.println("kappa = " + k_add);
//			
//		double[] probs_is = new double[testData.getNumRows()];
//		double[] probs_p = new double[testData.getNumRows()];
//		int[] truth = new int[testData.getNumRows()];
//
//		PrintStream out;
//		PrintStream outForAUC;
//		try {
//			File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/CP/outputs");
//			dir.mkdirs();
//			String outputFileName = "CP-Kappa"+ k_add +".csv";
//
//			File file = new File(dir, outputFileName);
//			out = new PrintStream(new FileOutputStream(file));
//			outputFileName = "CP-AUROC-Kappa"+ k_add +".csv";
//
//			File fileAUC = new File(dir, outputFileName);
//			outForAUC = new PrintStream(new FileOutputStream(fileAUC));
//		} catch (Exception e) {
//			throw new RuntimeException(e);
//		}
//
//		Map <Key, Double> stats= new HashMap<Key, Double>();
//		Map <String, Double> DEGdist= new HashMap<String, Double>();
//		for (int i = 0; i < testData.getNumColumns()-1; i++){
//			DEGdist.put(testData.getVariable(i).getName(), 0.0);
//		}
//
//		outForAUC.println("y, population-FGES, instance-specific-FGES");
//		out.println("features, fraction of occurance in cases");
//		System.out.println("------------- STARTING IS SEARCH --------------");
//		for (int i = 0; i < testData.getNumRows(); i++){//trainDataOrig.getNumRows()
//			
//			DataSet test67 = testData.subsetRows(new int[]{i});
//			
//			dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());			
//			truth[i] = test67.getInt(0,imP.getNodeIndex(imP.getNode(target)));
//			Graph graphI = learnBNIS(trainData, test67, k_add, graphP, knowledge, samplePrior);
//			probs_is[i]= estimation(trainData, test67, graphI, target);
//			
//			
//			//get the p from population model
//			int diseaseIndex_p = imP.getNodeIndex(imP.getNode(target));
//			imP = DirichletEstimator.estimate(priorP, trainData);
//			double prob_p = estimation_pop(imP, test67, diseaseIndex_p);
//			probs_p[i] = prob_p; 
//			
//			//graph comparison
//			GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
//			int n_a = cmp.getEdgesAdded().size();
//			int n_d = cmp.getEdgesRemoved().size();
//			
//			Key cur_key = new Key(n_a, n_d);
//			if(stats.get(cur_key)!=null)
//				stats.put(cur_key, stats.get(cur_key) + 1.0);
//			else
//				stats.put(cur_key, 1.0);
//	
//			outForAUC.println(test67.getInt(0,diseaseIndex_p) +", " + probs_p[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
//		}
//		double auroc = AUC.measure(truth, probs_is);
//		double auroc_p = AUC.measure(truth, probs_p);
//
//		double fcr = FCR.measure(truth, probs_is, T_plus, T_minus);
//		double fcr_p = FCR.measure(truth, probs_p, T_plus, T_minus);
//
//		System.out.println( "AUROC_P: "+ auroc_p);
//		System.out.println( "AUROC: "+ auroc);
//		System.out.println( "FCR: "+ fcr);
//		System.out.println( "FCR_P: "+ fcr_p);
//
//		
//		for (Key k : stats.keySet()){
//			System.out.println(k.print(k) + ":" + (stats.get(k)/trainData.getNumRows())*100);
//		}
//		System.out.println("-----------------");
//		}
//	}
//	private static double estimation_pop(BayesIm imP, DataSet test67, int diseaseIndex_p) {
//		int[] parents_p = imP.getParents(diseaseIndex_p);
//		Arrays.sort(parents_p);
//		int[] values_p = new int[parents_p.length];
//		for (int no = 0; no < parents_p.length; no++){
//			values_p [no] = test67.getInt(0, parents_p[no]);
//		}
//		double prob_p = imP.getProbability(diseaseIndex_p, imP.getRowIndex(diseaseIndex_p, values_p), 1);
//		return prob_p;
//	}
//	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge, double samplePrior){
//		// learn the instance-specific model
//		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
//		scoreI.setSamplePrior(samplePrior);
//		scoreI.setKAddition(kappa);
//		scoreI.setKDeletion(kappa);
//		scoreI.setKReorientation(kappa);
//		ISFges fgesI = new ISFges(scoreI);
//		fgesI.setSymmetricFirstStep(true);
//		fgesI.setFaithfulnessAssumed(true);
////		fgesI.setVerbose(true);
//		fgesI.setKnowledge(knowledge);
//		fgesI.setPopulationGraph(graphP);
//		fgesI.setInitialGraph(graphP);
//		Graph graphI = fgesI.search();
//		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
//		return graphI;
//	}
//	
//	private static double estimation(DataSet trainData, DataSet test, Graph graphI, String target){
//
//		DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
//		Graph dagI = iteratorI.next();
//		dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
//		BayesPm pmI = new BayesPm(dagI);
//
//		DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
//		BayesIm imI = DirichletEstimator.estimate(priorI, trainData);
//
//		int diseaseIndex_i = imI.getNodeIndex(imI.getNode(target));
////		int truth = test.getInt(0,diseaseIndex_i);
//    	
//		double p = estimation_pop(imI, test, diseaseIndex_i);
//		
//		return p;
//	}
//	private static double kappaTuning(DataSet trainData67, IKnowledge knowledge, Graph dagP, String target){
//		//inner LOOCV
//		int kvalues = 9;
//		double[] kappa_tune = new double[kvalues+1];
//		double[] auroc_tune = new double[kvalues+1];
//		for (int p = 0; p <= kvalues; p++){	
//			double k_add =  (p+1)/10.0; 
//			double k_delete = k_add;
//			double k_reverse = k_add;
//			kappa_tune[p]= k_add;
//			System.out.println("kappa: " + k_add);
//
//			double[] probs_is_tune = new double[trainData67.getNumRows()];
//			int[] truth_tune = new int[trainData67.getNumRows()];
//			for (int j = 0; j < trainData67.getNumRows(); j++){
//				DataSet trainData66 = trainData67.copy();
//				DataSet test66 = trainData67.subsetRows(new int[]{j});
//				trainData66.removeRows(new int[]{j});
//
//				// learn the instance-specific model
////				System.out.println("trainData66: "+trainData66.getNumRows());
//				ISBDeuScore scoreI = new ISBDeuScore(trainData66, test66);
//				scoreI.setKAddition(k_add);
//				scoreI.setKDeletion(k_delete);
//				scoreI.setKReorientation(k_reverse);
//				ISFges fgesI = new ISFges(scoreI);
//				fgesI.setKnowledge(knowledge);
//				fgesI.setPopulationGraph(dagP);
//				fgesI.setInitialGraph(dagP);
//				Graph graphI = fgesI.search();
//				graphI = GraphUtils.replaceNodes(graphI, trainData66.getVariables());
//
//				// learn a pop model from data + test
//				DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
//				Graph dagI = iteratorI.next();
//				dagI = GraphUtils.replaceNodes(dagI, trainData66.getVariables());
//				BayesPm pmI = new BayesPm(dagI);
//
//				DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
//				BayesIm imI = DirichletEstimator.estimate(priorI, trainData66);
//				fgesI.setPopulationGraph(dagP);
//
//				int diseaseIndex_i = imI.getNodeIndex(imI.getNode(target));
//				truth_tune [j] = test66.getInt(0,diseaseIndex_i);
//
//				double prob_i = estimation_pop(imI, test66, diseaseIndex_i);
//				probs_is_tune[j] = prob_i; 
//			}
//
//			auroc_tune[p] = FCR.measure(truth_tune, probs_is_tune, 0.85, 0.15);
////			System.out.println( "AUROC: "+ auroc_tune[p]);
//		}
//
//		QuickSort.sort(auroc_tune, kappa_tune);
//
//
//		double best_kappa = kappa_tune[kvalues];
//		double best_auroc = auroc_tune[kvalues];
//		if (best_auroc == 0.0){
//			best_kappa = 0.3;
//		}
//		System.out.println( "BEST fcr: "+ best_auroc + ", BEST kappa: " + best_kappa);
//		return best_kappa;
//	}
//	
//	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double samplePrior) {
//		BDeuScore scoreP = new BDeuScore(trainDataOrig);
//		scoreP.setSamplePrior(samplePrior);
//		Fges fgesP = new Fges (scoreP);
//		fgesP.setKnowledge(knowledge);
//		fgesP.setFaithfulnessAssumed(true);
//		Graph graphP = fgesP.search();
//		return graphP;
//	}
//	private static IKnowledge createKnowledge(DataSet trainDataOrig, String target) {
//		int numVars = trainDataOrig.getNumColumns();
//		IKnowledge knowledge = new Knowledge2();
//		int[] tiers = new int[2];
//		tiers[0] = 0;
//		tiers[1] = 1;
//		for (int i=0 ; i< numVars; i++) {
//			if (!trainDataOrig.getVariable(i).getName().equals(target)){
//				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
//			}
//			else{
//				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
//			}
//		}
//		knowledge.setTierForbiddenWithin(0, true);
//		return knowledge;
//	}
//	private static DataSet readData(String pathToData) {
//		Path trainDataFile = Paths.get(pathToData);
//		char delimiter = ',';
//		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
//		DataSet trainDataOrig = null;
//		try {
//			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
//			System.out.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
//		} catch (Exception IOException) {
//			IOException.printStackTrace();
//		}
//		return trainDataOrig;
//	}
//	private static Map<String, Double> sortByValue(Map<String, Double> dEGdist, final boolean order)
//	{
//		List<Entry<String, Double>> list = new LinkedList<>(dEGdist.entrySet());
//
//		// Sorting the list based on values
//		list.sort((o1, o2) -> order ? o1.getValue().compareTo(o2.getValue()) == 0
//				? o1.getKey().compareTo(o2.getKey())
//						: o1.getValue().compareTo(o2.getValue()) : o2.getValue().compareTo(o1.getValue()) == 0
//						? o2.getKey().compareTo(o1.getKey())
//								: o2.getValue().compareTo(o1.getValue()));
//		return list.stream().collect(Collectors.toMap(Entry::getKey, Entry::getValue, (a, b) -> b, LinkedHashMap::new));
//
//	}
//}