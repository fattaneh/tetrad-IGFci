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

public class TestISFGES_MELANOMA_NESTED_LOO {
	public static void main(String[] args) {

		String pathToData = "/Users/fattanehjabbari/CCD-Project/CS-BN/TDI_DEG/DEGmatrix.UPMCcell4greg.TDIDEGfeats.csv";
		DataSet trainDataOrig = readData(pathToData);

		// Create the knowledge
		IKnowledge knowledge = createKnowledge(trainDataOrig);

		// learn the population model
		double structPrior = 1.0;
		Graph graphP = BNlearn_pop(trainDataOrig, knowledge, structPrior);
		System.out.println("Pop graph:" + graphP.getEdges());

		// estimate MAP parameters from the population model
		DagInPatternIterator iterator = new DagInPatternIterator(graphP);
		Graph dagP = iterator.next();
		dagP = GraphUtils.replaceNodes(dagP, trainDataOrig.getVariables());
		BayesPm pmP = new BayesPm(dagP);
		DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
		BayesIm imP = DirichletEstimator.estimate(priorP, trainDataOrig);

		
		double T_plus = 0.9;
		double T_minus = 0.1;
		//outer LOOCV
		for (int p = 1; p <= 10; p++){
			
			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
//			double k_delete = k_add;
//			double k_reverse = k_add;
			System.out.println("kappa = " + k_add);
			
		double[] probs_is = new double[trainDataOrig.getNumRows()];
		double[] probs_p = new double[trainDataOrig.getNumRows()];

		int[] truth = new int[trainDataOrig.getNumRows()];

		PrintStream out;
		PrintStream outForAUC;
		try {
			File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/TDI-DEG/");
			dir.mkdirs();
			String outputFileName = "MelanomaTDI-DEG-Distribution-Kappa"+ k_add +".csv";

			File file = new File(dir, outputFileName);
			out = new PrintStream(new FileOutputStream(file));
			outputFileName = "MelanomaTDI-AUROC-Kappa"+ k_add +".csv";

			File fileAUC = new File(dir, outputFileName);
			outForAUC = new PrintStream(new FileOutputStream(fileAUC));
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		Map <Key, Double> stats= new HashMap<Key, Double>();
		Map <String, Double> DEGdist= new HashMap<String, Double>();
		for (int i = 0; i < trainDataOrig.getNumColumns()-1; i++){
			DEGdist.put(trainDataOrig.getVariable(i).getName(), 0.0);
		}

		outForAUC.println("PD1response, population-FGES, instance-specific-FGES");//, DEGs");
		out.println("DEGs, fraction of occurance in cases");
		
		for (int i = 0; i < trainDataOrig.getNumRows(); i++){
			
			DataSet trainData67 = trainDataOrig.copy();
			DataSet test67 = trainDataOrig.subsetRows(new int[]{i});
			trainData67.removeRows(new int[]{i});
			
			dagP = GraphUtils.replaceNodes(dagP, trainData67.getVariables());			
			truth[i] = test67.getInt(0,imP.getNodeIndex(imP.getNode("PD1response")));
			Graph graphI = learnBNIS(trainData67, test67, k_add, graphP, knowledge);
			probs_is[i]= estimation(trainData67, test67, graphI);
			
			
			//get the p from population model
			int diseaseIndex_p = imP.getNodeIndex(imP.getNode("PD1response"));
			double prob_p = estimation_pop(imP, test67, diseaseIndex_p);
			probs_p[i] = prob_p; 
			
			//graph comparison
			GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
			int n_a = cmp.getEdgesAdded().size();
			int n_d = cmp.getEdgesRemoved().size();
			Key cur_key = new Key(n_a, n_d);
			if(stats.get(cur_key)!=null)
				stats.put(cur_key, stats.get(cur_key) + 1.0);
			else
				stats.put(cur_key, 1.0);
	
			outForAUC.println(test67.getInt(0,diseaseIndex_p) +", " + probs_p[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
		}
		double auroc = AUC.measure(truth, probs_is);
		double fcr = FCR.measure(truth, probs_is, T_plus, T_minus);

		System.out.println( "AUROC: "+ auroc);
		System.out.println( "FCR: "+ fcr);

		for (Key k : stats.keySet()){
			System.out.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);
		}
		System.out.println("-----------------");
		}
	}
	private static double estimation_pop(BayesIm imP, DataSet test67, int diseaseIndex_p) {
		int[] parents_p = imP.getParents(diseaseIndex_p);
		Arrays.sort(parents_p);
		int[] values_p = new int[parents_p.length];
		for (int no = 0; no < parents_p.length; no++){
			values_p [no] = test67.getInt(0, parents_p[no]);
		}
		double prob_p = imP.getProbability(diseaseIndex_p, imP.getRowIndex(diseaseIndex_p, values_p), 1);
		return prob_p;
	}
	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
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
	
	private static double estimation(DataSet trainData, DataSet test, Graph graphI){

		DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
		Graph dagI = iteratorI.next();
		dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
		BayesPm pmI = new BayesPm(dagI);

		DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
		BayesIm imI = DirichletEstimator.estimate(priorI, trainData);

		int diseaseIndex_i = imI.getNodeIndex(imI.getNode("PD1response"));
//		int truth = test.getInt(0,diseaseIndex_i);
    	
		double p = estimation_pop(imI, test, diseaseIndex_i);
		
		return p;
	}
	private static double kappaTuning(DataSet trainData67, IKnowledge knowledge, Graph dagP){
		//inner LOOCV
		int kvalues = 9;
		double[] kappa_tune = new double[kvalues+1];
		double[] auroc_tune = new double[kvalues+1];
		for (int p = 0; p <= kvalues; p++){	
			double k_add =  (p+1)/10.0; 
			double k_delete = k_add;
			double k_reverse = k_add;
			kappa_tune[p]= k_add;
			System.out.println("kappa: " + k_add);

			double[] probs_is_tune = new double[trainData67.getNumRows()];
			int[] truth_tune = new int[trainData67.getNumRows()];
			for (int j = 0; j < trainData67.getNumRows(); j++){
				DataSet trainData66 = trainData67.copy();
				DataSet test66 = trainData67.subsetRows(new int[]{j});
				trainData66.removeRows(new int[]{j});

				// learn the instance-specific model
//				System.out.println("trainData66: "+trainData66.getNumRows());
				ISBDeuScore scoreI = new ISBDeuScore(trainData66, test66);
				scoreI.setKAddition(k_add);
				scoreI.setKDeletion(k_delete);
				scoreI.setKReorientation(k_reverse);
				ISFges fgesI = new ISFges(scoreI);
				fgesI.setKnowledge(knowledge);
				fgesI.setPopulationGraph(dagP);
				fgesI.setInitialGraph(dagP);
				Graph graphI = fgesI.search();
				graphI = GraphUtils.replaceNodes(graphI, trainData66.getVariables());

				// learn a pop model from data + test
				DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
				Graph dagI = iteratorI.next();
				dagI = GraphUtils.replaceNodes(dagI, trainData66.getVariables());
				BayesPm pmI = new BayesPm(dagI);

				DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
				BayesIm imI = DirichletEstimator.estimate(priorI, trainData66);
				fgesI.setPopulationGraph(dagP);

				int diseaseIndex_i = imI.getNodeIndex(imI.getNode("PD1response"));
				truth_tune [j] = test66.getInt(0,diseaseIndex_i);

				double prob_i = estimation_pop(imI, test66, diseaseIndex_i);
				probs_is_tune[j] = prob_i; 
			}

			auroc_tune[p] = FCR.measure(truth_tune, probs_is_tune, 0.85, 0.15);
//			System.out.println( "AUROC: "+ auroc_tune[p]);
		}

		QuickSort.sort(auroc_tune, kappa_tune);


		double best_kappa = kappa_tune[kvalues];
		double best_auroc = auroc_tune[kvalues];
		if (best_auroc == 0.0){
			best_kappa = 0.3;
		}
		System.out.println( "BEST fcr: "+ best_auroc + ", BEST kappa: " + best_kappa);
		return best_kappa;
	}
	
	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double structPrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setStructurePrior(structPrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setKnowledge(knowledge);
		fgesP.setSymmetricFirstStep(true);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		return graphP;
	}
	private static IKnowledge createKnowledge(DataSet trainDataOrig) {
		int numVars = trainDataOrig.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainDataOrig.getVariable(i).getName().equals("PD1response")){
				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);
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

// neste LOOCV with FCR
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
//import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;
//import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDataReader;
//import edu.cmu.tetrad.util.DataConvertUtils;
//import edu.cmu.tetrad.util.DelimiterUtils;
//
//public class TestISFGES_MELANOMA_NESTED_LOO {
//	public static void main(String[] args) {
//
//		String pathToData = "/Users/fattanehjabbari/CCD-Project/CS-BN/TDI_DEG/DEGmatrix.UPMCcell4greg.TDIDEGfeats.csv";
//		DataSet trainDataOrig = readData(pathToData);
//
//		// Create the knowledge
//		IKnowledge knowledge = createKnowledge(trainDataOrig);
//
//		// learn the population model
//		double structPrior = 1.0;
//		Graph graphP = BNlearn_pop(trainDataOrig, knowledge, structPrior);
//		System.out.println("Pop graph:" + graphP.getEdges());
//
//		// estimate MAP parameters from the population model
//		DagInPatternIterator iterator = new DagInPatternIterator(graphP);
//		Graph dagP = iterator.next();
//		dagP = GraphUtils.replaceNodes(dagP, trainDataOrig.getVariables());
//		BayesPm pmP = new BayesPm(dagP);
//		DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
//		BayesIm imP = DirichletEstimator.estimate(priorP, trainDataOrig);
//
//		//outer LOOCV
//		double[] probs_is = new double[trainDataOrig.getNumRows()];
//		double[] probs_p = new double[trainDataOrig.getNumRows()];
//
//		int[] truth = new int[trainDataOrig.getNumRows()];
//
//		PrintStream out;
//		PrintStream outForAUC;
//		try {
//			File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/TDI-DEG/");
//			dir.mkdirs();
//			String outputFileName = "MelanomaTDI-DEG-Distribution.csv";
//
//			File file = new File(dir, outputFileName);
//			out = new PrintStream(new FileOutputStream(file));
//			outputFileName = "MelanomaTDI-AUROC.csv";
//
//			File fileAUC = new File(dir, outputFileName);
//			outForAUC = new PrintStream(new FileOutputStream(fileAUC));
//		} catch (Exception e) {
//			throw new RuntimeException(e);
//		}
//
//		Map <Key, Double> stats= new HashMap<Key, Double>();
//		Map <String, Double> DEGdist= new HashMap<String, Double>();
//		for (int i = 0; i < trainDataOrig.getNumColumns()-1; i++){
//			DEGdist.put(trainDataOrig.getVariable(i).getName(), 0.0);
//		}
//
//		outForAUC.println("PD1response, population-FGES, instance-specific-FGES");//, DEGs");
//		out.println("DEGs, fraction of occurance in cases");
//		
//		for (int i = 0; i < trainDataOrig.getNumRows(); i++){
//
//			System.out.println("i = " + i);
//			
//			DataSet trainData67 = trainDataOrig.copy();
//			DataSet test67 = trainDataOrig.subsetRows(new int[]{i});
//			trainData67.removeRows(new int[]{i});
//			
//			dagP = GraphUtils.replaceNodes(dagP, trainData67.getVariables());
//			double best_kappa = kappaTuning(trainData67, knowledge, dagP);
//			
//			truth[i] = test67.getInt(0,imP.getNodeIndex(imP.getNode("PD1response")));
//			Graph graphI = learnBNIS(trainData67, test67, best_kappa, graphP, knowledge);
//			probs_is[i]= estimation(trainData67, test67, graphI);
//			
//			
//			//get the p from population model
//			int diseaseIndex_p = imP.getNodeIndex(imP.getNode("PD1response"));
//			double prob_p = estimation_pop(imP, test67, diseaseIndex_p);
//			probs_p[i] = prob_p; 
//			
//			//graph comparison
//			GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
//			int n_a = cmp.getEdgesAdded().size();
//			int n_d = cmp.getEdgesRemoved().size();
//			Key cur_key = new Key(n_a, n_d);
//			if(stats.get(cur_key)!=null)
//				stats.put(cur_key, stats.get(cur_key) + 1.0);
//			else
//				stats.put(cur_key, 1.0);
//	
//			outForAUC.println(test67.getInt(0,diseaseIndex_p) +", " + probs_p[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
//		}
//		double auroc = AUC.measure(truth, probs_is);
//		double fcr = FCR.measure(truth, probs_is, 0.85, 0.15);
//
//		System.out.println( "AUROC: "+ auroc);
//		System.out.println( "FCR: "+ fcr);
//
//		for (Key k : stats.keySet()){
//			System.out.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);
//		}
//		System.out.println("-----------------");
//
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
//	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge){
//		// learn the instance-specific model
//		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
//		scoreI.setKAddition(kappa);
//		scoreI.setKDeletion(kappa);
//		scoreI.setKReorientation(kappa);
//		ISFges fgesI = new ISFges(scoreI);
//		fgesI.setKnowledge(knowledge);
//		fgesI.setPopulationGraph(graphP);
//		fgesI.setInitialGraph(graphP);
//		Graph graphI = fgesI.search();
//		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
//		return graphI;
//	}
//	
//	private static double estimation(DataSet trainData, DataSet test, Graph graphI){
//
//		DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
//		Graph dagI = iteratorI.next();
//		dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
//		BayesPm pmI = new BayesPm(dagI);
//
//		DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
//		BayesIm imI = DirichletEstimator.estimate(priorI, trainData);
//
//		int diseaseIndex_i = imI.getNodeIndex(imI.getNode("PD1response"));
////		int truth = test.getInt(0,diseaseIndex_i);
//
//		double p = estimation_pop(imI, test, diseaseIndex_i);
//		
//		return p;
//	}
//	private static double kappaTuning(DataSet trainData67, IKnowledge knowledge, Graph dagP){
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
//				int diseaseIndex_i = imI.getNodeIndex(imI.getNode("PD1response"));
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
//	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double structPrior) {
//		BDeuScore scoreP = new BDeuScore(trainDataOrig);
//		scoreP.setStructurePrior(structPrior);
//		Fges fgesP = new Fges (scoreP);
//		fgesP.setKnowledge(knowledge);
//		fgesP.setSymmetricFirstStep(true);
//		Graph graphP = fgesP.search();
//		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
//		return graphP;
//	}
//	private static IKnowledge createKnowledge(DataSet trainDataOrig) {
//		int numVars = trainDataOrig.getNumColumns();
//		IKnowledge knowledge = new Knowledge2();
//		int[] tiers = new int[2];
//		tiers[0] = 0;
//		tiers[1] = 1;
//		for (int i=0 ; i< numVars; i++) {
//			if (!trainDataOrig.getVariable(i).getName().equals("PD1response")){
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
//		TabularDataReader trainDataReader = new VerticalDiscreteTabularDataReader(trainDataFile.toFile(), DelimiterUtils.toDelimiter(delimiter));
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
}

// neste LOOCV with AUC

//package edu.cmu.tetrad.test;
//
//import java.io.File;
//import java.io.FileOutputStream;
//import java.io.PrintStream;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.ArrayList;
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
//import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;
//import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDataReader;
//import edu.cmu.tetrad.util.DataConvertUtils;
//import edu.cmu.tetrad.util.DelimiterUtils;
//
//public class TestISFGES_MELANOMA_NESTED_LOO {
//	public static void main(String[] args) {
//
//		String pathToData = "/Users/fattanehjabbari/CCD-Project/CS-BN/TDI_DEG/DEGmatrix.UPMCcell4greg.TDIDEGfeats.csv";
//		DataSet trainDataOrig = readData(pathToData);
//
//		// Create the knowledge
//		IKnowledge knowledge = createKnowledge(trainDataOrig);
//
//		// learn the population model
//		double structPrior = 1.0;
//		Graph graphP = BNlearn_pop(trainDataOrig, knowledge, structPrior);
//		System.out.println("Pop graph:" + graphP.getEdges());
//
//		// estimate MAP parameters from the population model
//		DagInPatternIterator iterator = new DagInPatternIterator(graphP);
//		Graph dagP = iterator.next();
//		dagP = GraphUtils.replaceNodes(dagP, trainDataOrig.getVariables());
//		BayesPm pmP = new BayesPm(dagP);
//		DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
//		BayesIm imP = DirichletEstimator.estimate(priorP, trainDataOrig);
//
//		//outer LOOCV
//		double[] probs_is = new double[trainDataOrig.getNumRows()];
//		double[] probs_p = new double[trainDataOrig.getNumRows()];
//
//		int[] truth = new int[trainDataOrig.getNumRows()];
//
//		PrintStream out;
//		PrintStream outForAUC;
//		try {
//			File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/TDI-DEG/");
//			dir.mkdirs();
//			String outputFileName = "MelanomaTDI-DEG-Distribution.csv";
//
//			File file = new File(dir, outputFileName);
//			out = new PrintStream(new FileOutputStream(file));
//			outputFileName = "MelanomaTDI-AUROC.csv";
//
//			File fileAUC = new File(dir, outputFileName);
//			outForAUC = new PrintStream(new FileOutputStream(fileAUC));
//		} catch (Exception e) {
//			throw new RuntimeException(e);
//		}
//
//		Map <Key, Double> stats= new HashMap<Key, Double>();
//		Map <String, Double> DEGdist= new HashMap<String, Double>();
//		for (int i = 0; i < trainDataOrig.getNumColumns()-1; i++){
//			DEGdist.put(trainDataOrig.getVariable(i).getName(), 0.0);
//		}
//
//		outForAUC.println("PD1response, population-FGES, instance-specific-FGES");//, DEGs");
//		out.println("DEGs, fraction of occurance in cases");
//		
//		for (int i = 0; i < trainDataOrig.getNumRows(); i++){
//
//			System.out.println("i = " + i);
//			
//			DataSet trainData67 = trainDataOrig.copy();
//			DataSet test67 = trainDataOrig.subsetRows(new int[]{i});
//			trainData67.removeRows(new int[]{i});
//			
//			dagP = GraphUtils.replaceNodes(dagP, trainData67.getVariables());
//			double best_kappa = kappaTuning(trainData67, knowledge, dagP);
//			
//			truth[i] = test67.getInt(0,imP.getNodeIndex(imP.getNode("PD1response")));
//			Graph graphI = learnBNIS(trainData67, test67, best_kappa, graphP, knowledge);
//			probs_is[i]= estimation(trainData67, test67, graphI);
//			
//			
//			//get the p from population model
//			int diseaseIndex_p = imP.getNodeIndex(imP.getNode("PD1response"));
//			double prob_p = estimation_pop(imP, test67, diseaseIndex_p);
//			probs_p[i] = prob_p; 
//			
//			//graph comparison
//			GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
//			int n_a = cmp.getEdgesAdded().size();
//			int n_d = cmp.getEdgesRemoved().size();
//			Key cur_key = new Key(n_a, n_d);
//			if(stats.get(cur_key)!=null)
//				stats.put(cur_key, stats.get(cur_key) + 1.0);
//			else
//				stats.put(cur_key, 1.0);
//	
//			outForAUC.println(test67.getInt(0,diseaseIndex_p) +", " + probs_p[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
//		}
//		double auroc = AUC.measure(truth, probs_is);
//		System.out.println( "AUROC: "+ auroc);
//		for (Key k : stats.keySet()){
//			System.out.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);
//		}
//		System.out.println("-----------------");
//
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
//	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge){
//		// learn the instance-specific model
//		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
//		scoreI.setKAddition(kappa);
//		scoreI.setKDeletion(kappa);
//		scoreI.setKReorientation(kappa);
//		ISFges fgesI = new ISFges(scoreI);
//		fgesI.setKnowledge(knowledge);
//		fgesI.setPopulationGraph(graphP);
//		fgesI.setInitialGraph(graphP);
//		Graph graphI = fgesI.search();
//		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
//		return graphI;
//	}
//	
//	private static double estimation(DataSet trainData, DataSet test, Graph graphI){
//
//		DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
//		Graph dagI = iteratorI.next();
//		dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
//		BayesPm pmI = new BayesPm(dagI);
//
//		DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
//		BayesIm imI = DirichletEstimator.estimate(priorI, trainData);
//
//		int diseaseIndex_i = imI.getNodeIndex(imI.getNode("PD1response"));
////		int truth = test.getInt(0,diseaseIndex_i);
//
//		double p = estimation_pop(imI, test, diseaseIndex_i);
//		
//		return p;
//	}
//	private static double kappaTuning(DataSet trainData67, IKnowledge knowledge, Graph dagP){
//		//inner LOOCV
//		int kvalues = 9;
//		double[] kappa_tune = new double[kvalues+1];
//		double[] auroc_tune = new double[kvalues+1];
//		for (int p = 0; p <= kvalues; p++){	
//			double k_add =  (p+1)/10.0; 
//			double k_delete = k_add;
//			double k_reverse = k_add;
//			kappa_tune[p]= k_add;
////			System.out.println("kappa = " + k_add);
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
//				int diseaseIndex_i = imI.getNodeIndex(imI.getNode("PD1response"));
//				truth_tune [j] = test66.getInt(0,diseaseIndex_i);
//
//				double prob_i = estimation_pop(imI, test66, diseaseIndex_i);
//				probs_is_tune[j] = prob_i; 
//			}
//
//			auroc_tune[p] = AUC.measure(truth_tune, probs_is_tune);
////			System.out.println( "AUROC: "+ auroc_tune[p]);
//		}
//
//		QuickSort.sort(auroc_tune, kappa_tune);
//
//
//		double best_kappa = kappa_tune[kvalues];
//		double best_auroc = auroc_tune[kvalues];
//		System.out.println( "BEST AUROC: "+ best_auroc + ", BEST kappa: " + best_kappa);
//		return best_kappa;
//	}
//	
//	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double structPrior) {
//		BDeuScore scoreP = new BDeuScore(trainDataOrig);
//		scoreP.setStructurePrior(structPrior);
//		Fges fgesP = new Fges (scoreP);
//		fgesP.setKnowledge(knowledge);
//		fgesP.setSymmetricFirstStep(true);
//		Graph graphP = fgesP.search();
//		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
//		return graphP;
//	}
//	private static IKnowledge createKnowledge(DataSet trainDataOrig) {
//		int numVars = trainDataOrig.getNumColumns();
//		IKnowledge knowledge = new Knowledge2();
//		int[] tiers = new int[2];
//		tiers[0] = 0;
//		tiers[1] = 1;
//		for (int i=0 ; i< numVars; i++) {
//			if (!trainDataOrig.getVariable(i).getName().equals("PD1response")){
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
//		TabularDataReader trainDataReader = new VerticalDiscreteTabularDataReader(trainDataFile.toFile(), DelimiterUtils.toDelimiter(delimiter));
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
