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
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFGS_Train_Test {
	
	public static void main(String[] args) {
		
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
//		String dataName = "port";
//		String pathToTrainData = pathToFolder + "PORT/" + dataName + "_train.csv";
//		String pathToTestData = pathToFolder + "PORT/" + dataName + "_test.csv";
//
//		String target = "217.DIREOUT";
//		
		String pathToFolder = "/Users/fattanehjabbari/PPT_project/Data-Oct21/ProcessesData/";
		String dataSubName = "Medications_AllTypes";
		String dataName = "processed_PPT_Dataset2_"+dataSubName+ "_3Month_DeID_disc2_Oct21";
		String pathToTrainData = pathToFolder + dataName + "_train.csv";
		String pathToTestData = pathToFolder + dataName + "_test.csv";
		String target = "Composite_Outcome";
		
		// Read in the data
		DataSet trainData = readData(pathToTrainData);
		DataSet testData = readData(pathToTestData);	
		
		// Create the knowledge
		IKnowledge knowledge = createKnowledge(trainData, target);

		// learn the population model using all training data
		double samplePrior = 1.0;
		double structurePrior = 1.0;
		Graph graphP = BNlearn_pop(trainData, knowledge, samplePrior, structurePrior);
		System.out.println("Pop graph:" + graphP.getEdges());

		double T_plus = 0.8;
		double T_minus = 0.2;
		
		// loop over kappa
		for (int p = 5; p <= 5; p++){

			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
	
			System.out.println("kappa = " + k_add);

			double[] probs_is = new double[testData.getNumRows()];
			double[] probs_p = new double[testData.getNumRows()];
			int[] truth = new int[testData.getNumRows()];
			Map <Key, Double> stats= new HashMap<Key, Double>();
//			PrintStream out;
			PrintStream out, outForAUC, outForPredisctors;
			try {
				File dir = new File( pathToFolder + "/outputs/" + dataName);
				dir.mkdirs();
				
				String outputFileName = dataName + "-FeatureDist-Kappa"+ k_add +".csv";
				File file = new File(dir, outputFileName);
				out = new PrintStream(new FileOutputStream(file));
				
				outputFileName = dataName + "-AUROC-Kappa"+ k_add +".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
				
				outputFileName = dataName + "_Predictors_Kappa"+ k_add +".csv";
				File filePredisctors = new File(dir, outputFileName);
				outForPredisctors = new PrintStream(new FileOutputStream(filePredisctors));
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			outForAUC.println("y, population-FGES, instance-specific-FGES");//, DEGs");

			Map <String, Double> fdist= new HashMap<String, Double>();
			for (int i = 0; i < trainData.getNumColumns(); i++){
				fdist.put(trainData.getVariable(i).getName(), 0.0);
			}
			out.println("features, fraction of occurance in cases");

			// loop over test cases
			for (int i = 0; i < testData.getNumRows(); i++){
				
				DataSet test = testData.subsetRows(new int[]{i});
				
				// learn the IS graph
				Graph graphI = learnBNIS(trainData, test, k_add, graphP, knowledge, samplePrior);

				// compute probability distribution of the target variable
				int targetIndex = trainData.getColumn(trainData.getVariable(target)); //imP.getNodeIndex(imP.getNode(target));
				truth[i] = test.getInt(0, targetIndex);
				
				//get the prob from IS model
				probs_is[i]= estimation(trainData, test, graphI, target);
				List<Node> parents_i = graphI.getParents(graphI.getNode(target));
				for (Node no: parents_i){
					outForPredisctors.print(no.getName() + ",");
					fdist.put(no.getName(), fdist.get(no.getName())+1.0);
				}
				outForPredisctors.println();
				//get the prob from population model
				double prob_p = estimation(trainData, test, graphP, target);
				probs_p[i] = prob_p; 

				//graph comparison
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();

//				if (n_d==1 && n_a==0){
//					System.out.println("case_i: " + i);
//					System.out.println("graph_i: " + graphI.getEdges());
//				
//					BDeuScore score1 = new BDeuScore(trainData);
//					ISBDeuScore score2 = new ISBDeuScore(trainData, test);
//					//				
//					int [] parents_old = new int[1];
//					parents_old[0] = trainData.getColumn(trainData.getVariable("F17"));
//
//					int [] parents_new = new int[2];
//					parents_new[0] = trainData.getColumn(trainData.getVariable("F13"));
//					parents_new[1] = trainData.getColumn(trainData.getVariable("F17"));
//
//					System.out.println("i score with 1 parents" + score2.localScore(0, parents_old, parents_new , new int[0]));
//					System.out.println("i score with 2 parents" + score2.localScore(0, parents_new,  parents_new , new int[0]));
//					System.out.println("i score pop with 2 parents" + score1.localScore(0, parents_new));
//
//					System.out.println("-------------");
//				}

				Key cur_key = new Key(n_a, n_d);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);

				outForAUC.println(test.getInt(0, targetIndex) +", " + probs_p[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
			}
			
			double auroc_p = AUC.measure(truth, probs_p);
			double auroc = AUC.measure(truth, probs_is);
			
			double fcr_p = FCR.measure(truth, probs_p, T_plus, T_minus);
			double fcr = FCR.measure(truth, probs_is, T_plus, T_minus);

			System.out.println( "AUROC_P: "+ auroc_p);
			System.out.println( "AUROC: "+ auroc);
			System.out.println( "FCR_P: "+ fcr_p);
			System.out.println( "FCR: "+ fcr);


			for (Key k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);
			}
			Map<String, Double> sortedfdist = sortByValue(fdist, false);

			for (String k : sortedfdist.keySet()){
				out.println(k + ", " + (fdist.get(k)/testData.getNumRows()));
				
			}
			System.out.println("-----------------");
			outForAUC.close();
			outForPredisctors.close();
			out.close();

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
//	private static double kappaTuning(DataSet trainData, IKnowledge knowledge, Graph dagP, String target){
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
//			double[] probs_is_tune = new double[trainData.getNumRows()];
//			int[] truth_tune = new int[trainData.getNumRows()];
//			for (int j = 0; j < trainData.getNumRows(); j++){
//				DataSet trainData66 = trainData.copy();
//				DataSet test66 = trainData.subsetRows(new int[]{j});
//				trainData66.removeRows(new int[]{j});
//
//				// learn the instance-specific model
//				//				System.out.println("trainData66: "+trainData66.getNumRows());
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
//			//			System.out.println( "AUROC: "+ auroc_tune[p]);
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
	private static IKnowledge createKnowledge(DataSet trainDataOrig, String target) {
		int numVars = trainDataOrig.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainDataOrig.getVariable(i).getName().equals(target)){
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