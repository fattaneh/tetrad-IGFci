package edu.cmu.tetrad.search;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFGS_MB_Train_Test {
	public static void main(String[] args) {
		
		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/Shyam-data/";
		String dataName = "port_uni40";
		String pathToTrainData = pathToFolder + "PORT/" + dataName + "_train.csv";
		String pathToTestData = pathToFolder + "PORT/" + dataName + "_test.csv";

		String target = "217.DIREOUT";
		
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/Shyam-data/";
//		String dataName = "CP";
//		String pathToTrainData = pathToFolder + "Chronic pancreatitis/" + dataName + "_train.csv";
//		String pathToTestData = pathToFolder + "Chronic pancreatitis/" + dataName + "_test.csv";
//
//		String target = "disease";
		
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/Shyam-data/";
//		String pathToKfold =  pathToFolder + "GenIMS/";
//		String dataName = "genims_mortality_4i";		
//		String pathToTrainData = pathToFolder + "GenIMS/" + dataName + "_train.csv";
//		String pathToTestData = pathToFolder + "GenIMS/" + dataName + "_test.csv";
//		String target = "day90_status";
		
		// Read in the data
		DataSet trainData = readData(pathToTrainData);
		DataSet testData = readData(pathToTestData);	

		// Create the knowledge
		IKnowledge knowledge = null;//createKnowledge(trainData, target);

		// learn the population model using all training data
		double samplePrior = 0.1;
		double structurePrior = 1.0;
		Graph graphP = BNlearn_pop(trainData, knowledge, samplePrior, structurePrior);
		PrintStream logFile;
		try {
			File dir = new File( pathToFolder + "/IGES/MB/" + dataName + "/PESS" + samplePrior);
			dir.mkdirs();
			String outputFileName = dataName + "PESS" + samplePrior +"_log.txt";
			File fileAUC = new File(dir, outputFileName);
			logFile = new PrintStream(new FileOutputStream(fileAUC));

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		logFile.println(trainData.getNumRows() +", " + trainData.getNumColumns());
		logFile.println("Pop graph:" + graphP.getEdges());

		double T_plus = 0.9;
		double T_minus = 0.1;
		System.out.println("PESS = " + samplePrior);
		logFile.println("PESS = " + samplePrior);


		
		for (int p = 2; p <= 2; p++){

			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
	
			System.out.println("kappa = " + k_add);

			double[] probs_is = new double[testData.getNumRows()];
			double[] probs_p = new double[testData.getNumRows()];
			int[] truth = new int[testData.getNumRows()];
			double[] llr = new double[testData.getNumRows()];
			double average_llr = 0.0;
			
			Map <KeyMB, Double> stats= new HashMap<KeyMB, Double>();
			PrintStream outForAUC, out;
			try {
				File dir = new File( pathToFolder + "/IGES/MB/" + dataName + "/PESS" + samplePrior);
				dir.mkdirs();
				String outputFileName = dataName + "-AUROC-Kappa"+ k_add + "PESS" + samplePrior +".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
				
				outputFileName = dataName + "FeatureDist-Kappa"+ k_add + "PESS" + samplePrior +".csv";
				File filePredisctors = new File(dir, outputFileName);
				out = new PrintStream(new FileOutputStream(filePredisctors));

			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			System.out.println("kappa = " + k_add);
			logFile.println("kappa = " + k_add);
			
			Map <String, Double> fdist= new HashMap<String, Double>();
			for (int i = 0; i < trainData.getNumColumns(); i++){
				fdist.put(trainData.getVariable(i).getName(), 0.0);
			}

			out.println("features, fraction of occurance in cases");
			outForAUC.println("y, population-FGES, instance-specific-FGES");//, DEGs");

			// loop over test cases
			for (int i = 0; i < testData.getNumRows(); i++){
				if (i % 100 == 0){	
					System.out.println("i = " + i);
				}
				DataSet test = testData.subsetRows(new int[]{i});
				
				// learn the IS graph
				Graph graphI = learnBNIS(trainData, test, k_add, graphP, knowledge, samplePrior);

				// compute probability distribution of the target variable
				int targetIndex = trainData.getColumn(trainData.getVariable(target)); //imP.getNodeIndex(imP.getNode(target));
				truth[i] = test.getInt(0, targetIndex);
				
				//get the prob from IS model
				DagInPatternIterator iterator = new DagInPatternIterator(graphI);
				Graph dagI = iterator.next();
				dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
				Graph mb_i = GraphUtils.markovBlanketDag(dagI.getNode(target), dagI);
				probs_is[i]= estimation(trainData, test, mb_i, target);
				List<Node> mb_nodes = mb_i.getNodes();
				mb_nodes.remove(mb_i.getNode(target));
				for (Node no: mb_nodes){
					fdist.put(no.getName(), fdist.get(no.getName()) + 1.0);
				}
				
				//get the prob from population model
				DagInPatternIterator iteratorP = new DagInPatternIterator(graphP);
				Graph dagP = iteratorP.next();
				dagP = iteratorP.next();
				System.out.println(dagP);
				dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());
				Graph mb_p = GraphUtils.markovBlanketDag(dagP.getNode(target), dagP);
				System.out.println(mb_p);
				probs_p[i] = estimation(trainData, test, mb_p, target);; 

				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setSamplePrior(samplePrior);
				scoreI.setKAddition(k_add);
				scoreI.setKDeletion(k_add);
				scoreI.setKReorientation(k_add);
				ISFges iges = new ISFges(scoreI);				
				iges.setPopulationGraph(graphP);
//				iges.setInitialGraph(graphP);
				List <Node> mb_nodes_all = mb_i.getNodes();
				mb_nodes_all.addAll(mb_p.getNodes());
				Graph mb_i2 = new EdgeListGraph(mb_nodes_all);
				Graph mb_p2 = new EdgeListGraph(mb_nodes_all);
				for (Edge e : mb_i.getEdges()){
					mb_i2.addEdge(e);
				}
				for (Edge e : mb_p.getEdges()){
					mb_p2.addEdge(e);
				}
				llr[i] = iges.scoreDag(mb_i2, graphI) - iges.scoreDag(mb_p2, graphP);
				average_llr += llr[i];
				
				//graph comparison
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(mb_i, mb_p);
				
				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();
				int n_r = cmp.getEdgesReorientedFrom().size();

				KeyMB cur_key = new KeyMB(n_a, n_d, n_r);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);

				outForAUC.println(test.getInt(0, targetIndex) +", " + probs_p[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
			}
			
			double auroc_p = AUC.measure(truth, probs_p);
			double auroc = AUC.measure(truth, probs_is);
			
			System.out.println( "AUROC_P: "+ auroc_p);
			System.out.println( "AUROC: "+ auroc);
//			System.out.println( "FCR_P: "+ fcr_p);
//			System.out.println( "FCR: "+ fcr);
			logFile.println( "AUROC_P: "+ auroc_p);
			logFile.println( "AUROC: "+ auroc);
//			logFile.println( "FCR_P: "+ fcr_p);
//			logFile.println( "FCR: "+ fcr);

			for (KeyMB k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);
				logFile.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);

			}
			System.out.println("average_llr: " + (average_llr/ testData.getNumRows()));
			logFile.println("average_llr: " + (average_llr/ testData.getNumRows()));
			System.out.println("-----------------");
			logFile.println("-----------------");
			
			
			Map<String, Double> sortedfdist = sortByValue(fdist, false);

			for (String k : sortedfdist.keySet()){
				out.println(k + ", " + (fdist.get(k)/testData.getNumRows()));
				
			}
			outForAUC.close();
			out.close();
		}
		logFile.close();
	}

	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge, double samplePrior){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
		scoreI.setSamplePrior(samplePrior);
		scoreI.setKAddition(kappa);
		scoreI.setKDeletion(kappa);
		scoreI.setKReorientation(kappa);
		ISFges fgesI = new ISFges(scoreI);
//		fgesI.setKnowledge(knowledge);
		fgesI.setPopulationGraph(graphP);
		fgesI.setInitialGraph(graphP);
		Graph graphI = fgesI.search();
		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
		return graphI;
	}

	private static double estimation(DataSet trainData, DataSet test, Graph mb, String target){

//		List<Node> mbNodes = mb.getNodes();
//		mbNodes.remove(mb.getNode(target));
//		System.out.println("mb nodes: " + mbNodes);	
//		System.out.println("parents:  " + dag.getParents(dag.getNode(target)));	

		double [] probs = classify(mb, trainData, test, (DiscreteVariable) test.getVariable(target));
		
//		BayesPm pm = new BayesPm(dag);
//		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pm, 1.0);
//		BayesIm im = DirichletEstimator.estimate(prior, trainData);
//		int targetIndex = im.getNodeIndex(im.getNode(target));
//		int[] parents = im.getParents(targetIndex);
//		Arrays.sort(parents);
//
//		int[] values = new int[parents.length];
//		for (int no = 0; no < parents.length; no++){
//			values [no] = test.getInt(0, parents[no]);
//		}
//		double prob = im.getProbability(targetIndex, im.getRowIndex(targetIndex, values), 1);
//		System.out.println("p_mb: " + probs[1]);
//		System.out.println("p_pa: " + prob);
//		return prob;
		return probs[1];
	}

	public static double[] classify(Graph mb, DataSet train, DataSet test, DiscreteVariable targetVariable) {

		List<Node> mbNodes = mb.getNodes();

		//The Markov blanket nodes will correspond to a subset of the variables
		//in the training dataset.  Find the subset dataset.
		DataSet trainDataSubset = train.subsetColumns(mbNodes);

		//To create a Bayes net for the Markov blanket we need the DAG.
		BayesPm bayesPm = new BayesPm(mb);

		//To parameterize the Bayes net we need the number of values
		//of each variable.
		List varsTrain = trainDataSubset.getVariables();

		for (int i1 = 0; i1 < varsTrain.size(); i1++) {
			DiscreteVariable trainingVar = (DiscreteVariable) varsTrain.get(i1);
			bayesPm.setCategories(mbNodes.get(i1), trainingVar.getCategories());
		}

		//Create an updater for the instantiated Bayes net.
		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(bayesPm, 1.0);
		BayesIm bayesIm = DirichletEstimator.estimate(prior, trainDataSubset);

		RowSummingExactUpdater updater = new RowSummingExactUpdater(bayesIm);

		//The subset dataset of the dataset to be classified containing
		//the variables in the Markov blanket.
		List <Node> mb_Nodes_test = new ArrayList<Node> ();
		for (Node n: mbNodes){
			mb_Nodes_test.add(test.getVariable(n.getName()));
		}
		DataSet testSubset = test.subsetColumns(mb_Nodes_test);
		
		//Get the raw data from the dataset to be classified, the number
		//of variables, and the number of cases.
		double[] estimatedProbs = new double[targetVariable.getNumCategories()];

		//The variables in the dataset.
		List<Node> varsClassify = testSubset.getVariables();

		//For each case in the dataset to be classified compute the estimated
		//value of the target variable and increment the appropriate element
		//of the crosstabulation array.

		//Create an Evidence instance for the instantiated Bayes net
		//which will allow that updating.
		Proposition proposition = Proposition.tautology(bayesIm);

		//Restrict all other variables to their observed values in
		//this case.
		int numMissing = 0;

		for (int testIndex = 0; testIndex < varsClassify.size(); testIndex++) {
			DiscreteVariable var = (DiscreteVariable) varsClassify.get(testIndex);

			// If it's the target, ignore it.
			if (var.equals(targetVariable)) {
				continue;
			}

			int trainIndex = proposition.getNodeIndex(var.getName());

			// If it's not in the train subset, ignore it.
			if (trainIndex == -99) {
				continue;
			}

			int testValue = testSubset.getInt(0, testIndex);

			if (testValue == -99) {
				numMissing++;
			} else {
				proposition.setCategory(trainIndex, testValue);
			}
		}

		Evidence evidence = Evidence.tautology(bayesIm);
		evidence.getProposition().restrictToProposition(proposition);
		updater.setEvidence(evidence);

		// for each possible value of target compute its probability in
		// the updated Bayes net.  Select the value with the highest
		// probability as the estimated getValue.
		int targetIndex = proposition.getNodeIndex(targetVariable.getName());


		for (int category = 0; category < targetVariable.getNumCategories(); category++) {
			double marginal = updater.getMarginal(targetIndex, category);
			estimatedProbs [category] = marginal;
		}

		return estimatedProbs;
	}

	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double samplePrior, double structurePrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setSamplePrior(samplePrior);
		scoreP.setStructurePrior(structurePrior);
		Fges fgesP = new Fges (scoreP);
//		fgesP.setKnowledge(knowledge);
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
