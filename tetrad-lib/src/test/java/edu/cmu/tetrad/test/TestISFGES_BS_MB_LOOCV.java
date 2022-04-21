package edu.cmu.tetrad.test;

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
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

//class KeyMB {
//
//	public final int n_a;
//	public final int n_d;
//	public final int n_r;
//
//
//	public KeyMB(final int n_a, final int n_d, final int n_r) {
//		this.n_a = n_a;
//		this.n_d = n_d;
//		this.n_r = n_r;
//	}
//	@Override
//	public boolean equals (final Object O) {
//		if (!(O instanceof KeyMB)) return false;
//		if (((KeyMB) O).n_a != n_a) return false;
//		if (((KeyMB) O).n_d != n_d) return false;
//		if (((KeyMB) O).n_r != n_r) return false;
//		return true;
//	}
//	 @Override
//	 public int hashCode() {
//		 return this.n_a ^ this.n_d ^ this.n_r ;
//	 }
//	 public String print(KeyMB key){
//		return "("+key.n_a +", "+ key.n_d +", "+ key.n_r + ")";
//	 }
//
//}
public class TestISFGES_BS_MB_LOOCV {
	public static void main(String[] args) {

		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
		//		String dataName = "genims_mortality_4i";
		//		String pathToData = pathToFolder + "GenIMS/" + dataName + ".csv";
		//		String target = "day90_status";

		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
		//		String dataName = "genims_sepsis_5i";
		//		String pathToData = pathToFolder + "GenIMS/" + dataName + ".csv";
		//		String target = "everss";

		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
		//		String dataName = "port_all";
		//		String pathToData = pathToFolder + "PORT/" + dataName + ".csv";
		//		String target = "217.DIREOUT";


		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/UCI/";
		//		String dataName = "breast-cancer.data_imputed";
		//		String pathToData = pathToFolder + dataName + ".csv";
		//		String target = "y";

		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/UCI/";
		//		String dataName = "SPECT.train";
		//		String pathToData = pathToFolder + dataName + ".csv";
		//		String target = "y";

		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/TDI_DEG/";
		//		String dataName = "DEGmatrix.UPMCcell4greg.TDIDEGfeats";
		//		String pathToData = pathToFolder + "/" + dataName + ".csv";
		//		String target = "PD1response";

		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/lung_cancer/data/";
		String dataName = "LCMR_Processed4_NA_surv";
		String pathToData = pathToFolder + dataName + ".csv";
		String target = "Survive1";

		// Read in the data
		DataSet trainDataOrig = readData(pathToData);

		int numBootstraps = 6;

		// learn the population model using all training data
		double samplePrior = 1.0;
		double structurePrior = 1.0;
		Graph graphP = BNlearn_pop(trainDataOrig, samplePrior, structurePrior);

		List <Graph> graphsP = new ArrayList<Graph>();
		List<DataSet> allBS = new ArrayList<DataSet>();
		for (int bs = 0; bs < numBootstraps; bs++){
			DataSet bsData = DataUtils.getBootstrapSample(trainDataOrig, trainDataOrig.getNumRows());
			allBS.add(bsData);
			Graph graphP_bs = BNlearn_pop(bsData, samplePrior, structurePrior);
			graphsP.add(graphP_bs);
			//			System.out.println("Pop graph:" + graphP.getEdges());
		}
		
		double T_plus = 0.9;
		double T_minus = 0.1;

		System.out.println("PESS = " + samplePrior);


		for (int p = 10; p <= 10; p++){

			double k_add =  p/10.0; 

			double[][] probs_is = new double[trainDataOrig.getNumRows()][numBootstraps];
			double[][] probs_p = new double[trainDataOrig.getNumRows()][numBootstraps];
			int[] truth = new int[trainDataOrig.getNumRows()];
	
			PrintStream outForAUC_Pop, outForAUC_IS;
			try {
				File dir = new File( pathToFolder + "/outputs/MB_BS" + numBootstraps + "/"+ dataName + "/PESS" + samplePrior);
				dir.mkdirs();
				String outputFileName = dataName + "-AUROC-Kappa"+ k_add + "PESS" + samplePrior +"-pop.csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC_Pop = new PrintStream(new FileOutputStream(fileAUC));
				outputFileName = dataName + "-AUROC-Kappa"+ k_add + "PESS" + samplePrior +"-Is.csv";
				fileAUC = new File(dir, outputFileName);
				outForAUC_IS = new PrintStream(new FileOutputStream(fileAUC));
				outputFileName = dataName + "FeatureDist-Kappa"+ k_add + "PESS" + samplePrior +".csv";
		
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			System.out.println("kappa = " + k_add);
	
			//LOOCV loop
			for (int i = 0; i < trainDataOrig.getNumRows(); i++){
				if (i % 10==0)
					System.out.println("i: " + i);
				

				DataSet trainData = trainDataOrig.copy();
				DataSet test = trainDataOrig.subsetRows(new int[]{i});
				trainData.removeRows(new int[]{i});

				List <Graph> graphsI_bs = new ArrayList<Graph>();
				for (int bs = 0; bs < numBootstraps; bs++){
					DataSet bsData = DataUtils.getBootstrapSample(trainData, trainData.getNumRows());


					// learn the IS graph
					Graph graphI = learnBNIS(bsData, test, k_add, graphP, samplePrior);

					// compute probability distribution of the target variable
					int targetIndex = bsData.getColumn(bsData.getVariable(target)); //imP.getNodeIndex(imP.getNode(target));
					truth[i] = test.getInt(0, targetIndex);

					//get the prob from IS model
					DagInPatternIterator iterator = new DagInPatternIterator(graphI);
					Graph dagI = iterator.next();
					dagI = GraphUtils.replaceNodes(dagI, bsData.getVariables());
					Graph mb_i = GraphUtils.markovBlanketDag(dagI.getNode(target), dagI);
					probs_is[i][bs] = estimation(bsData, test, (Dag) mb_i, target);

					//get the prob from population model
					DagInPatternIterator iteratorP = new DagInPatternIterator(graphsP.get(bs));
					Graph dagP = iteratorP.next();
					dagP = GraphUtils.replaceNodes(dagP, allBS.get(bs).getVariables());
					Graph mb_p = GraphUtils.markovBlanketDag(dagP.getNode(target), dagP);
					probs_p[i][bs] = estimation(allBS.get(bs), test, (Dag) mb_p, target);

				}

			}
			// write the results to output files
			// write the first line , i.e. header
			outForAUC_Pop.print("y");
			outForAUC_IS.print("y");

			for (int bs = 0; bs < numBootstraps; bs ++){
				outForAUC_Pop.print(",P_Pop_" + bs);
				outForAUC_IS.print(",P_IS_" + bs);
			}
			outForAUC_Pop.println();
			outForAUC_IS.println();

			// write the probs
			for (int i = 0; i < trainDataOrig.getNumRows(); i++){
				outForAUC_Pop.print(truth[i]);
				outForAUC_IS.print(truth[i]);
				for (int bs = 0; bs < numBootstraps; bs ++){
					outForAUC_Pop.print(","+ probs_p[i][bs]);
					outForAUC_IS.print(","+ probs_is[i][bs]);
				}
				outForAUC_Pop.println();
				outForAUC_IS.println();
			}
			outForAUC_Pop.close();
			outForAUC_IS.close();
			System.out.println("-----------------");
		
		}
	}

	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, double samplePrior){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
		scoreI.setSamplePrior(samplePrior);
		scoreI.setKAddition(kappa);
		scoreI.setKDeletion(kappa);
		scoreI.setKReorientation(kappa);
		ISFges fgesI = new ISFges(scoreI);
		fgesI.setPopulationGraph(graphP);
		fgesI.setInitialGraph(graphP);
		Graph graphI = fgesI.search();
		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
		return graphI;
	}

	private static double estimation(DataSet trainData, DataSet test, Dag mb, String target){

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

	public static double[] classify(Dag mb, DataSet train, DataSet test, DiscreteVariable targetVariable) {

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
		DataSet testSubset = test.subsetColumns(mbNodes);

		//Get the raw data from the dataset to be classified, the number
		//of variables, and the number of cases.
		int numCases = testSubset.getNumRows();
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


	private static Graph BNlearn_pop(DataSet trainDataOrig, double samplePrior, double structurePrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setSamplePrior(samplePrior);
		scoreP.setStructurePrior(structurePrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setSymmetricFirstStep(true);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		return graphP;
	}
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